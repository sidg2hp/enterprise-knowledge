"""Optional, read-only Neo4j context adapter.

The local JSON retriever remains the source of truth for offline operation.  This
module deliberately imports the Neo4j driver only when a complete configuration
is present, so the backend can be installed and tested without Neo4j (or its
Python package).  Graph results are context only; callers must still ground
their answer in the local evidence returned by :mod:`knowledge`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Mapping


_WRITE_KEYWORDS = re.compile(
    r"\b(?:CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|FOREACH|CALL\s+DBMS)\b",
    re.IGNORECASE,
)
_READ_START = re.compile(r"^(?:MATCH|OPTIONAL\s+MATCH|UNWIND|WITH|RETURN)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Neo4jConfig:
    """Connection settings read from environment variables."""

    uri: str = ""
    username: str = ""
    password: str = ""
    database: str = "neo4j"
    timeout_seconds: float = 2.0

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        timeout = os.getenv("NEO4J_TIMEOUT_SECONDS", "2")
        try:
            timeout_seconds = max(0.1, float(timeout))
        except ValueError:
            timeout_seconds = 2.0
        return cls(
            uri=os.getenv("NEO4J_URI", "").strip(),
            username=os.getenv("NEO4J_USERNAME", "").strip(),
            password=os.getenv("NEO4J_PASSWORD", ""),
            database=os.getenv("NEO4J_DATABASE", "neo4j").strip() or "neo4j",
            timeout_seconds=timeout_seconds,
        )

    @property
    def complete(self) -> bool:
        return bool(self.uri and self.username and self.password)


def is_safe_read_query(cypher: str) -> bool:
    """Return whether a Cypher statement is an uncomplicated read.

    The adapter rejects comments, multiple statements, write clauses, and
    procedure calls.  This is intentionally stricter than Neo4j's read
    transaction API: an allow-list is safer when a caller supplies query text.
    """

    normalized = " ".join(cypher.strip().split())
    if not normalized or ";" in normalized or "//" in normalized or "/*" in normalized:
        return False
    if _WRITE_KEYWORDS.search(normalized) or not _READ_START.match(normalized):
        return False
    return True


class Neo4jGraphAdapter:
    """Best-effort read-only graph context lookup with a disabled fallback."""

    def __init__(self, config: Neo4jConfig | None = None):
        self.config = config or Neo4jConfig.from_env()
        self._driver: Any | None = None
        self._disabled_reason: str | None = None

    @classmethod
    def from_env(cls) -> "Neo4jGraphAdapter":
        return cls(Neo4jConfig.from_env())

    @property
    def configured(self) -> bool:
        return self.config.complete

    @property
    def enabled(self) -> bool:
        return self.config.complete and self._disabled_reason is None

    def status(self) -> dict[str, Any]:
        if not self.config.complete:
            return {"enabled": False, "configured": False, "reason": "not_configured"}
        if self._disabled_reason:
            return {"enabled": False, "configured": True, "reason": self._disabled_reason}
        return {"enabled": True, "configured": True, "reason": None}

    def _ensure_driver(self) -> Any | None:
        if not self.config.complete or self._disabled_reason:
            return None
        if self._driver is not None:
            return self._driver
        try:
            from neo4j import GraphDatabase  # type: ignore[import-not-found]

            self._driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                connection_timeout=self.config.timeout_seconds,
            )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            self._disabled_reason = "driver_unavailable"
            return None
        return self._driver

    def read(self, cypher: str, parameters: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute one validated read query, returning rows or an empty list."""

        if not is_safe_read_query(cypher):
            raise ValueError("only single-statement read-only Cypher is allowed")
        driver = self._ensure_driver()
        if driver is None:
            return []
        try:
            with driver.session(database=self.config.database) as session:
                rows = session.execute_read(
                    lambda tx: [dict(record) for record in tx.run(cypher, parameters or {})]
                )
        except (OSError, RuntimeError, TimeoutError, ValueError):
            # A graph is an optional context source.  A network/auth/schema
            # issue must never make offline lexical retrieval fail.
            self._disabled_reason = "query_unavailable"
            return []
        return rows if isinstance(rows, list) else []

    def search_context(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """Find text-bearing graph nodes using a parameterized read query."""

        if not query.strip() or not self.config.complete:
            return []
        safe_limit = min(max(int(limit), 1), 20)
        cypher = (
            "MATCH (node) "
            "WHERE any(key IN keys(node) WHERE toString(node[key]) CONTAINS $query) "
            "RETURN labels(node) AS labels, properties(node) AS properties "
            "LIMIT $limit"
        )
        rows = self.read(cypher, {"query": query.strip(), "limit": safe_limit})
        contexts: list[dict[str, Any]] = []
        for row in rows:
            properties = row.get("properties", {})
            if not isinstance(properties, dict):
                properties = {}
            labels = row.get("labels", [])
            if not isinstance(labels, list):
                labels = []
            contexts.append({"labels": labels, "properties": properties})
        return contexts

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None


def get_graph_adapter() -> Neo4jGraphAdapter:
    """Create a per-request adapter so environment changes are observable in tests."""

    return Neo4jGraphAdapter.from_env()


__all__ = [
    "Neo4jConfig",
    "Neo4jGraphAdapter",
    "get_graph_adapter",
    "is_safe_read_query",
]
