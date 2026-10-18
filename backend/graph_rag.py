"""Deterministic graph retrieval for locally ingested text documents.

The graph index intentionally has no required third-party dependencies.  It
turns the stable document/chunk records produced by :mod:`knowledge` into a
small JSON graph, which makes graph retrieval reproducible in offline CI.  A
Neo4j writer is provided as an optional projection of the same records; local
indexing never depends on the writer being configured or reachable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Mapping


_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)?)(?:\s+[A-Z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)?){0,3}\b"
)
_INTERNAL_ENTITY_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[A-Z][A-Za-z0-9]+)+\b")
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9_-]{1,}\b")
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]*\)")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)

_ENTITY_STOP_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "but",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "when",
    "where",
    "which",
    "who",
    "with",
}
_KNOWN_TECHNOLOGIES = {
    "api": "API",
    "aws": "AWS",
    "bm25": "BM25",
    "docker": "Docker",
    "fastapi": "FastAPI",
    "graph rag": "Graph RAG",
    "kubernetes": "Kubernetes",
    "langgraph": "LangGraph",
    "markdown": "Markdown",
    "neo4j": "Neo4j",
    "postgres": "Postgres",
    "postgresql": "PostgreSQL",
    "python": "Python",
    "txt": "TXT",
}

_RELATION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("reports_to", r"\breports?\s+to\b"),
    ("works_at", r"\bworks?\s+(?:at|for)\b|\bemployed\s+by\b"),
    ("uses", r"\buses?\b|\busing\b|\bbuilt\s+with\b|\bwritten\s+in\b|\bdeveloped\s+with\b"),
    ("depends_on", r"\bdepends?\s+on\b"),
    ("requires", r"\brequires?\b|\bneeds?\b"),
    ("supports", r"\bsupports?\b|\benables?\b"),
    ("owns", r"\bowns?\b|\bmanages?\b|\bleads?\b"),
    ("collaborates_with", r"\bworks?\s+with\b|\bcollaborates?\s+with\b|\bpartners?\s+with\b"),
    ("located_in", r"\blocated\s+in\b|\bbased\s+in\b"),
    ("part_of", r"\bpart\s+of\b|\bmember\s+of\b"),
    ("related_to", r"\brelated\s+to\b|\bassociated\s+with\b"),
)
_RELATION_REGEXES = tuple((name, re.compile(pattern, re.IGNORECASE)) for name, pattern in _RELATION_PATTERNS)


def _entity_id(name: str) -> str:
    return "entity_" + hashlib.sha256(name.casefold().encode("utf-8")).hexdigest()[:20]


def _relation_id(source: str, target: str, relation_type: str) -> str:
    value = f"{source}\0{relation_type}\0{target}"
    return "relation_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _canonical_name(value: str) -> str:
    return " ".join(value.strip().strip("#`*_[](){}<>,.;:!?\"").split())


def _entity_type(name: str) -> str:
    key = name.casefold()
    if key in _KNOWN_TECHNOLOGIES or name.isupper() or re.search(r"[a-z][A-Z]", name):
        return "TECHNOLOGY"
    words = name.split()
    if len(words) >= 2 and all(word[:1].isupper() for word in words):
        return "PERSON_OR_ORGANIZATION"
    return "CONCEPT"


def _add_candidate(candidates: dict[str, dict[str, Any]], name: str, start: int, end: int) -> None:
    canonical = _canonical_name(name)
    if not canonical:
        return
    words = [word.casefold() for word in _TOKEN_RE.findall(canonical)]
    if not words or (len(words) == 1 and words[0] in _ENTITY_STOP_WORDS):
        return
    key = canonical.casefold()
    current = candidates.get(key)
    record = {
        "entity_id": _entity_id(canonical),
        "name": canonical,
        "type": _entity_type(canonical),
        "start_char": start,
        "end_char": end,
    }
    if current is None or (start, end, canonical) < (
        current["start_char"],
        current["end_char"],
        current["name"],
    ):
        candidates[key] = record


def extract_entities(text: str) -> list[dict[str, Any]]:
    """Extract stable entity records from one text chunk.

    The extractor uses Markdown links, title-cased phrases, acronyms, known
    technology names, and internal-capitalization names.  It deliberately does
    not use a model, so the same chunk always produces the same entities.
    """

    if not isinstance(text, str) or not text.strip():
        return []
    candidates: dict[str, dict[str, Any]] = {}

    # Link labels and headings are useful identifiers even when their words are
    # not title-cased in the body.
    for match in _MARKDOWN_LINK_RE.finditer(text):
        _add_candidate(candidates, match.group(0).split("]", 1)[0].lstrip("["), match.start(), match.end())
    for match in re.finditer(r"(?m)^\s{0,3}#{1,6}\s+([^\n#]+)", text):
        _add_candidate(candidates, match.group(1), match.start(1), match.end(1))

    for regex in (_ENTITY_RE, _INTERNAL_ENTITY_RE, _ACRONYM_RE):
        for match in regex.finditer(text):
            _add_candidate(candidates, match.group(0), match.start(), match.end())

    lowered = text.casefold()
    for key, display in _KNOWN_TECHNOLOGIES.items():
        for match in re.finditer(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", lowered):
            _add_candidate(candidates, display, match.start(), match.end())

    return sorted(candidates.values(), key=lambda item: (item["start_char"], item["end_char"], item["name"].casefold()))


def _relation_type(between: str) -> str:
    for relation_type, regex in _RELATION_REGEXES:
        if regex.search(between):
            return relation_type
    return "mentions"


def extract_relations(text: str, entities: Iterable[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Extract directed, deterministic relations between entities in a chunk."""

    if not isinstance(text, str) or not text.strip():
        return []
    entity_records = list(entities) if entities is not None else extract_entities(text)
    valid = [item for item in entity_records if item.get("entity_id") and item.get("name")]
    relations: dict[tuple[str, str, str], dict[str, Any]] = {}
    for sentence in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", text):
        start, end = sentence.start(), sentence.end()
        mentions = [
            item
            for item in valid
            if start <= int(item.get("start_char", -1)) < end
        ]
        mentions.sort(key=lambda item: (int(item.get("start_char", 0)), str(item.get("name", "")).casefold()))
        for source, target in zip(mentions, mentions[1:]):
            source_end = int(source.get("end_char", source["start_char"]))
            target_start = int(target.get("start_char", target["end_char"]))
            relation_type = _relation_type(text[source_end:target_start])
            key = (str(source["entity_id"]), str(target["entity_id"]), relation_type)
            relations[key] = {
                "relation_id": _relation_id(*key),
                "source_entity_id": key[0],
                "target_entity_id": key[1],
                "relation_type": relation_type,
                "start_char": int(source.get("start_char", 0)),
                "end_char": int(target.get("end_char", 0)),
            }
    return sorted(relations.values(), key=lambda item: item["relation_id"])


def _empty_state() -> dict[str, Any]:
    return {"schema_version": 1, "documents": {}, "chunks": {}, "entities": {}, "relations": {}}


def _unique_sorted(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values if value})


class GraphRAGIndex:
    """Atomic JSON graph index with citation-aware graph retrieval."""

    def __init__(self, path: str | Path | None = None, neo4j_writer: "Neo4jGraphWriter | None" = None):
        configured = path or os.getenv("GRAPH_RAG_INDEX_PATH") or Path(__file__).resolve().parent / "data" / "graph.json"
        self.path = Path(configured)
        self.neo4j_writer = neo4j_writer
        self._lock = RLock()
        self._load_error: str | None = None

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return _empty_state()
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            if not isinstance(value, dict) or value.get("schema_version") != 1:
                raise ValueError("unsupported graph index schema")
            state = _empty_state()
            for key in ("documents", "chunks", "entities", "relations"):
                if isinstance(value.get(key), dict):
                    state[key] = value[key]
            self._load_error = None
            return state
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            # A damaged optional graph must not take down lexical ingestion or
            # querying.  The next successful upsert replaces it atomically.
            self._load_error = type(exc).__name__
            return _empty_state()

    def _write(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="graph-rag-", suffix=".json", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _document_for_storage(document: Mapping[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "document_id": str(document["document_id"]),
            "filename": str(document.get("filename", "")),
            "extension": str(document.get("extension", "")),
            "content_sha256": str(document.get("content_sha256", "")),
            "chunks": chunks,
        }

    def _remove_document_from_state(self, state: dict[str, Any], document_id: str) -> None:
        old_document = state["documents"].pop(document_id, None)
        old_chunk_ids = {
            str(chunk.get("chunk_id"))
            for chunk in (old_document or {}).get("chunks", [])
            if chunk.get("chunk_id")
        }
        old_chunk_ids.update(
            chunk_id for chunk_id, chunk in state["chunks"].items() if chunk.get("document_id") == document_id
        )
        for chunk_id in old_chunk_ids:
            state["chunks"].pop(chunk_id, None)
        for entity_id in list(state["entities"]):
            entity = state["entities"][entity_id]
            entity["chunk_ids"] = _unique_sorted(set(entity.get("chunk_ids", [])) - old_chunk_ids)
            entity["citations"] = _unique_sorted(
                citation
                for citation in entity.get("citations", [])
                if citation in {chunk.get("citation") for chunk in state["chunks"].values()}
            )
            if not entity["chunk_ids"]:
                del state["entities"][entity_id]
        for relation_id in list(state["relations"]):
            relation = state["relations"][relation_id]
            relation["chunk_ids"] = _unique_sorted(set(relation.get("chunk_ids", [])) - old_chunk_ids)
            relation["citations"] = _unique_sorted(
                citation
                for citation in relation.get("citations", [])
                if citation in {chunk.get("citation") for chunk in state["chunks"].values()}
            )
            if not relation["chunk_ids"]:
                del state["relations"][relation_id]

    def index_document(self, document: Mapping[str, Any]) -> dict[str, Any]:
        """Upsert one :func:`knowledge.ingest_document` result and its graph."""

        document_id = str(document.get("document_id", "")).strip()
        raw_chunks = document.get("chunks")
        if not document_id or not isinstance(raw_chunks, list):
            raise ValueError("document must contain a document_id and chunks list")

        chunks: list[dict[str, Any]] = []
        extracted_entities: dict[str, dict[str, Any]] = {}
        extracted_relations: dict[str, dict[str, Any]] = {}
        for index, raw_chunk in enumerate(raw_chunks):
            if not isinstance(raw_chunk, Mapping):
                raise ValueError("document chunks must be objects")
            chunk_id = str(raw_chunk.get("chunk_id") or f"{document_id}_chunk_{index:04d}")
            text = str(raw_chunk.get("text", ""))
            citation = str(raw_chunk.get("citation") or f"{document_id}#chunk-{index}")
            chunk = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "chunk_index": int(raw_chunk.get("chunk_index", index)),
                "text": text,
                "citation": citation,
            }
            chunks.append(chunk)
            entities = extract_entities(text)
            relations = extract_relations(text, entities)
            for entity in entities:
                entity_id = entity["entity_id"]
                stored = extracted_entities.setdefault(
                    entity_id,
                    {"entity_id": entity_id, "name": entity["name"], "type": entity["type"], "chunk_ids": [], "citations": []},
                )
                stored["chunk_ids"].append(chunk_id)
                stored["citations"].append(citation)
            for relation in relations:
                relation_id = relation["relation_id"]
                stored_relation = extracted_relations.setdefault(
                    relation_id,
                    {
                        "relation_id": relation_id,
                        "source_entity_id": relation["source_entity_id"],
                        "target_entity_id": relation["target_entity_id"],
                        "relation_type": relation["relation_type"],
                        "chunk_ids": [],
                        "citations": [],
                    },
                )
                stored_relation["chunk_ids"].append(chunk_id)
                stored_relation["citations"].append(citation)

        with self._lock:
            state = self._read()
            self._remove_document_from_state(state, document_id)
            state["documents"][document_id] = self._document_for_storage(document, chunks)
            for chunk in chunks:
                state["chunks"][chunk["chunk_id"]] = chunk
            for entity in extracted_entities.values():
                entity["chunk_ids"] = _unique_sorted(entity["chunk_ids"])
                entity["citations"] = _unique_sorted(entity["citations"])
                existing = state["entities"].get(entity["entity_id"])
                if existing:
                    existing.update({key: entity[key] for key in ("name", "type")})
                    existing["chunk_ids"] = _unique_sorted(
                        [*existing.get("chunk_ids", []), *entity["chunk_ids"]]
                    )
                    existing["citations"] = _unique_sorted(
                        [*existing.get("citations", []), *entity["citations"]]
                    )
                else:
                    state["entities"][entity["entity_id"]] = entity
            for relation in extracted_relations.values():
                relation["chunk_ids"] = _unique_sorted(relation["chunk_ids"])
                relation["citations"] = _unique_sorted(relation["citations"])
                existing = state["relations"].get(relation["relation_id"])
                if existing:
                    existing.update(
                        {
                            key: relation[key]
                            for key in ("source_entity_id", "target_entity_id", "relation_type")
                        }
                    )
                    existing["chunk_ids"] = _unique_sorted(
                        [*existing.get("chunk_ids", []), *relation["chunk_ids"]]
                    )
                    existing["citations"] = _unique_sorted(
                        [*existing.get("citations", []), *relation["citations"]]
                    )
                else:
                    state["relations"][relation["relation_id"]] = relation
            self._write(state)

        if self.neo4j_writer is not None:
            try:
                self.neo4j_writer.upsert(
                    state["documents"][document_id],
                    list(extracted_entities.values()),
                    list(extracted_relations.values()),
                )
            except Exception:  # pragma: no cover - defensive optional integration boundary
                pass
        return state["documents"][document_id]

    upsert_document = index_document

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            state = self._read()
            if document_id not in state["documents"] and not any(
                chunk.get("document_id") == document_id for chunk in state["chunks"].values()
            ):
                return False
            self._remove_document_from_state(state, document_id)
            self._write(state)
        return True

    delete = delete_document

    def _seed_entities(self, state: Mapping[str, Any], query: str) -> list[tuple[str, float]]:
        query_key = query.casefold().strip()
        query_tokens = {token.casefold() for token in _TOKEN_RE.findall(query_key)}
        scored: list[tuple[float, str]] = []
        for entity_id, entity in state["entities"].items():
            name = str(entity.get("name", ""))
            name_key = name.casefold()
            name_tokens = {token.casefold() for token in _TOKEN_RE.findall(name_key)}
            score = 0.0
            if name_key and name_key in query_key:
                score += 3.0
            overlap = len(name_tokens & query_tokens)
            score += overlap / max(1, len(name_tokens))
            if score:
                scored.append((score, str(entity_id)))
        return [(entity_id, score) for score, entity_id in sorted(scored, key=lambda item: (-item[0], item[1]))]

    def retrieve(self, query: str, *, limit: int = 5, max_hops: int = 2) -> dict[str, Any]:
        """Return graph paths and chunk context, each with source citations."""

        empty = {"query": query if isinstance(query, str) else "", "paths": [], "contexts": [], "citations": []}
        if not isinstance(query, str) or not query.strip():
            return empty
        safe_limit = min(max(int(limit), 1), 50)
        safe_hops = min(max(int(max_hops), 0), 5)
        with self._lock:
            state = self._read()
        seeds = self._seed_entities(state, query)
        if not seeds:
            return empty

        adjacency: dict[str, list[dict[str, Any]]] = {}
        for relation in state["relations"].values():
            source = str(relation.get("source_entity_id", ""))
            target = str(relation.get("target_entity_id", ""))
            adjacency.setdefault(source, []).append(relation)
            # Undirected traversal lets a query find a citing chunk from either
            # end while preserving the relation's original direction in output.
            adjacency.setdefault(target, []).append({**relation, "_reverse": True})

        paths: list[dict[str, Any]] = []
        seen_paths: set[tuple[str, ...]] = set()
        for seed_id, seed_score in seeds[:safe_limit]:
            queue: list[tuple[str, list[str], list[str]]] = [(seed_id, [seed_id], [])]
            while queue:
                current_id, entity_ids, relation_ids = queue.pop(0)
                key = tuple(entity_ids)
                if key not in seen_paths:
                    path_relations = [state["relations"][rid] for rid in relation_ids if rid in state["relations"]]
                    citations = _unique_sorted(citation for relation in path_relations for citation in relation.get("citations", []))
                    paths.append(
                        {
                            "seed_entity_id": seed_id,
                            "score": seed_score,
                            "entity_ids": entity_ids,
                            "entities": [state["entities"][entity_id] for entity_id in entity_ids if entity_id in state["entities"]],
                            "relations": path_relations,
                            "citations": citations,
                        }
                    )
                    seen_paths.add(key)
                if len(relation_ids) >= safe_hops:
                    continue
                for relation in sorted(adjacency.get(current_id, []), key=lambda item: str(item.get("relation_id", ""))):
                    next_id = str(relation.get("target_entity_id")) if not relation.get("_reverse") else str(relation.get("source_entity_id"))
                    if next_id in entity_ids or next_id not in state["entities"]:
                        continue
                    queue.append((next_id, entity_ids + [next_id], relation_ids + [str(relation["relation_id"])]))

        paths.sort(key=lambda item: (-float(item["score"]), len(item["entity_ids"]), tuple(item["entity_ids"])))
        contexts_by_chunk: dict[str, dict[str, Any]] = {}
        for path in paths:
            for entity in path["entities"]:
                for chunk_id in entity.get("chunk_ids", []):
                    chunk = state["chunks"].get(chunk_id)
                    if not chunk:
                        continue
                    context = contexts_by_chunk.setdefault(
                        chunk_id,
                        {
                            "chunk_id": chunk["chunk_id"],
                            "document_id": chunk["document_id"],
                            "text": chunk["text"],
                            "citation": chunk["citation"],
                            "source_chunk_citations": [chunk["citation"]],
                            "paths": [],
                        },
                    )
                    if path not in context["paths"]:
                        context["paths"].append(path)
        contexts = sorted(contexts_by_chunk.values(), key=lambda item: (item["chunk_id"]))[:safe_limit]
        citations = _unique_sorted(item["citation"] for item in contexts)
        return {"query": query, "paths": paths, "contexts": contexts, "citations": citations}

    query = retrieve

    def search(self, query: str, *, limit: int = 5, max_hops: int = 2) -> list[dict[str, Any]]:
        """Return only context records for callers that need a retriever shape."""

        return self.retrieve(query, limit=limit, max_hops=max_hops)["contexts"]

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = self._read()
        return {
            "enabled": True,
            "path": str(self.path),
            "document_count": len(state["documents"]),
            "chunk_count": len(state["chunks"]),
            "entity_count": len(state["entities"]),
            "relation_count": len(state["relations"]),
            "load_error": self._load_error,
        }


# These statements are constants by design: values, including the relation
# type, are always parameters and are never interpolated into Cypher.
UPSERT_DOCUMENT_CYPHER = (
    "MERGE (document:Document {document_id: $document_id}) "
    "SET document.filename = $filename, document.extension = $extension, "
    "document.content_sha256 = $content_sha256"
)
UPSERT_CHUNK_CYPHER = (
    "MERGE (document:Document {document_id: $document_id}) "
    "MERGE (chunk:Chunk {chunk_id: $chunk_id}) "
    "SET chunk.document_id = $document_id, chunk.chunk_index = $chunk_index, "
    "chunk.text = $text, chunk.citation = $citation "
    "MERGE (document)-[:HAS_CHUNK]->(chunk)"
)
UPSERT_ENTITY_CYPHER = (
    "MERGE (entity:Entity {entity_id: $entity_id}) "
    "SET entity.name = $name, entity.entity_type = $entity_type, "
    "entity.chunk_ids = $chunk_ids, entity.citations = $citations"
)
UPSERT_RELATION_CYPHER = (
    "MATCH (source:Entity {entity_id: $source_entity_id}), "
    "(target:Entity {entity_id: $target_entity_id}) "
    "MERGE (source)-[relation:RELATED_TO {relation_id: $relation_id}]->(target) "
    "SET relation.relation_type = $relation_type, relation.chunk_ids = $chunk_ids, "
    "relation.citations = $citations"
)


class Neo4jGraphWriter:
    """Optional Neo4j projection using fixed, parameterized Cypher."""

    def __init__(
        self,
        driver: Any | None = None,
        *,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str = "neo4j",
    ):
        self._driver = driver
        self.uri = (uri if uri is not None else os.getenv("NEO4J_URI", "")).strip()
        self.username = (username if username is not None else os.getenv("NEO4J_USERNAME", "")).strip()
        self.password = password if password is not None else os.getenv("NEO4J_PASSWORD", "")
        self.database = database.strip() or "neo4j"
        self._disabled_reason: str | None = None

    @classmethod
    def from_env(cls) -> "Neo4jGraphWriter":
        return cls(database=os.getenv("NEO4J_DATABASE", "neo4j"))

    @property
    def configured(self) -> bool:
        return bool(self._driver is not None or (self.uri and self.username and self.password))

    def _ensure_driver(self) -> Any | None:
        if self._driver is not None:
            return self._driver
        if not self.configured or self._disabled_reason:
            return None
        try:
            from neo4j import GraphDatabase  # type: ignore[import-not-found]

            self._driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        except (ImportError, OSError, RuntimeError, ValueError):
            self._disabled_reason = "driver_unavailable"
            return None
        return self._driver

    def upsert(
        self,
        document: Mapping[str, Any],
        entities: Iterable[Mapping[str, Any]],
        relations: Iterable[Mapping[str, Any]],
    ) -> bool:
        """Write a graph projection, returning false for unavailable Neo4j."""

        driver = self._ensure_driver()
        if driver is None:
            return False
        try:
            with driver.session(database=self.database) as session:
                session.run(
                    UPSERT_DOCUMENT_CYPHER,
                    document_id=document["document_id"],
                    filename=document.get("filename", ""),
                    extension=document.get("extension", ""),
                    content_sha256=document.get("content_sha256", ""),
                ).consume()
                for chunk in document.get("chunks", []):
                    session.run(UPSERT_CHUNK_CYPHER, **dict(chunk)).consume()
                for entity in entities:
                    session.run(
                        UPSERT_ENTITY_CYPHER,
                        entity_id=entity["entity_id"],
                        name=entity["name"],
                        entity_type=entity.get("type", "CONCEPT"),
                        chunk_ids=list(entity.get("chunk_ids", [])),
                        citations=list(entity.get("citations", [])),
                    ).consume()
                for relation in relations:
                    session.run(UPSERT_RELATION_CYPHER, **dict(relation)).consume()
        except (OSError, RuntimeError, TimeoutError, ValueError):
            self._disabled_reason = "write_unavailable"
            return False
        return True

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self._driver is not None and self._disabled_reason is None,
            "configured": self.configured,
            "reason": self._disabled_reason or (None if self.configured else "not_configured"),
        }

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None


__all__ = [
    "GraphRAGIndex",
    "Neo4jGraphWriter",
    "UPSERT_CHUNK_CYPHER",
    "UPSERT_DOCUMENT_CYPHER",
    "UPSERT_ENTITY_CYPHER",
    "UPSERT_RELATION_CYPHER",
    "extract_entities",
    "extract_relations",
]
