"""Small JSON-line observability helpers that never emit document text.

The logger records operational metadata for query and ingestion events while
deliberately omitting query/document bodies.  Keeping the event schema and
allowlist here makes accidental content logging harder than logging arbitrary
request dictionaries.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, TextIO


LOGGER_NAME = "enterprise.backend.events"
EVENT_VERSION = 1


def new_request_id() -> str:
    """Create a request identifier suitable for correlating event records."""

    return str(uuid.uuid4())


def ensure_request_id(request_id: str | None = None) -> str:
    """Use a supplied non-empty request ID or create one."""

    if request_id is not None and request_id.strip():
        return request_id.strip()
    return new_request_id()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class JsonEventLogger:
    """Emit allowlisted operational events as one JSON object per log line."""

    def __init__(self, logger: logging.Logger | None = None, stream: TextIO | None = None):
        self.logger = logger or logging.getLogger(LOGGER_NAME)
        self.stream = stream

    def _write(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if self.stream is not None:
            self.stream.write(line + "\n")
            self.stream.flush()
        else:
            self.logger.info(line)

    def emit(self, event: str, request_id: str | None = None, **fields: Any) -> dict[str, Any]:
        """Emit a generic event using only scalar metadata fields.

        ``text``, ``content``, and arbitrary nested values are intentionally
        rejected rather than redacted after the fact, so callers notice an
        unsafe logging attempt during development.
        """

        forbidden = {"text", "content", "body", "document", "document_text", "query"}
        unsafe = forbidden.intersection(fields)
        if unsafe:
            raise ValueError(f"document/request content fields cannot be logged: {sorted(unsafe)}")
        safe_fields: dict[str, Any] = {}
        for key, value in fields.items():
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise TypeError(f"event field {key!r} must be scalar metadata")
            safe_fields[key] = value
        payload = {
            "event": event,
            "event_version": EVENT_VERSION,
            "request_id": ensure_request_id(request_id),
            "timestamp": _timestamp(),
            **safe_fields,
        }
        self._write(payload)
        return payload

    def query(
        self,
        request_id: str | None = None,
        *,
        query: str | None = None,
        result_count: int = 0,
        citation_count: int = 0,
        abstained: bool = False,
        latency_ms: float | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Log query metadata; the query text itself is never included."""

        fields: dict[str, Any] = {
            "query_length": len(query) if query is not None else None,
            "query_sha256": _sha256(query) if query is not None else None,
            "result_count": result_count,
            "citation_count": citation_count,
            "abstained": abstained,
        }
        if latency_ms is not None:
            fields["latency_ms"] = round(latency_ms, 3)
        if error is not None:
            fields["error"] = error
        return self.emit("query", request_id, **fields)

    def ingestion(
        self,
        request_id: str | None = None,
        *,
        filename: str | None = None,
        size_bytes: int | None = None,
        content_sha256: str | None = None,
        chunk_count: int = 0,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Log ingestion metadata; document bytes and text are never included."""

        fields: dict[str, Any] = {
            "filename": filename,
            "size_bytes": size_bytes,
            "content_sha256": content_sha256,
            "chunk_count": chunk_count,
        }
        if error is not None:
            fields["error"] = error
        return self.emit("ingestion", request_id, **fields)


def get_event_logger() -> JsonEventLogger:
    """Return a logger configured for the backend event namespace."""

    return JsonEventLogger()


def log_query_event(
    request_id: str | None = None, *, logger: JsonEventLogger | None = None, **kwargs: Any
) -> dict[str, Any]:
    """Convenience wrapper for emitting a query event."""

    return (logger or get_event_logger()).query(request_id, **kwargs)


def log_ingestion_event(
    request_id: str | None = None, *, logger: JsonEventLogger | None = None, **kwargs: Any
) -> dict[str, Any]:
    """Convenience wrapper for emitting an ingestion event."""

    return (logger or get_event_logger()).ingestion(request_id, **kwargs)
