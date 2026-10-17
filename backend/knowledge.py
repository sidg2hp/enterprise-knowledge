"""Deterministic local document ingestion and lexical retrieval.

The backend deliberately keeps its knowledge store boring: UTF-8 documents are
normalized, chunked, and persisted as JSON.  This makes the vertical slice
usable offline and keeps document and citation identifiers stable across
process restarts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from threading import RLock
from typing import Any


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown"}
MAX_DOCUMENT_BYTES = 5 * 1024 * 1024
CHUNK_SIZE = 1200

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
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
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


class DocumentError(ValueError):
    """Raised when an uploaded document cannot be ingested."""


def normalize_text(content: bytes) -> str:
    """Decode and normalize text so equivalent uploads produce same metadata."""

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentError("documents must be valid UTF-8 text") from exc
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def _split_long_piece(piece: str, start: int) -> list[tuple[str, int, int]]:
    result: list[tuple[str, int, int]] = []
    cursor = 0
    while cursor < len(piece):
        end = min(cursor + CHUNK_SIZE, len(piece))
        if end < len(piece):
            boundary = piece.rfind(" ", cursor, end)
            if boundary > cursor + CHUNK_SIZE // 2:
                end = boundary
        chunk = piece[cursor:end].strip()
        if chunk:
            left = start + cursor + len(piece[cursor:end]) - len(piece[cursor:end].lstrip())
            result.append((chunk, left, left + len(chunk)))
        cursor = end
        while cursor < len(piece) and piece[cursor].isspace():
            cursor += 1
    return result


def chunk_text(text: str) -> list[tuple[str, int, int]]:
    """Split normalized text at paragraph boundaries with stable positions."""

    if not text:
        return []
    paragraphs = list(re.finditer(r"\S.*?(?=\n{2,}|\Z)", text, re.DOTALL))
    if not paragraphs:
        return []

    pieces: list[tuple[str, int, int]] = []
    for match in paragraphs:
        piece = match.group(0).strip()
        if len(piece) <= CHUNK_SIZE:
            left = match.start() + len(match.group(0)) - len(match.group(0).lstrip())
            pieces.append((piece, left, left + len(piece)))
        else:
            pieces.extend(_split_long_piece(piece, match.start()))

    chunks: list[tuple[str, int, int]] = []
    for piece, start, end in pieces:
        if chunks and end - chunks[-1][1] <= CHUNK_SIZE:
            previous_text, previous_start, _ = chunks[-1]
            combined = f"{previous_text}\n\n{piece}"
            chunks[-1] = (combined, previous_start, end)
        else:
            chunks.append((piece, start, end))
    return chunks


def _document_id(filename: str, text: str) -> str:
    digest = hashlib.sha256(f"{filename}\0{text}".encode("utf-8")).hexdigest()
    return f"doc_{digest[:24]}"


def ingest_document(filename: str, content: bytes) -> dict[str, Any]:
    """Create a stable document record from a supported TXT/Markdown upload."""

    safe_name = Path(filename).name
    extension = Path(safe_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentError("only .txt, .md, and .markdown files are supported")
    if not content:
        raise DocumentError("document is empty")
    if len(content) > MAX_DOCUMENT_BYTES:
        raise DocumentError("document exceeds the 5 MiB limit")

    text = normalize_text(content)
    if not text:
        raise DocumentError("document is empty")
    document_id = _document_id(safe_name, text)
    content_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    chunks = []
    for index, (chunk, start, end) in enumerate(chunk_text(text)):
        chunks.append(
            {
                "chunk_id": f"{document_id}_chunk_{index:04d}",
                "document_id": document_id,
                "chunk_index": index,
                "text": chunk,
                "start_char": start,
                "end_char": end,
                "citation": f"{document_id}#chunk-{index}",
            }
        )
    return {
        "document_id": document_id,
        "filename": safe_name,
        "extension": extension,
        "content_sha256": content_sha256,
        "size_bytes": len(text.encode("utf-8")),
        "text_length": len(text),
        "chunks": chunks,
    }


def _tokens(value: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(value) if token.lower() not in _STOP_WORDS]


class DocumentStore:
    """Small atomic JSON-backed document store and lexical retriever."""

    def __init__(self, path: str | Path | None = None):
        configured = (
            path
            or os.getenv("KNOWLEDGE_STORE_PATH")
            or Path(__file__).resolve().parent / "data" / "documents.json"
        )
        self.path = Path(configured)
        self._lock = RLock()

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"documents": {}}
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"could not read document store: {self.path}") from exc
        return value if isinstance(value, dict) and isinstance(value.get("documents"), dict) else {"documents": {}}

    def _write(self, value: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="documents-", suffix=".json", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def upsert(self, document: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            state = self._read()
            state["documents"][document["document_id"]] = document
            self._write(state)
        return document

    def list_documents(self) -> list[dict[str, Any]]:
        with self._lock:
            documents = self._read()["documents"].values()
            return [
                {
                    key: document[key]
                    for key in (
                        "document_id",
                        "filename",
                        "extension",
                        "content_sha256",
                        "size_bytes",
                        "text_length",
                    )
                }
                | {"chunk_count": len(document.get("chunks", []))}
                for document in sorted(documents, key=lambda item: item["document_id"])
            ]

    def delete(self, document_id: str) -> bool:
        with self._lock:
            state = self._read()
            if document_id not in state["documents"]:
                return False
            del state["documents"][document_id]
            self._write(state)
            return True

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        unique_query_tokens = set(query_tokens)
        results: list[dict[str, Any]] = []
        with self._lock:
            documents = self._read()["documents"].values()
            for document in documents:
                for chunk in document.get("chunks", []):
                    chunk_tokens = _tokens(chunk["text"])
                    counts = {token: chunk_tokens.count(token) for token in unique_query_tokens}
                    matched_terms = sorted(token for token, count in counts.items() if count)
                    if not matched_terms:
                        continue
                    score = sum(min(counts[token], 3) for token in matched_terms) / len(unique_query_tokens)
                    results.append(
                        {
                            "citation": chunk["citation"],
                            "chunk_id": chunk["chunk_id"],
                            "document_id": chunk["document_id"],
                            "filename": document["filename"],
                            "chunk_index": chunk["chunk_index"],
                            "text": chunk["text"],
                            "score": round(score, 6),
                            "matched_terms": matched_terms,
                        }
                    )
        results.sort(key=lambda result: (-result["score"], result["citation"]))
        return results[: max(1, limit)]


_default_store = DocumentStore()


def get_document_store() -> DocumentStore:
    return _default_store


def configure_document_store(store: DocumentStore) -> None:
    """Swap the process store, primarily for embedding and isolated tests."""

    global _default_store
    _default_store = store
