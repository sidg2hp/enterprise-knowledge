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

try:  # Support both `uvicorn main:app` from backend/ and package imports.
    from .retrieval import (
        BM25Hit,
        BM25Ranker,
        DeterministicHashEmbedder,
        EmbeddingError,
        EmbeddingProvider,
        FAISSVectorIndex,
        FallbackEmbeddingProvider,
        HybridRetriever,
        OllamaEmbeddingAdapter,
        OptionalDependencyError,
        PurePythonVectorIndex,
    )
except ImportError:  # pragma: no cover - exercised by direct server commands
    from retrieval import (
        BM25Hit,
        BM25Ranker,
        DeterministicHashEmbedder,
        EmbeddingError,
        EmbeddingProvider,
        FAISSVectorIndex,
        FallbackEmbeddingProvider,
        HybridRetriever,
        OllamaEmbeddingAdapter,
        OptionalDependencyError,
        PurePythonVectorIndex,
    )


try:  # Support both `uvicorn main:app` from backend/ and package imports.
    from .graph_rag import GraphRAGIndex, Neo4jGraphWriter
except ImportError:  # pragma: no cover - exercised by direct server commands
    try:
        from graph_rag import GraphRAGIndex, Neo4jGraphWriter
    except ImportError:  # pragma: no cover - clean installs without the optional module
        class Neo4jGraphWriter:
            """No-op writer used when the optional graph_rag package is absent."""

            @classmethod
            def from_env(cls) -> "Neo4jGraphWriter":
                return cls()

            def upsert(self, *_args: Any, **_kwargs: Any) -> bool:
                return False

        class GraphRAGIndex:
            """Dependency-free chunk graph fallback for offline installations."""

            def __init__(self, path: str | Path, neo4j_writer: Any | None = None):
                self.path = Path(path)
                self._documents: dict[str, dict[str, Any]] = {}
                try:
                    with self.path.open("r", encoding="utf-8") as handle:
                        value = json.load(handle)
                    if isinstance(value, dict) and isinstance(value.get("documents"), dict):
                        self._documents = value["documents"]
                except (OSError, json.JSONDecodeError):
                    pass

            def _write(self) -> None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.path.with_suffix(self.path.suffix + ".tmp")
                temporary.write_text(
                    json.dumps({"documents": self._documents}, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, self.path)

            def index_document(self, document: dict[str, Any]) -> dict[str, Any]:
                self._documents[str(document["document_id"])] = document
                self._write()
                return document

            def delete_document(self, document_id: str) -> bool:
                deleted = self._documents.pop(document_id, None) is not None
                if deleted:
                    self._write()
                return deleted

            def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
                terms = set(_tokens(query))
                matches: list[dict[str, Any]] = []
                for document in self._documents.values():
                    for chunk in document.get("chunks", []):
                        matched = terms.intersection(_tokens(str(chunk.get("text", ""))))
                        if not matched:
                            continue
                        matches.append(
                            {
                                "chunk_id": chunk["chunk_id"],
                                "document_id": document["document_id"],
                                "text": chunk["text"],
                                "citation": chunk["citation"],
                                "matched_entities": sorted(matched),
                                "paths": [],
                                "source_chunk_citations": [chunk["citation"]],
                            }
                        )
                matches.sort(key=lambda item: (-len(item["matched_entities"]), item["citation"]))
                return matches[: max(1, limit)]

            def status(self) -> dict[str, Any]:
                chunks = [chunk for document in self._documents.values() for chunk in document.get("chunks", [])]
                return {
                    "enabled": True,
                    "path": str(self.path),
                    "document_count": len(self._documents),
                    "chunk_count": len(chunks),
                    "entity_count": 0,
                    "relation_count": 0,
                    "load_error": None,
                }


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

    # Preserve paragraph boundaries so graph paths and lexical evidence can
    # cite the smallest stable source chunk rather than a merged document span.
    return pieces


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
    """Small atomic JSON-backed document store and modular hybrid retriever.

    The default mode is deterministic BM25.  Set ``KNOWLEDGE_RETRIEVER_MODE``
    to ``hybrid`` or ``vector`` to opt into embeddings; Ollama and FAISS remain
    optional and degrade to the local hash embedder and vector index.
    """

    def __init__(self, path: str | Path | None = None, retriever: HybridRetriever | None = None):
        configured = (
            path
            or os.getenv("KNOWLEDGE_STORE_PATH")
            or Path(__file__).resolve().parent / "data" / "documents.json"
        )
        self.path = Path(configured)
        self._lock = RLock()
        self.retriever = retriever or HybridRetriever.from_environment(tokenizer=_tokens)
        graph_path = os.getenv("GRAPH_RAG_INDEX_PATH") or self.path.with_name("graph.json")
        self.graph_rag = GraphRAGIndex(graph_path, neo4j_writer=Neo4jGraphWriter.from_env())

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
            self.graph_rag.index_document(document)
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
            self.graph_rag.delete_document(document_id)
            return True

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        with self._lock:
            documents = self._read()["documents"].values()
            records = [
                {
                    "citation": chunk["citation"],
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "filename": document["filename"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                }
                for document in documents
                for chunk in document.get("chunks", [])
            ]
        return self.retriever.retrieve(query, records, max(1, limit))

    def graph_search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Retrieve graph-linked chunks, preserving their source identifiers."""

        return self.graph_rag.search(query, limit=max(1, limit))

    def graph_status(self) -> dict[str, Any]:
        return self.graph_rag.status()


_default_store = DocumentStore()


def get_document_store() -> DocumentStore:
    return _default_store


def configure_document_store(store: DocumentStore) -> None:
    """Swap the process store, primarily for embedding and isolated tests."""

    global _default_store
    _default_store = store
