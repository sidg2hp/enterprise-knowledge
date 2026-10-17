"""Optional, offline-friendly retrieval building blocks.

The module intentionally has no mandatory third-party imports.  BM25 and a
small hashing embedder make the default path deterministic and usable in an
offline CI job.  FAISS and Ollama are adapters: they are loaded or contacted
only when a caller explicitly selects a vector-capable mode.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)


class OptionalDependencyError(RuntimeError):
    """Raised when an optional vector backend was requested but is unavailable."""


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider cannot return valid vectors."""


class EmbeddingProvider(Protocol):
    """Minimal protocol implemented by local and remote embedding providers."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector for each input string."""


def default_tokenizer(value: str) -> list[str]:
    """Tokenize text without requiring a NLP package."""

    return [token.lower() for token in _TOKEN_RE.findall(value)]


@dataclass(frozen=True)
class BM25Hit:
    """A lexical ranking result before document metadata is attached."""

    key: str
    score: float
    matched_terms: tuple[str, ...]


class BM25Ranker:
    """Small Okapi-BM25 implementation with deterministic tie breaking.

    ``documents`` accepted by :meth:`rank` are mappings with a ``text`` field;
    their key is taken from ``key`` and then common stable identifiers.  The
    ranker does not retain caller mappings, so a document store can safely
    rebuild it after an upsert or delete.
    """

    def __init__(
        self,
        *,
        tokenizer: Callable[[str], list[str]] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("BM25 parameters must satisfy k1 >= 0 and 0 <= b <= 1")
        self.tokenizer = tokenizer or default_tokenizer
        self.k1 = k1
        self.b = b
        self._documents: list[tuple[str, list[str]]] = []
        self._document_frequency: Counter[str] = Counter()
        self._average_length = 0.0

    @staticmethod
    def _key(document: Mapping[str, Any], position: int) -> str:
        for name in ("key", "chunk_id", "citation", "id"):
            if document.get(name) is not None:
                return str(document[name])
        return str(position)

    def fit(self, documents: Sequence[Mapping[str, Any]]) -> "BM25Ranker":
        """Index ``documents`` and return this ranker for fluent setup."""

        self._documents = []
        self._document_frequency = Counter()
        lengths: list[int] = []
        for position, document in enumerate(documents):
            tokens = self.tokenizer(str(document.get("text", "")))
            key = self._key(document, position)
            self._documents.append((key, tokens))
            lengths.append(len(tokens))
            self._document_frequency.update(set(tokens))
        self._average_length = sum(lengths) / len(lengths) if lengths else 0.0
        return self

    index = fit

    def _score(self, query_terms: Sequence[str], tokens: Sequence[str]) -> tuple[float, tuple[str, ...]]:
        if not tokens:
            return 0.0, ()
        counts = Counter(tokens)
        matched = tuple(sorted({term for term in query_terms if counts[term]}))
        if not matched:
            return 0.0, ()
        document_count = len(self._documents)
        length_factor = len(tokens) / self._average_length if self._average_length else 1.0
        score = 0.0
        for term in matched:
            frequency = counts[term]
            document_frequency = self._document_frequency[term]
            idf = math.log(1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
            score += idf * (frequency * (self.k1 + 1.0)) / (frequency + self.k1 * (1.0 - self.b + self.b * length_factor))
        return score, matched

    def rank(self, query: str, documents: Sequence[Mapping[str, Any]] | None = None, limit: int | None = None) -> list[BM25Hit]:
        """Return positive BM25 hits, ordered by score and stable key."""

        if documents is not None:
            self.fit(documents)
        query_terms = self.tokenizer(query)
        if not query_terms or not self._documents:
            return []
        hits: list[BM25Hit] = []
        for key, tokens in self._documents:
            score, matched = self._score(query_terms, tokens)
            if matched:
                hits.append(BM25Hit(key, score, matched))
        hits.sort(key=lambda item: (-item.score, item.key))
        return hits if limit is None else hits[: max(0, limit)]

    search = rank


class DeterministicHashEmbedder:
    """Dependency-free embeddings for deterministic local vector retrieval."""

    def __init__(self, dimension: int = 96) -> None:
        if dimension < 8:
            raise ValueError("embedding dimension must be at least 8")
        self.dimension = dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimension
            for token in default_tokenizer(text):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[index] += sign
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / norm for value in vector] if norm else vector)
        return vectors


class OllamaEmbeddingAdapter:
    """Ollama ``/api/embed`` adapter with no import-time or network side effect.

    Ollama is opt-in.  Set ``OLLAMA_BASE_URL`` and optionally
    ``OLLAMA_EMBEDDING_MODEL`` (or pass them to the constructor) and wrap this adapter with
    :class:`FallbackEmbeddingProvider` when deterministic degradation is
    desired.  Older Ollama versions exposing only ``/api/embeddings`` are
    supported as a compatibility fallback.
    """

    def __init__(self, base_url: str, model: str, timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_environment(cls) -> "OllamaEmbeddingAdapter | None":
        base_url = os.getenv("OLLAMA_BASE_URL", "").strip()
        model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text").strip()
        if not base_url:
            return None
        try:
            timeout = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "5"))
        except ValueError:
            timeout = 5.0
        return cls(base_url, model, timeout=max(0.1, timeout))

    @staticmethod
    def _vectors(payload: Any) -> list[list[float]]:
        values = payload.get("embeddings") if isinstance(payload, dict) else None
        if values is None and isinstance(payload, dict) and payload.get("embedding") is not None:
            values = [payload["embedding"]]
        if not isinstance(values, list) or not values or not all(isinstance(vector, list) for vector in values):
            raise EmbeddingError("Ollama returned no embeddings")
        try:
            vectors = [[float(value) for value in vector] for vector in values]
        except (TypeError, ValueError) as exc:
            raise EmbeddingError("Ollama returned an invalid embedding") from exc
        if not vectors or not vectors[0] or any(len(vector) != len(vectors[0]) for vector in vectors):
            raise EmbeddingError("Ollama returned embeddings with inconsistent dimensions")
        return vectors

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = json.dumps({"model": self.model, "input": list(texts)}).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return self._vectors(json.loads(response.read().decode("utf-8")))
        except (OSError, URLError, TimeoutError, ValueError, EmbeddingError) as first_error:
            # The legacy endpoint accepts one prompt at a time.
            try:
                vectors: list[list[float]] = []
                for text in texts:
                    legacy = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
                    legacy_request = Request(
                        f"{self.base_url}/api/embeddings",
                        data=legacy,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urlopen(legacy_request, timeout=self.timeout) as response:
                        vectors.extend(self._vectors(json.loads(response.read().decode("utf-8"))))
                return vectors
            except (OSError, URLError, TimeoutError, ValueError, EmbeddingError) as second_error:
                raise EmbeddingError(f"Ollama embedding request failed: {second_error}") from first_error


class FallbackEmbeddingProvider:
    """Use a primary provider when available and deterministic vectors otherwise."""

    def __init__(self, primary: EmbeddingProvider, fallback: EmbeddingProvider | None = None) -> None:
        self.primary = primary
        self.fallback = fallback or DeterministicHashEmbedder()
        self.last_mode = "primary"
        self._use_fallback = False

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self._use_fallback:
            self.last_mode = "deterministic-fallback"
            return self.fallback.embed(texts)
        try:
            vectors = self.primary.embed(texts)
            if len(vectors) != len(texts) or (vectors and not vectors[0]):
                raise EmbeddingError("embedding provider returned an invalid result")
            self.last_mode = "primary"
            return vectors
        except (EmbeddingError, OSError, URLError, TimeoutError, ValueError):
            self._use_fallback = True
            self.last_mode = "deterministic-fallback"
            return self.fallback.embed(texts)


class PurePythonVectorIndex:
    """Cosine vector index used when FAISS is not installed."""

    backend = "stdlib"

    def __init__(self, vectors: Sequence[Sequence[float]], keys: Sequence[str]) -> None:
        if len(vectors) != len(keys):
            raise ValueError("vectors and keys must have the same length")
        self._vectors = [list(map(float, vector)) for vector in vectors]
        self._keys = list(keys)
        self._dimension = len(self._vectors[0]) if self._vectors else 0
        if self._dimension and any(len(vector) != self._dimension for vector in self._vectors):
            raise ValueError("vectors must have consistent dimensions")

    def search(self, query: Sequence[float], limit: int) -> list[tuple[str, float]]:
        if self._dimension and len(query) != self._dimension:
            raise ValueError("query vector dimension does not match vector index")
        query_norm = math.sqrt(sum(value * value for value in query))
        if not query_norm:
            return []
        results: list[tuple[str, float]] = []
        for key, vector in zip(self._keys, self._vectors):
            vector_norm = math.sqrt(sum(value * value for value in vector))
            score = sum(left * right for left, right in zip(query, vector)) / (query_norm * vector_norm) if vector_norm else 0.0
            results.append((key, score))
        results.sort(key=lambda item: (-item[1], item[0]))
        return results[: max(0, limit)]


class FAISSVectorIndex:
    """Thin optional FAISS cosine-similarity adapter.

    Importing this module never imports FAISS.  Call :meth:`available` to
    inspect support or :meth:`build` to request it and receive a clear error.
    """

    backend = "faiss"

    def __init__(self, index: Any, keys: Sequence[str]) -> None:
        self._index = index
        self._keys = list(keys)

    @staticmethod
    def available() -> bool:
        try:
            import faiss  # type: ignore[import-not-found]
            import numpy  # type: ignore[import-not-found]
        except ImportError:
            return False
        return faiss is not None and numpy is not None

    @classmethod
    def build(cls, vectors: Sequence[Sequence[float]], keys: Sequence[str]) -> "FAISSVectorIndex":
        try:
            import faiss  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]
        except ImportError as exc:
            raise OptionalDependencyError("FAISS retrieval requires the optional faiss-cpu and numpy packages") from exc
        if len(vectors) != len(keys) or not vectors:
            raise ValueError("vectors and keys must be non-empty and have the same length")
        matrix = np.asarray(vectors, dtype="float32")
        if matrix.ndim != 2 or not matrix.shape[1]:
            raise ValueError("vectors must be a two-dimensional matrix")
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return cls(index, keys)

    def search(self, query: Sequence[float], limit: int) -> list[tuple[str, float]]:
        import faiss  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]

        vector = np.asarray([query], dtype="float32")
        if vector.shape[1] != self._index.d:
            raise ValueError("query vector dimension does not match FAISS index")
        faiss.normalize_L2(vector)
        faiss_scores, positions = self._index.search(vector, max(0, limit))
        return [
            (self._keys[position], float(score))
            for score, position in zip(faiss_scores[0], positions[0])
            if position >= 0
        ]


def _configured_embedding_provider() -> EmbeddingProvider:
    ollama = OllamaEmbeddingAdapter.from_environment()
    if ollama is None:
        return DeterministicHashEmbedder()
    return FallbackEmbeddingProvider(ollama)


class HybridRetriever:
    """Combine BM25 and optional vector evidence while preserving chunk records."""

    def __init__(
        self,
        *,
        mode: str = "bm25",
        lexical_weight: float = 0.7,
        vector_weight: float = 0.3,
        embedding_provider: EmbeddingProvider | None = None,
        prefer_faiss: bool = True,
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        normalized_mode = mode.strip().lower().replace("-", "_")
        aliases = {"lexical": "bm25", "hybrid": "hybrid", "vector": "vector", "bm25": "bm25"}
        if normalized_mode not in aliases:
            raise ValueError("retriever mode must be bm25, hybrid, or vector")
        if lexical_weight < 0 or vector_weight < 0 or lexical_weight + vector_weight == 0:
            raise ValueError("retrieval weights must be non-negative and not both zero")
        self.mode = aliases[normalized_mode]
        self.lexical_weight = lexical_weight
        self.vector_weight = vector_weight
        self.embedding_provider = embedding_provider or _configured_embedding_provider()
        self.prefer_faiss = prefer_faiss
        self.ranker = BM25Ranker(tokenizer=tokenizer)
        self.vector_backend = "disabled"

    @classmethod
    def from_environment(cls, *, tokenizer: Callable[[str], list[str]] | None = None) -> "HybridRetriever":
        mode = os.getenv("KNOWLEDGE_RETRIEVER_MODE", os.getenv("RETRIEVER_MODE", "bm25"))
        try:
            lexical_weight = float(os.getenv("KNOWLEDGE_LEXICAL_WEIGHT", "0.7"))
            vector_weight = float(os.getenv("KNOWLEDGE_VECTOR_WEIGHT", "0.3"))
        except ValueError:
            lexical_weight, vector_weight = 0.7, 0.3
        return cls(mode=mode, lexical_weight=lexical_weight, vector_weight=vector_weight, tokenizer=tokenizer)

    @staticmethod
    def _copy_record(record: Mapping[str, Any]) -> dict[str, Any]:
        return dict(record)

    def retrieve(self, query: str, records: Sequence[Mapping[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        record_by_key = {str(record.get("chunk_id", record.get("citation", index))): record for index, record in enumerate(records)}
        lexical_hits = self.ranker.rank(query, records)
        lexical_by_key = {hit.key: hit for hit in lexical_hits}
        vector_by_key: dict[str, float] = {}
        active_mode = "bm25"
        if self.mode in {"hybrid", "vector"} and records:
            texts = [str(record.get("text", "")) for record in records]
            keys = list(record_by_key)
            try:
                vectors = self.embedding_provider.embed(texts)
                if len(vectors) != len(records):
                    raise EmbeddingError("embedding provider returned the wrong number of vectors")
                index: PurePythonVectorIndex | FAISSVectorIndex
                if self.prefer_faiss and FAISSVectorIndex.available():
                    index = FAISSVectorIndex.build(vectors, keys)
                else:
                    index = PurePythonVectorIndex(vectors, keys)
                self.vector_backend = index.backend
                query_vector = self.embedding_provider.embed([query])[0]
                vector_by_key = dict(index.search(query_vector, max(limit, len(records))))
                active_mode = "vector" if self.mode == "vector" else "hybrid"
            except (EmbeddingError, OptionalDependencyError, OSError, URLError, TimeoutError, ValueError):
                self.vector_backend = "unavailable"
                # A vector request must remain useful in offline environments.
                if self.mode == "vector":
                    active_mode = "bm25-fallback"

        keys = set(lexical_by_key) | set(vector_by_key)
        if not keys:
            return []
        max_lexical = max((hit.score for hit in lexical_by_key.values()), default=0.0)
        results: list[dict[str, Any]] = []
        for key in keys:
            record = record_by_key.get(key)
            if record is None:
                continue
            lexical_score = lexical_by_key[key].score if key in lexical_by_key else 0.0
            vector_score = vector_by_key.get(key)
            normalized_lexical = lexical_score / max_lexical if max_lexical else 0.0
            normalized_vector = ((vector_score + 1.0) / 2.0) if vector_score is not None else 0.0
            if active_mode.startswith("bm25"):
                score = lexical_score
            elif active_mode == "vector":
                score = normalized_vector
            else:
                total_weight = self.lexical_weight + self.vector_weight
                score = (self.lexical_weight * normalized_lexical + self.vector_weight * normalized_vector) / total_weight
            result = self._copy_record(record)
            result.update(
                {
                    "score": round(score, 6),
                    "lexical_score": round(lexical_score, 6),
                    "vector_score": round(vector_score, 6) if vector_score is not None else None,
                    "retriever_mode": active_mode,
                    "retrieval": {
                        "mode": active_mode,
                        "lexical_score": round(lexical_score, 6),
                        "vector_score": round(vector_score, 6) if vector_score is not None else None,
                        "vector_backend": self.vector_backend,
                    },
                    "matched_terms": list(lexical_by_key[key].matched_terms) if key in lexical_by_key else [],
                }
            )
            results.append(result)
        results.sort(key=lambda item: (-item["score"], str(item.get("citation", item.get("chunk_id", "")))))
        return results[:limit]


__all__ = [
    "BM25Hit",
    "BM25Ranker",
    "DeterministicHashEmbedder",
    "EmbeddingError",
    "EmbeddingProvider",
    "FAISSVectorIndex",
    "FallbackEmbeddingProvider",
    "HybridRetriever",
    "OllamaEmbeddingAdapter",
    "OptionalDependencyError",
    "PurePythonVectorIndex",
]
