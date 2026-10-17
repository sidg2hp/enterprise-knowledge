"""Deterministic, dependency-free quality metrics for retrieval responses.

The functions in this module intentionally operate on the small dictionaries
returned by :mod:`knowledge` as well as plain string identifiers.  They are
useful in offline regression tests and evaluation fixtures; no model or
network service is involved.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
_STOP_WORDS = frozenset(
    {
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
)


def _identifier(value: object) -> str:
    """Extract a stable identifier from a result row or return a string."""

    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("citation", "chunk_id", "document_id", "id", "key"):
            candidate = value.get(key)
            if candidate is not None:
                return str(candidate)
    return str(value)


def _identifiers(values: Iterable[object]) -> list[str]:
    return [_identifier(value) for value in values]


def recall_at_k(retrieved: Sequence[object], relevant: Iterable[object], k: int) -> float:
    """Return recall at ``k`` for one query.

    Relevant identifiers are de-duplicated, while the denominator remains the
    number of unique relevant identifiers.  Empty relevance sets and non-
    positive cutoffs return ``0.0`` so batch evaluation cannot produce NaN.
    """

    if k <= 0:
        return 0.0
    relevant_ids = set(_identifiers(relevant))
    if not relevant_ids:
        return 0.0
    retrieved_ids = set(_identifiers(retrieved[:k]))
    return round(len(relevant_ids & retrieved_ids) / len(relevant_ids), 6)


def mrr_at_k(retrieved: Sequence[object], relevant: Iterable[object], k: int) -> float:
    """Return reciprocal rank of the first relevant result within ``k``."""

    if k <= 0:
        return 0.0
    relevant_ids = set(_identifiers(relevant))
    if not relevant_ids:
        return 0.0
    for rank, identifier in enumerate(_identifiers(retrieved[:k]), start=1):
        if identifier in relevant_ids:
            return round(1.0 / rank, 6)
    return 0.0


def mean_recall_at_k(
    cases: Iterable[tuple[Sequence[object], Iterable[object]]], k: int
) -> float:
    """Macro-average :func:`recall_at_k` over ``(retrieved, relevant)`` cases."""

    values = [recall_at_k(retrieved, relevant, k) for retrieved, relevant in cases]
    return round(sum(values) / len(values), 6) if values else 0.0


def mean_mrr_at_k(
    cases: Iterable[tuple[Sequence[object], Iterable[object]]], k: int
) -> float:
    """Macro-average :func:`mrr_at_k` over ``(retrieved, relevant)`` cases."""

    values = [mrr_at_k(retrieved, relevant, k) for retrieved, relevant in cases]
    return round(sum(values) / len(values), 6) if values else 0.0


def retrieval_metrics(
    retrieved: Sequence[object], relevant: Iterable[object], k: int = 5
) -> dict[str, float]:
    """Return a compact, serializable Recall@k/MRR@k result."""

    return {f"recall@{k}": recall_at_k(retrieved, relevant, k), f"mrr@{k}": mrr_at_k(retrieved, relevant, k)}


@dataclass(frozen=True)
class CitationValidation:
    """Result of checking citations against the returned evidence rows."""

    valid: bool
    cited: tuple[str, ...]
    valid_citations: tuple[str, ...]
    invalid_citations: tuple[str, ...]

    @property
    def score(self) -> float:
        """Fraction of supplied citations that resolve to returned evidence."""

        return round(len(self.valid_citations) / len(self.cited), 6) if self.cited else 1.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"score": self.score}


def validate_citations(
    citations: Iterable[object], evidence: Iterable[Mapping[str, object]]
) -> CitationValidation:
    """Check that every citation points to one of the supplied evidence rows."""

    cited = tuple(_identifiers(citations))
    available = {_identifier(row) for row in evidence}
    valid = tuple(citation for citation in cited if citation in available)
    invalid = tuple(citation for citation in cited if citation not in available)
    return CitationValidation(not invalid, cited, valid, invalid)


def citations_are_valid(
    citations: Iterable[object], evidence: Iterable[Mapping[str, object]]
) -> bool:
    """Boolean convenience wrapper around :func:`validate_citations`."""

    return validate_citations(citations, evidence).valid


def citation_validity_score(
    citations: Iterable[object], evidence: Iterable[Mapping[str, object]]
) -> float:
    """Return the fraction of citations resolved by the evidence."""

    return validate_citations(citations, evidence).score


def _tokens(value: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(value)
        if token.lower() not in _STOP_WORDS
    }


def groundedness_score(
    response: str,
    evidence: Iterable[Mapping[str, object]],
    citations: Iterable[object] | None = None,
) -> float:
    """Score response token support in returned evidence on a ``0..1`` scale.

    This is a deterministic lexical proxy, not a claim that a language model
    has verified facts.  Citation markers in square brackets are ignored for
    the lexical comparison.  If citations are supplied, unresolved citations
    reduce the score proportionally.
    """

    if not response or not response.strip():
        return 0.0
    evidence_rows = list(evidence)
    if not evidence_rows:
        return 0.0
    response_without_markers = re.sub(r"\[[^\]]*\]", " ", response)
    response_tokens = _tokens(response_without_markers)
    if not response_tokens:
        return 0.0
    supported_tokens = set()
    for row in evidence_rows:
        text = row.get("text", "")
        if isinstance(text, str):
            supported_tokens.update(_tokens(text))
    lexical_score = len(response_tokens & supported_tokens) / len(response_tokens)
    citation_score = (
        validate_citations(citations or (), evidence_rows).score if citations is not None else 1.0
    )
    return round(lexical_score * citation_score, 6)


def abstention_score(
    abstained: bool,
    evidence: Iterable[Mapping[str, object]],
    citations: Iterable[object] | None = None,
) -> float:
    """Score whether abstention matches the availability of usable evidence.

    With no citations argument, any evidence row is considered usable.  When
    citations are provided, only evidence referenced by a valid citation is
    usable, matching the backend's citation-gated answer behavior.
    """

    evidence_rows = list(evidence)
    usable = bool(evidence_rows)
    if citations is not None:
        usable = bool(validate_citations(citations, evidence_rows).valid_citations)
    expected_abstention = not usable
    return 1.0 if bool(abstained) == expected_abstention else 0.0


def evaluate_grounding(
    response: str,
    evidence: Iterable[Mapping[str, object]],
    citations: Iterable[object],
    abstained: bool,
) -> dict[str, float | bool]:
    """Return groundedness, citation validity, and abstention scores together."""

    evidence_rows = list(evidence)
    validation = validate_citations(citations, evidence_rows)
    return {
        "groundedness": groundedness_score(response, evidence_rows, citations),
        "citation_valid": validation.valid,
        "citation_validity": validation.score,
        "abstention": abstention_score(abstained, evidence_rows, citations),
    }


# Short aliases make the metric names convenient in notebooks and fixtures.
recall = recall_at_k
mrr = mrr_at_k
