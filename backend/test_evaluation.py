from evaluation import (
    abstention_score,
    citation_validity_score,
    citations_are_valid,
    evaluate_grounding,
    groundedness_score,
    mrr_at_k,
    recall_at_k,
    retrieval_metrics,
    validate_citations,
)


EVIDENCE = [
    {"citation": "doc_a#chunk-0", "text": "Jordan has Python experience."},
    {"citation": "doc_b#chunk-0", "text": "The policy requires structured interviews."},
]


def test_retrieval_metrics_are_deterministic_for_result_rows():
    retrieved = [EVIDENCE[1], EVIDENCE[0], {"citation": "doc_c#chunk-0"}]
    assert recall_at_k(retrieved, ["doc_a#chunk-0", "doc_b#chunk-0"], 2) == 1.0
    assert mrr_at_k(retrieved, ["doc_a#chunk-0"], 2) == 0.5
    assert retrieval_metrics(retrieved, ["doc_a#chunk-0"], 2) == {
        "recall@2": 1.0,
        "mrr@2": 0.5,
    }


def test_empty_relevance_and_cutoffs_are_safe():
    assert recall_at_k(["a"], [], 5) == 0.0
    assert mrr_at_k(["a"], ["a"], 0) == 0.0


def test_citation_validation_identifies_unresolved_citations():
    result = validate_citations(["doc_a#chunk-0", "missing"], EVIDENCE)
    assert result.valid is False
    assert result.invalid_citations == ("missing",)
    assert result.score == 0.5
    assert not citations_are_valid(["missing"], EVIDENCE)
    assert citation_validity_score(["doc_a#chunk-0"], EVIDENCE) == 1.0


def test_groundedness_and_abstention_scores_use_evidence_and_citations():
    response = "Jordan has Python experience. [doc_a#chunk-0]"
    assert groundedness_score(response, EVIDENCE, ["doc_a#chunk-0"]) == 1.0
    assert abstention_score(False, EVIDENCE, ["doc_a#chunk-0"]) == 1.0
    assert abstention_score(True, [], []) == 1.0
    assert evaluate_grounding(response, EVIDENCE, ["doc_a#chunk-0"], False) == {
        "groundedness": 1.0,
        "citation_valid": True,
        "citation_validity": 1.0,
        "abstention": 1.0,
    }
