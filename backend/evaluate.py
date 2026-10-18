"""Run the offline retrieval and grounding regression set.

Usage from the repository root::

    python -m backend.evaluate

The command builds an isolated temporary store, so it never changes the
developer's uploaded documents. It prints measured metrics rather than making
quality claims in documentation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

try:
    from .evaluation import evaluate_grounding, mean_mrr_at_k, mean_recall_at_k
    from .graph import app_graph
    from .knowledge import DocumentStore, configure_document_store, ingest_document
except ImportError:  # pragma: no cover - direct script compatibility
    from evaluation import evaluate_grounding, mean_mrr_at_k, mean_recall_at_k
    from graph import app_graph
    from knowledge import DocumentStore, configure_document_store, ingest_document


def run_evaluation(path: str | Path | None = None) -> dict[str, Any]:
    fixture_path = Path(path or Path(__file__).with_name("evaluation_cases.json"))
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="enterprise-eval-") as directory:
        store = DocumentStore(Path(directory) / "documents.json")
        configure_document_store(store)
        documents = {}
        for item in fixture["documents"]:
            document = ingest_document(item["filename"], item["content"].encode("utf-8"))
            store.upsert(document)
            documents[item["filename"]] = document

        retrieval_cases = []
        grounding_results = []
        for case in fixture["cases"]:
            result = app_graph.invoke({"query": case["query"]})
            expected = documents.get(case.get("document"))
            relevant = []
            if expected:
                relevant_indices = case.get("relevant_chunk_indices")
                relevant = [
                    chunk["citation"]
                    for chunk in expected["chunks"]
                    if relevant_indices is None or chunk["chunk_index"] in relevant_indices
                ]
            if relevant:
                retrieval_cases.append((result.get("evidence", []), relevant))
            grounding_results.append({
                "result": evaluate_grounding(
                    result.get("response", ""),
                    result.get("evidence", []),
                    result.get("citations", []),
                    result.get("abstained", False),
                ),
                "abstained": result.get("abstained", False),
            })

        answer_results = [item["result"] for item in grounding_results if not item["abstained"]]
        return {
            "fixture_version": fixture.get("version"),
            "case_count": len(retrieval_cases),
            "retrieval": {
                "recall@5": mean_recall_at_k(retrieval_cases, 5),
                "mrr@5": mean_mrr_at_k(retrieval_cases, 5),
            },
            "grounding": {
                metric: round(
                    sum(float(item[metric]) for item in answer_results) / len(answer_results),
                    6,
                )
                for metric in ("groundedness", "citation_validity", "abstention")
            },
            "abstention_accuracy": round(
                sum(float(item["result"]["abstention"]) for item in grounding_results) / len(grounding_results),
                6,
            ),
        }


if __name__ == "__main__":
    print(json.dumps(run_evaluation(), indent=2, sort_keys=True))
