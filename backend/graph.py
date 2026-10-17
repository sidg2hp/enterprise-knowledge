"""LangGraph workflow for grounded, extractive knowledge queries."""

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

try:  # Support both `uvicorn main:app` from backend/ and package imports.
    from .knowledge import get_document_store
except ImportError:  # pragma: no cover - exercised by the direct server command
    from knowledge import get_document_store


class GraphState(TypedDict, total=False):
    query: str
    domain: str
    evidence: list[dict[str, Any]]
    response: str
    citations: list[str]
    abstained: bool


def query_analyzer(state: GraphState) -> dict[str, str]:
    query = state.get("query", "").lower()
    domain = "GENERAL"
    if any(word in query for word in ("resume", "candidate", "hire", "hr", "recruit")):
        domain = "HR"
    elif any(word in query for word in ("policy", "audit", "compliance")):
        domain = "POLICY"
    elif any(word in query for word in ("precedent", "history", "historical")):
        domain = "PRECEDENT"
    return {"domain": domain}


def real_retriever(state: GraphState) -> dict[str, Any]:
    """Call the persisted lexical retriever; no mock context is injected."""

    return {"evidence": get_document_store().search(state.get("query", ""))}


def synthesizer(state: GraphState) -> dict[str, Any]:
    evidence = state.get("evidence", [])
    if not evidence:
        return {
            "response": "I don't have enough evidence in the knowledge base to answer that.",
            "citations": [],
            "abstained": True,
        }
    excerpts = [f"{item['text']} [{item['citation']}]" for item in evidence]
    return {
        "response": "Based on the knowledge base: " + " ".join(excerpts),
        "citations": [item["citation"] for item in evidence],
        "abstained": False,
    }


def evaluator(state: GraphState) -> dict[str, Any]:
    """Keep citations tied to returned evidence and enforce abstention."""

    evidence = state.get("evidence", [])
    evidence_citations = [item["citation"] for item in evidence]
    citations = [citation for citation in state.get("citations", []) if citation in evidence_citations]
    if not evidence:
        return {"citations": [], "abstained": True}
    return {"citations": citations, "abstained": not bool(citations)}


workflow = StateGraph(GraphState)
workflow.add_node("query_analyzer", query_analyzer)
workflow.add_node("retriever", real_retriever)
workflow.add_node("synthesizer", synthesizer)
workflow.add_node("evaluator", evaluator)
workflow.set_entry_point("query_analyzer")
workflow.add_edge("query_analyzer", "retriever")
workflow.add_edge("retriever", "synthesizer")
workflow.add_edge("synthesizer", "evaluator")
workflow.add_edge("evaluator", END)
app_graph = workflow.compile()
