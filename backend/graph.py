"""Deterministic multi-domain LangGraph workflow for grounded queries.

The workflow has a small specialist node for each supported domain, followed by
one shared retrieval/synthesis/evaluation path.  Specialists add routing
metadata, not invented answers.  The persisted lexical store remains the
source of cited evidence. A local GraphRAG index contributes graph-linked
context by default, while the optional Neo4j adapter can augment it.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

try:  # Support both `uvicorn main:app` from backend/ and package imports.
    from .graph_adapter import get_graph_adapter
    from .knowledge import get_document_store
    from .llm import try_generate_grounded_answer
except ImportError:  # pragma: no cover - exercised by the direct server command
    from graph_adapter import get_graph_adapter
    from knowledge import get_document_store
    from llm import try_generate_grounded_answer


DOMAINS = ("HR", "POLICY", "PRECEDENT", "FINANCE", "ENGINEERING", "GENERAL")

_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "HR": (
        "resume",
        "candidate",
        "hire",
        "hr",
        "recruit",
        "employee",
        "interview",
        "onboarding",
        "benefits",
        "performance",
    ),
    "POLICY": (
        "policy",
        "policies",
        "audit",
        "compliance",
        "regulation",
        "rule",
        "procedure",
        "guideline",
        "governance",
    ),
    "PRECEDENT": (
        "precedent",
        "history",
        "historical",
        "previous",
        "prior",
        "case",
        "decision",
        "ruling",
    ),
    "FINANCE": (
        "finance",
        "financial",
        "budget",
        "cost",
        "revenue",
        "expense",
        "invoice",
        "pricing",
        "forecast",
        "payroll",
        "profit",
    ),
    "ENGINEERING": (
        "engineering",
        "engineer",
        "code",
        "software",
        "api",
        "architecture",
        "deploy",
        "deployment",
        "bug",
        "database",
        "langgraph",
        "infrastructure",
    ),
}

_DOMAIN_PROFILES: dict[str, dict[str, str]] = {
    "HR": {
        "specialist": "hr_agent",
        "focus": "people, recruiting, employee, and talent records",
    },
    "POLICY": {
        "specialist": "policy_agent",
        "focus": "rules, controls, procedures, and compliance guidance",
    },
    "PRECEDENT": {
        "specialist": "precedent_agent",
        "focus": "prior decisions, historical cases, and institutional precedent",
    },
    "FINANCE": {
        "specialist": "finance_agent",
        "focus": "budgets, costs, financial operations, and forecasts",
    },
    "ENGINEERING": {
        "specialist": "engineering_agent",
        "focus": "software, systems, architecture, and technical operations",
    },
    "GENERAL": {
        "specialist": "general_agent",
        "focus": "cross-functional enterprise knowledge",
    },
}


class GraphState(TypedDict, total=False):
    query: str
    domain: str
    domain_metadata: dict[str, Any]
    agent_metadata: dict[str, Any]
    evidence: list[dict[str, Any]]
    graph_context: list[dict[str, Any]]
    graph_context_status: dict[str, Any]
    response: str
    citations: list[str]
    abstained: bool
    generation_mode: str


def _keyword_matches(query: str, keywords: tuple[str, ...]) -> list[str]:
    # Word boundaries avoid classifying "costume" as FINANCE because it starts
    # with "cost", while keeping the matching rules deterministic and inspectable.
    import re

    lowered = query.lower()
    return [word for word in keywords if re.search(rf"\b{re.escape(word)}\b", lowered)]


def query_analyzer(state: GraphState) -> dict[str, Any]:
    """Classify a query and expose enough metadata to audit routing."""

    query = state.get("query", "").strip()
    matches = {domain: _keyword_matches(query, keywords) for domain, keywords in _DOMAIN_KEYWORDS.items()}
    ranked = sorted(matches, key=lambda domain: (-len(matches[domain]), DOMAINS.index(domain)))
    best = ranked[0] if ranked and matches[ranked[0]] else "GENERAL"
    matched = matches.get(best, [])
    score = len(matched)
    confidence = round(min(1.0, score / 3), 3) if best != "GENERAL" else 0.0
    profile = _DOMAIN_PROFILES[best]
    metadata = {
        "domain": best,
        "confidence": confidence,
        "matched_keywords": matched,
        "routing_reason": (
            f"matched {', '.join(matched)}"
            if matched
            else "no specialist keywords matched; using cross-functional routing"
        ),
        "specialist": profile["specialist"],
    }
    return {"domain": best, "domain_metadata": metadata}


def _specialist_node(state: GraphState, domain: str) -> dict[str, Any]:
    profile = _DOMAIN_PROFILES[domain]
    metadata = dict(state.get("domain_metadata", {}))
    metadata.update(
        {
            "domain": domain,
            "specialist": profile["specialist"],
            "focus": profile["focus"],
            "evidence_policy": "Use retrieved passages only; abstain when they do not support an answer.",
        }
    )
    return {
        "domain": domain,
        "domain_metadata": metadata,
        "agent_metadata": {
            "name": profile["specialist"],
            "domain": domain,
            "focus": profile["focus"],
            "mode": "deterministic_extract_then_synthesize",
        },
    }


def hr_agent(state: GraphState) -> dict[str, Any]:
    return _specialist_node(state, "HR")


def policy_agent(state: GraphState) -> dict[str, Any]:
    return _specialist_node(state, "POLICY")


def precedent_agent(state: GraphState) -> dict[str, Any]:
    return _specialist_node(state, "PRECEDENT")


def finance_agent(state: GraphState) -> dict[str, Any]:
    return _specialist_node(state, "FINANCE")


def engineering_agent(state: GraphState) -> dict[str, Any]:
    return _specialist_node(state, "ENGINEERING")


def general_agent(state: GraphState) -> dict[str, Any]:
    return _specialist_node(state, "GENERAL")


def _route_domain(state: GraphState) -> str:
    domain = state.get("domain", "GENERAL")
    return domain if domain in DOMAINS else "GENERAL"


def graph_context_retriever(state: GraphState) -> dict[str, Any]:
    """Retrieve local graph context and optionally augment it with Neo4j rows."""

    store = get_document_store()
    local_context: list[dict[str, Any]] = []
    try:
        local_context = store.graph_search(state.get("query", ""))
    except (ValueError, RuntimeError, OSError):
        # The graph is an enhancement; lexical retrieval must keep working.
        local_context = []
    local_status = store.graph_status()

    adapter = get_graph_adapter()
    status = adapter.status()
    context = list(local_context)
    neo4j_context: list[dict[str, Any]] = []
    try:
        if adapter.configured:
            neo4j_context = adapter.search_context(state.get("query", ""))
            context.extend({"source": "neo4j", **item} for item in neo4j_context)
        status = adapter.status()
    except (AttributeError, ValueError, RuntimeError, OSError):
        # The graph is an enhancement; lexical retrieval must keep working.
        status = {**status, "enabled": False, "reason": "query_unavailable"}
    # Keep the existing ``enabled`` field tied to the optional Neo4j adapter so
    # current API clients retain their meaning, and expose local GraphRAG state
    # alongside it.
    status = {
        **status,
        "local_enabled": bool(local_status.get("enabled", True)),
        "local_result_count": len(local_context),
        "neo4j_result_count": len(neo4j_context),
    }
    return {"graph_context": context, "graph_context_status": status}


def real_retriever(state: GraphState) -> dict[str, Any]:
    """Call the persisted lexical retriever; no mock context is injected."""

    return {"evidence": get_document_store().search(state.get("query", ""))}


def synthesizer(state: GraphState) -> dict[str, Any]:
    """Use one shared, extractive synthesis path for every specialist."""

    evidence = state.get("evidence", [])
    if not evidence:
        return {
            "response": "I don't have enough evidence in the knowledge base to answer that.",
            "citations": [],
            "abstained": True,
            "generation_mode": "abstention",
        }
    citations = [item["citation"] for item in evidence]
    generated, generation_mode = try_generate_grounded_answer(
        state.get("query", ""),
        state.get("domain", "GENERAL"),
        evidence,
        state.get("graph_context", []),
    )
    if generated:
        response = generated
        missing_citations = [citation for citation in citations if citation not in response]
        if missing_citations:
            response = f"{response}\nSources: {', '.join(f'[{citation}]' for citation in missing_citations)}"
    else:
        excerpts = [f"{item['text']} [{item['citation']}]" for item in evidence]
        response = "Based on the knowledge base: " + " ".join(excerpts)
    return {
        "response": response,
        "citations": citations,
        "abstained": False,
        "generation_mode": generation_mode,
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
workflow.add_node("hr_agent", hr_agent)
workflow.add_node("policy_agent", policy_agent)
workflow.add_node("precedent_agent", precedent_agent)
workflow.add_node("finance_agent", finance_agent)
workflow.add_node("engineering_agent", engineering_agent)
workflow.add_node("general_agent", general_agent)
workflow.add_node("graph_context", graph_context_retriever)
workflow.add_node("retriever", real_retriever)
workflow.add_node("synthesizer", synthesizer)
workflow.add_node("evaluator", evaluator)
workflow.set_entry_point("query_analyzer")
workflow.add_conditional_edges(
    "query_analyzer",
    _route_domain,
    {domain: f"{domain.lower()}_agent" for domain in DOMAINS},
)
for _domain in DOMAINS:
    workflow.add_edge(f"{_domain.lower()}_agent", "graph_context")
workflow.add_edge("graph_context", "retriever")
workflow.add_edge("retriever", "synthesizer")
workflow.add_edge("synthesizer", "evaluator")
workflow.add_edge("evaluator", END)
app_graph = workflow.compile()


__all__ = [
    "DOMAINS",
    "GraphState",
    "app_graph",
    "engineering_agent",
    "evaluator",
    "finance_agent",
    "general_agent",
    "graph_context_retriever",
    "hr_agent",
    "policy_agent",
    "precedent_agent",
    "query_analyzer",
    "real_retriever",
    "synthesizer",
]
