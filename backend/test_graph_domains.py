"""Focused coverage for multi-domain routing and optional graph context."""

from __future__ import annotations

from graph import app_graph
from graph_adapter import Neo4jConfig, Neo4jGraphAdapter, is_safe_read_query


def test_all_supported_domains_route_to_specialist_metadata():
    examples = {
        "HR": "Which candidate passed the interview?",
        "POLICY": "Which policy controls the audit procedure?",
        "PRECEDENT": "What historical precedent supports this decision?",
        "FINANCE": "What is the budget forecast and expense?",
        "ENGINEERING": "How should the API deployment architecture work?",
        "GENERAL": "Where is the company handbook?",
    }

    for expected, query in examples.items():
        result = app_graph.invoke({"query": query})
        assert result["domain"] == expected
        assert result["domain_metadata"]["domain"] == expected
        assert result["agent_metadata"]["domain"] == expected
        assert result["agent_metadata"]["name"].endswith("_agent")


def test_unconfigured_graph_adapter_is_a_noop():
    adapter = Neo4jGraphAdapter(Neo4jConfig())

    assert adapter.configured is False
    assert adapter.search_context("budget") == []
    assert adapter.status() == {"enabled": False, "configured": False, "reason": "not_configured"}


def test_graph_adapter_allows_reads_and_rejects_writes():
    assert is_safe_read_query("MATCH (n) RETURN n")
    assert is_safe_read_query("OPTIONAL MATCH (n) RETURN n")
    assert not is_safe_read_query("MATCH (n) SET n.secret = 'x' RETURN n")
    assert not is_safe_read_query("CREATE (n:Secret) RETURN n")
    assert not is_safe_read_query("MATCH (n) RETURN n; MATCH (m) RETURN m")
