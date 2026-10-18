from __future__ import annotations

import json
from pathlib import Path

from graph_rag import (
    GraphRAGIndex,
    Neo4jGraphWriter,
    UPSERT_RELATION_CYPHER,
    extract_entities,
    extract_relations,
)
from knowledge import ingest_document


def _document() -> dict:
    return ingest_document(
        "platform.md",
        b"# Platform Apollo\n\nAlice uses Python and reports to Bob.\n\nBob supports Platform Apollo.",
    )


def test_entity_and_relation_extraction_is_deterministic():
    text = "Alice uses Python and reports to Bob."
    first_entities = extract_entities(text)
    second_entities = extract_entities(text)
    assert first_entities == second_entities
    names = {item["name"] for item in first_entities}
    assert {"Alice", "Python", "Bob"}.issubset(names)

    relations = extract_relations(text, first_entities)
    relation_types = {item["relation_type"] for item in relations}
    assert {"uses", "reports_to"}.issubset(relation_types)
    assert all(item["relation_id"].startswith("relation_") for item in relations)


def test_graph_persists_atomically_and_reloads(tmp_path: Path):
    path = tmp_path / "nested" / "graph.json"
    graph = GraphRAGIndex(path)
    document = _document()
    graph.index_document(document)

    assert path.exists()
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == 1
    assert document["document_id"] in persisted["documents"]
    assert persisted["chunks"]
    assert GraphRAGIndex(path).status()["document_count"] == 1


def test_query_retrieval_returns_paths_context_and_citations(tmp_path: Path):
    graph = GraphRAGIndex(tmp_path / "graph.json")
    document = _document()
    graph.index_document(document)

    result = graph.retrieve("Who reports to Bob and uses Python?")

    assert result["paths"]
    assert result["contexts"]
    assert result["citations"]
    assert set(result["citations"]) == {chunk["citation"] for chunk in document["chunks"]}
    for context in result["contexts"]:
        assert context["citation"] in result["citations"]
        assert context["citation"] in context["source_chunk_citations"]
        assert all(citation in result["citations"] for path in context["paths"] for citation in path["citations"])


def test_delete_removes_document_chunks_entities_relations_and_retrieval(tmp_path: Path):
    graph = GraphRAGIndex(tmp_path / "graph.json")
    document = _document()
    graph.index_document(document)
    assert graph.delete_document(document["document_id"]) is True
    assert graph.delete_document(document["document_id"]) is False
    assert graph.status()["document_count"] == 0
    assert graph.status()["chunk_count"] == 0
    assert graph.retrieve("Python")["contexts"] == []


def test_safe_fallback_for_corrupt_index_and_unconfigured_neo4j(tmp_path: Path):
    path = tmp_path / "graph.json"
    path.write_text("not json", encoding="utf-8")
    graph = GraphRAGIndex(path)
    assert graph.retrieve("anything") == {"query": "anything", "paths": [], "contexts": [], "citations": []}
    assert Neo4jGraphWriter().upsert({}, [], []) is False


def test_neo4j_writer_uses_fixed_parameterized_relation_cypher():
    assert "$source_entity_id" in UPSERT_RELATION_CYPHER
    assert "$target_entity_id" in UPSERT_RELATION_CYPHER
    assert "$relation_type" in UPSERT_RELATION_CYPHER
    assert "RELATED_TO" in UPSERT_RELATION_CYPHER
    assert "{relation_type}" not in UPSERT_RELATION_CYPHER
