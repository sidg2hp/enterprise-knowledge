from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from graph import app_graph
from knowledge import DocumentStore, configure_document_store, ingest_document
from main import app


@pytest.fixture()
def store(tmp_path: Path):
    configured = DocumentStore(tmp_path / "documents.json")
    configure_document_store(configured)
    return configured


def test_ingestion_is_deterministic_and_has_stable_chunk_metadata():
    first = ingest_document("policy.md", b"# Remote Work\n\nEmployees may work remotely three days per week.")
    second = ingest_document("policy.md", b"# Remote Work\r\n\r\nEmployees may work remotely three days per week.\r\n")
    assert first["document_id"] == second["document_id"]
    assert first["content_sha256"] == second["content_sha256"]
    assert first["chunks"] == second["chunks"]
    assert first["chunks"][0]["citation"].startswith(first["document_id"] + "#chunk-")


def test_graph_retrieves_real_uploaded_content_and_citations(store):
    document = ingest_document("candidate.txt", b"Jordan has Python and LangGraph experience.")
    store.upsert(document)

    result = app_graph.invoke({"query": "Which candidate has Python experience?"})

    assert result["domain"] == "HR"
    assert result["abstained"] is False
    assert result["evidence"][0]["text"] == "Jordan has Python and LangGraph experience."
    assert result["citations"] == [result["evidence"][0]["citation"]]
    assert result["citations"][0] in result["response"]
    assert result["graph_context_status"]["local_enabled"] is True
    assert result["graph_context_status"]["local_result_count"] > 0
    assert result["graph_context"][0]["citation"] == result["evidence"][0]["citation"]

    assert store.delete(document["document_id"]) is True
    assert store.graph_search("Python") == []


def test_graph_abstains_without_matching_evidence(store):
    store.upsert(ingest_document("candidate.txt", b"Jordan has Python experience."))

    result = app_graph.invoke({"query": "What is the office parking allowance?"})

    assert result["abstained"] is True
    assert result["citations"] == []
    assert result["evidence"] == []
    assert "enough evidence" in result["response"]


def test_documents_api_upload_list_query_and_delete(store):
    client = TestClient(app)
    upload = client.post(
        "/documents",
        files={"file": ("handbook.md", b"The recruiter handbook requires structured interviews.", "text/markdown")},
    )
    assert upload.status_code == 201
    document_id = upload.json()["document_id"]

    listed = client.get("/documents")
    assert listed.status_code == 200
    assert listed.json()["documents"][0]["document_id"] == document_id

    query = client.post("/query", json={"query": "What does the recruiter handbook require?"})
    assert query.status_code == 200
    payload = query.json()
    assert payload["abstained"] is False
    assert payload["evidence"][0]["document_id"] == document_id
    assert payload["citations"] == [item["citation"] for item in payload["evidence"]]

    deleted = client.delete(f"/documents/{document_id}")
    assert deleted.status_code == 200
    assert client.get("/documents").json()["documents"] == []
    assert client.delete(f"/documents/{document_id}").status_code == 404


def test_documents_api_rejects_unsupported_files(store):
    client = TestClient(app)
    response = client.post("/documents", files={"file": ("notes.pdf", b"not supported", "application/pdf")})
    assert response.status_code == 400


def test_health_and_readiness_endpoints(store):
    client = TestClient(app)
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready", "document_count": 0}


def test_query_returns_quality_and_operational_metadata(store):
    store.upsert(ingest_document("engineering.md", b"All production API changes require peer review."))
    client = TestClient(app)
    response = client.post(
        "/query",
        headers={"X-Request-ID": "test-request-123"},
        json={"query": "What is required before a production API change?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"] == "test-request-123"
    assert response.headers["X-Request-ID"] == "test-request-123"
    assert payload["generation_mode"] == "extractive"
    assert payload["retrieval"]["mode"] == "bm25"
    assert payload["graph_context"]["status"]["enabled"] is False
    assert payload["evaluation"]["citation_validity"] == 1.0
