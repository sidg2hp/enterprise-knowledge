import io
import json

import pytest

from observability import JsonEventLogger, log_ingestion_event, log_query_event, new_request_id


def test_query_event_is_json_and_excludes_query_text():
    stream = io.StringIO()
    logger = JsonEventLogger(stream=stream)
    request_id = new_request_id()
    payload = log_query_event(
        request_id,
        logger=logger,
        query="What is the secret document text?",
        result_count=2,
        citation_count=1,
        abstained=False,
    )
    decoded = json.loads(stream.getvalue())
    assert decoded == payload
    assert decoded["event"] == "query"
    assert decoded["request_id"] == request_id
    assert decoded["query_length"] == len("What is the secret document text?")
    assert "What is the secret document text?" not in stream.getvalue()
    assert "text" not in decoded
    assert "content" not in decoded


def test_ingestion_event_contains_metadata_but_no_document_body():
    stream = io.StringIO()
    logger = JsonEventLogger(stream=stream)
    payload = log_ingestion_event(
        logger=logger,
        filename="handbook.md",
        size_bytes=42,
        content_sha256="a" * 64,
        chunk_count=3,
    )
    decoded = json.loads(stream.getvalue())
    assert decoded == payload
    assert decoded["event"] == "ingestion"
    assert decoded["chunk_count"] == 3
    assert "document body" not in stream.getvalue()
    assert "text" not in decoded


def test_logger_rejects_content_fields_in_generic_events():
    with pytest.raises(ValueError):
        JsonEventLogger(stream=io.StringIO()).emit("unsafe", text="document body")
