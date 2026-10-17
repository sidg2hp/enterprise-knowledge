from pathlib import Path
from time import perf_counter

import os

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:  # Support both `uvicorn main:app` from backend/ and package imports.
    from .graph import app_graph
    from .knowledge import DocumentError, get_document_store, ingest_document
    from .evaluation import evaluate_grounding
    from .observability import ensure_request_id, get_event_logger
except ImportError:  # pragma: no cover - exercised by the direct server command
    from graph import app_graph
    from knowledge import DocumentError, get_document_store, ingest_document
    from evaluation import evaluate_grounding
    from observability import ensure_request_id, get_event_logger


app = FastAPI(
    title="Enterprise Knowledge Assistant API",
    version="0.2.0",
    description="Evidence-grounded document search with deterministic citations.",
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

event_logger = get_event_logger()


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = ensure_request_id(request.headers.get("X-Request-ID"))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


@app.get("/")
def read_root():
    return {
        "message": "Enterprise Knowledge Assistant API",
        "docs": "/docs",
        "health": "/healthz",
    }


@app.get("/healthz")
def health_check():
    """Liveness endpoint used by local containers and deployment probes."""

    return {"status": "ok"}


@app.get("/readyz")
def readiness_check():
    """Report whether the JSON-backed knowledge store can be read."""

    try:
        documents = get_document_store().list_documents()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="knowledge store is unavailable") from exc
    return {"status": "ready", "document_count": len(documents)}


@app.post("/query")
def run_query(req: QueryRequest, request: Request):
    started = perf_counter()
    request_id = request.state.request_id
    try:
        result = app_graph.invoke({"query": req.query.strip()})
        evidence = result.get("evidence", [])
        citations = result.get("citations", [])
        evaluation = evaluate_grounding(
            result.get("response", ""), evidence, citations, result.get("abstained", False)
        )
        latency_ms = (perf_counter() - started) * 1000
        event_logger.query(
            request_id,
            query=req.query,
            result_count=len(evidence),
            citation_count=len(citations),
            abstained=result.get("abstained", False),
            latency_ms=latency_ms,
            domain=result.get("domain", "GENERAL"),
            generation_mode=result.get("generation_mode", "unknown"),
            retriever_mode=evidence[0].get("retriever_mode", "none") if evidence else "none",
        )
        return {
            "request_id": request_id,
            "response": result.get("response"),
            "domain": result.get("domain", "GENERAL"),
            "domain_metadata": result.get("domain_metadata", {}),
            "agent": result.get("agent_metadata", {}),
            "generation_mode": result.get("generation_mode", "unknown"),
            "retrieval": {
                "mode": evidence[0].get("retriever_mode", "none") if evidence else "none",
                "vector_backend": evidence[0].get("retrieval", {}).get("vector_backend") if evidence else None,
                "result_count": len(evidence),
            },
            "graph_context": {
                "status": result.get("graph_context_status", {"enabled": False, "reason": "unknown"}),
                "result_count": len(result.get("graph_context", [])),
            },
            "citations": citations,
            "evidence": evidence,
            "abstained": result.get("abstained", False),
            "evaluation": evaluation,
        }
    except Exception as exc:
        event_logger.query(
            request_id,
            query=req.query,
            latency_ms=(perf_counter() - started) * 1000,
            error=type(exc).__name__,
        )
        raise


@app.post("/documents", status_code=201)
async def upload_document(request: Request, file: UploadFile = File(...)):
    request_id = request.state.request_id
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="a filename is required")
    try:
        document = ingest_document(filename, await file.read())
        get_document_store().upsert(document)
    except DocumentError as exc:
        event_logger.ingestion(request_id, filename=filename, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    event_logger.ingestion(
        request_id,
        filename=document["filename"],
        size_bytes=document["size_bytes"],
        content_sha256=document["content_sha256"],
        chunk_count=len(document["chunks"]),
    )
    return {
        "request_id": request_id,
        "document_id": document["document_id"],
        "filename": document["filename"],
        "content_sha256": document["content_sha256"],
        "chunk_count": len(document["chunks"]),
    }


@app.get("/documents")
def list_documents():
    return {"documents": get_document_store().list_documents()}


@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    if not get_document_store().delete(document_id):
        raise HTTPException(status_code=404, detail="document not found")
    return {"deleted": True, "document_id": document_id}
