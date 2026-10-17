from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from graph import app_graph
from knowledge import DocumentError, get_document_store, ingest_document


app = FastAPI(title="Enterprise Knowledge Assistant API")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Enterprise Knowledge Assistant API"}


@app.post("/query")
def run_query(req: QueryRequest):
    result = app_graph.invoke({"query": req.query.strip()})
    return {
        "response": result.get("response"),
        "domain": result.get("domain", "GENERAL"),
        "citations": result.get("citations", []),
        "evidence": result.get("evidence", []),
        "abstained": result.get("abstained", False),
    }


@app.post("/documents", status_code=201)
async def upload_document(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="a filename is required")
    try:
        document = ingest_document(filename, await file.read())
        get_document_store().upsert(document)
    except DocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
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
