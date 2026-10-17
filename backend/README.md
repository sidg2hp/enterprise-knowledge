# Enterprise Knowledge Assistant backend

This backend is an offline-friendly recruiter knowledge vertical slice. It
accepts UTF-8 TXT and Markdown files, stores deterministic document/chunk
metadata in `data/documents.json`, and answers queries with a stdlib lexical
retriever. Queries abstain when no indexed chunk matches.

## Setup

Use Python 3.10.8 (the version is pinned in `.python-version`):

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
```

On Windows, use `.venv\\Scripts\\python.exe` in place of `.venv/bin/python`.

## Run

```bash
uvicorn main:app --reload
```

The optional `KNOWLEDGE_STORE_PATH` environment variable changes the JSON
store location. The default is `backend/data/documents.json`.

## Test

```bash
python -m pytest -q
```

The tests use temporary stores and do not require network access or API keys.

## API flow

```bash
curl -F "file=@candidate.md" http://localhost:8000/documents
curl http://localhost:8000/documents
curl -H "Content-Type: application/json" -d '{"query":"Which candidate knows Python?"}' http://localhost:8000/query
curl -X DELETE http://localhost:8000/documents/DOCUMENT_ID
```

## Deploy with Docker

```bash
docker build -t enterprise-knowledge-backend .
docker run --rm -p 8000:8000 -v "$(pwd)/data:/app/data" enterprise-knowledge-backend
```
