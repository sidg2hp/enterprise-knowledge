# Enterprise Knowledge Assistant backend

This backend is an offline-friendly evidence-grounded decision-intelligence
slice. It accepts UTF-8 TXT and Markdown files, stores deterministic
document/chunk metadata in `data/documents.json`, and answers queries with a
real BM25 retriever by default. Optional vector retrieval, FAISS, Ollama, and
read-only Neo4j adapters are available without making them required for CI.
Queries abstain when no indexed chunk matches.

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
store location. The default is `backend/data/documents.json`. Set
`KNOWLEDGE_RETRIEVER_MODE=hybrid` or `vector` to exercise vector retrieval;
the default `bm25` mode is deterministic and offline-safe.

## Test

```bash
python -m pytest -q
```

The tests use temporary stores and do not require network access or API keys.

Run the versioned offline evaluation fixture from the repository root:

```bash
python -m backend.evaluate
```

It reports Recall@5, MRR@5, citation validity, a lexical groundedness proxy,
and abstention accuracy. The current metrics are regression signals, not a
claim of universal answer quality.

Optional integrations use the pinned packages in
`requirements.optional.lock`. Ollama is configured with `OLLAMA_BASE_URL`,
`OLLAMA_EMBEDDING_MODEL`, and `OLLAMA_CHAT_MODEL`. Neo4j uses
`NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD`; its adapter rejects
write Cypher and degrades to disabled context when unavailable.

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
