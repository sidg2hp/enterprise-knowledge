from fastapi import FastAPI
from pydantic import BaseModel
from graph import app_graph

app = FastAPI(title="Enterprise Knowledge Assistant API")

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Enterprise Knowledge Assistant API"}

@app.post("/query")
def run_query(req: QueryRequest):
    result = app_graph.invoke({"query": req.query})
    return {
        "response": result.get("response"),
        "domain": result.get("domain"),
        "citations": result.get("citations")
    }

