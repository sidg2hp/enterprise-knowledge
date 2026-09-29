from fastapi import FastAPI

app = FastAPI(title="Enterprise Knowledge Assistant API")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Enterprise Knowledge Assistant API"}
