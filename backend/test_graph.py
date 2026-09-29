from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_hr_query():
    response = client.post("/query", json={"query": "Find a resume for Python"})
    assert response.status_code == 200
    data = response.json()
    assert data["domain"] == "HR"
    assert "HR-101" in data["citations"]

def test_policy_query():
    response = client.post("/query", json={"query": "What is the remote work policy?"})
    assert response.status_code == 200
    data = response.json()
    assert data["domain"] == "POLICY"
    assert "POL-202" in data["citations"]
