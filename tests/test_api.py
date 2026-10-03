import pytest
from fastapi.testclient import TestClient

from api.main import app, get_rag
from sellerpolicy.config import Settings
from sellerpolicy.pipeline import SellerPolicyRAG


@pytest.fixture
def client(rag):
    app.dependency_overrides[get_rag] = lambda: rag
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["index_ready"] is True
    assert body["llm_enabled"] is False


def test_ask_returns_answer_and_sources(client):
    response = client.post("/ask", json={"question": "What is the SAFE-T claim deadline?", "mode": "hybrid", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["found"] and "30 days" in body["answer"]
    assert body["sources"][0]["doc_id"] == "returns"
    assert any(s["cited"] for s in body["sources"])


def test_ask_validation_error(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "mode": "magic"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "top_k": 99}).status_code == 422


def test_ask_blank_question(client):
    assert client.post("/ask", json={"question": "     "}).status_code == 422


def test_ask_503_when_index_missing(tmp_path):
    empty = SellerPolicyRAG(settings=Settings(_env_file=None, embedding_provider="hashing", storage_dir=tmp_path / "x"))
    app.dependency_overrides[get_rag] = lambda: empty
    try:
        assert TestClient(app).post("/ask", json={"question": "What is ODR?"}).status_code == 503
    finally:
        app.dependency_overrides.clear()
