import pytest
from fastapi.testclient import TestClient

from dastavez import api


@pytest.fixture
def client(toy_answerer):
    api.app.dependency_overrides[api.get_answerer] = lambda: toy_answerer
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_ask_returns_answer_citations_and_snippets(client):
    r = client.post("/ask", json={"question": "What is the second tranche loan for street vendors?"})
    assert r.status_code == 200
    body = r.json()
    assert body["not_found"] is False
    assert body["citations"][0]["doc_id"] == "svanidhi" and body["citations"][0]["page"] == 3
    assert body["sentences"][0]["citation"]["snippet"]
    assert "[svanidhi p.3]" in body["answer"]


def test_ask_not_found(client):
    r = client.post("/ask", json={"question": "Who won the 1983 cricket world cup?"})
    assert r.status_code == 200 and r.json()["not_found"] is True and r.json()["citations"] == []


def test_ask_validation(client):
    assert client.post("/ask", json={"question": "x"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "mode": "bogus"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "k": 0}).status_code == 422


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Dastavez" in r.text
