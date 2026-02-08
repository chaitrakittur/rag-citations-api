import os
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OPENAI_API_KEY set")
def test_ingest_and_ask():
    ingest = client.post("/ingest", json={
        "source_id": "demo_doc",
        "text": "FastAPI is a Python web framework. Streamlit is used for data apps. This system tracks expenses.",
        "metadata": {"type": "demo"}
    })
    assert ingest.status_code == 200
    assert ingest.json()["chunks_added"] >= 1

    ask = client.post("/ask", json={"question": "What is FastAPI used for?"})
    assert ask.status_code == 200
    data = ask.json()
    assert "answer" in data
    assert "citations" in data
