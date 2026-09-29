from fastapi.testclient import TestClient

from pattern_studio.storage import JSONLStore
from pattern_studio.webapp import create_app


def test_dashboard_serves_saved_analysis(tmp_path):
    JSONLStore(tmp_path).write("clusters", [{"id": "a", "name": "Sign in help", "chat_ids": ["1"]}])
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/api/analysis").json()["clusters"][0]["name"] == "Sign in help"
    page = client.get("/")
    assert page.status_code == 200
    assert "Conversation Pattern Studio" in page.text
