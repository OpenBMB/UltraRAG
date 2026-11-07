"""Connection checks must exercise the selected model, not just a public catalog."""

import importlib
from types import SimpleNamespace

import pytest
import requests


@pytest.fixture
def client(tmp_path, monkeypatch):
    module = importlib.import_module("ui.backend.app")
    monkeypatch.setattr(module, "AUTH_DB_PATH", tmp_path / "users.sqlite3")
    app = module.create_app(admin_mode=True, no_mcp=True)
    with app.test_client() as client:
        yield client


SETTINGS = {
    "provider": "custom",
    "baseUrl": "https://model.example/v1",
    "apiKey": "private-test-key",
    "model": "chosen-model",
}


def test_catalog_success_does_not_hide_unavailable_selected_model(client, monkeypatch):
    monkeypatch.setattr(
        requests, "get", lambda *a, **k: SimpleNamespace(status_code=200)
    )

    def post(url, **kwargs):
        assert kwargs["json"]["model"] == "chosen-model"
        return SimpleNamespace(status_code=503, text="private-test-key echoed upstream")

    monkeypatch.setattr(requests, "post", post)
    result = client.post("/api/ai/test", json=SETTINGS).get_json()
    assert result["success"] is False
    assert "503" in result["error"]
    assert "private-test-key" not in str(result)


def test_selected_model_must_return_completion_choices(client, monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: SimpleNamespace(status_code=200, json=lambda: {"data": []}),
    )
    assert client.post("/api/ai/test", json=SETTINGS).get_json()["success"] is False
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: SimpleNamespace(
            status_code=200, json=lambda: {"choices": [{"message": {"content": "ok"}}]}
        ),
    )
    assert client.post("/api/ai/test", json=SETTINGS).get_json() == {
        "success": True,
        "model": "chosen-model",
    }
