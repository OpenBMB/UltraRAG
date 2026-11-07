"""Shared workspace configuration is administrative; chat receives metadata only."""

import copy
import importlib

import pytest

from ui.backend import pipeline_manager as pm

CONFIG = {
    "index_backend": "milvus",
    "milvus": {"uri": "http://user:password@db", "token": "database-secret"},
}


@pytest.fixture
def app(tmp_path, monkeypatch):
    module = importlib.import_module("ui.backend.app")
    base = tmp_path / "ui/backend"
    base.mkdir(parents=True)
    prompts = tmp_path / "prompt"
    prompts.mkdir()
    (prompts / "example.jinja").write_text("original")
    monkeypatch.setattr(module, "BASE_DIR", base)
    monkeypatch.setattr(module, "AUTH_DB_PATH", tmp_path / "users.sqlite3")
    monkeypatch.setattr(pm, "load_kb_config", lambda: copy.deepcopy(CONFIG))
    monkeypatch.setattr(pm, "save_kb_config", lambda payload: None)
    monkeypatch.setattr(
        pm, "list_kb_files", lambda: {"index": [], "db_config": copy.deepcopy(CONFIG)}
    )
    monkeypatch.setattr(
        pm,
        "list_pipelines",
        lambda: [
            {
                "name": "example",
                "is_ready": True,
                "config": {"api_key": "pipeline-secret"},
            }
        ],
    )
    monkeypatch.setattr(
        pm, "load_pipeline", lambda _: {"_raw_yaml": "api_key: pipeline-secret"}
    )
    monkeypatch.setattr(
        pm, "load_parameters", lambda _: {"api_key": "parameter-secret"}
    )
    return module.create_app(admin_mode=True, no_mcp=True), prompts


@pytest.mark.parametrize("identity", [None, "reader"])
def test_chat_metadata_contains_no_workspace_credentials(app, identity):
    application, _ = app
    with application.test_client() as client:
        if identity:
            with client.session_transaction() as session:
                session["auth_user_id"] = identity
        for path in ("/api/pipelines", "/api/kb/config", "/api/kb/files"):
            response = client.get(path)
            assert response.status_code == 200
            assert not any(
                marker in response.get_data(as_text=True)
                for marker in ("pipeline-secret", "database-secret", "password@db")
            )
        assert client.post("/api/kb/config", json={}).status_code == 403
        assert client.get("/api/pipelines/example").status_code == 403
        assert client.get("/api/pipelines/example/parameters").status_code == 403


@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("GET", "/api/prompts", None),
        ("GET", "/api/prompts/example.jinja", None),
        ("POST", "/api/prompts", {"name": "new.jinja"}),
        ("PUT", "/api/prompts/example.jinja", {"content": "changed"}),
        ("DELETE", "/api/prompts/example.jinja", None),
        ("POST", "/api/prompts/example.jinja/rename", {"new_name": "changed.jinja"}),
    ],
)
def test_prompt_operations_require_administrator(app, method, path, payload):
    application, folder = app
    with application.test_client() as client:
        assert client.open(path, method=method, json=payload).status_code == 403
    assert (folder / "example.jinja").read_text() == "original"
    assert not (folder / "new.jinja").exists()


def test_admin_access_and_prompt_writes_are_preserved(app, tmp_path):
    application, folder = app
    outside = tmp_path / "outside"
    outside.mkdir()
    (folder / "link").symlink_to(outside, target_is_directory=True)
    with application.test_client() as client:
        with client.session_transaction() as session:
            session["auth_user_id"] = "admin"
        assert (
            client.get("/api/pipelines/example/parameters").get_json()["api_key"]
            == "parameter-secret"
        )
        assert (
            client.get("/api/kb/config").get_json()["milvus"]["token"]
            == "database-secret"
        )
        assert client.post("/api/kb/config", json=CONFIG).status_code == 200
        assert (
            client.post(
                "/api/prompts", json={"name": "safe.jinja", "content": "hello"}
            ).status_code
            == 200
        )
        assert (
            client.put(
                "/api/prompts/safe.jinja", json={"content": "updated"}
            ).status_code
            == 200
        )
        assert client.get("/api/prompts/safe.jinja").get_json()["content"] == "updated"
        assert (
            client.post(
                "/api/prompts/safe.jinja/rename", json={"new_name": "renamed.jinja"}
            ).status_code
            == 200
        )
        assert client.delete("/api/prompts/renamed.jinja").status_code == 200
        assert (
            client.post(
                "/api/prompts", json={"name": "link/escape.jinja", "content": "blocked"}
            ).status_code
            == 400
        )
    assert not (outside / "escape.jinja").exists()
