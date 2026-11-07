"""Default deployments must not accept publicly forgeable login cookies."""

import importlib
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from flask import Flask
from flask.sessions import SecureCookieSessionInterface

from ui.backend.session_secret import load_session_secret


def test_public_development_secret_cannot_forge_admin_login(
    tmp_path: Path, monkeypatch
) -> None:
    module = importlib.import_module("ui.backend.app")
    monkeypatch.delenv("ULTRARAG_SESSION_SECRET", raising=False)
    monkeypatch.setattr(module, "AUTH_DB_PATH", tmp_path / "users.sqlite3")
    app = module.create_app(admin_mode=True, no_mcp=True)
    attacker = Flask("public-key")
    attacker.secret_key = "ultrarag-dev-session-secret"
    serializer = SecureCookieSessionInterface().get_signing_serializer(attacker)
    forged = serializer.dumps({"auth_user_id": "admin"})
    with app.test_client() as client:
        client.set_cookie(app.config["SESSION_COOKIE_NAME"], forged)
        assert client.get("/api/auth/me").get_json()["logged_in"] is False


def test_secret_is_shared_by_workers_persistent_and_private(tmp_path, monkeypatch):
    monkeypatch.delenv("ULTRARAG_SESSION_SECRET", raising=False)
    path = tmp_path / "db" / "session.secret"
    with ThreadPoolExecutor(8) as pool:
        values = list(pool.map(lambda _: load_session_secret(path), range(24)))
    assert len(set(values)) == 1
    assert len(values[0]) == 64
    assert load_session_secret(path) == values[0]
    assert load_session_secret(tmp_path / "other" / "session.secret") != values[0]
    assert sorted(p.name for p in path.parent.iterdir()) == ["session.secret"]
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_explicit_environment_secret_does_not_create_a_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ULTRARAG_SESSION_SECRET", "explicit-test-configuration")
    path = tmp_path / "session.secret"
    assert load_session_secret(path) == "explicit-test-configuration"
    assert not path.exists()


def test_invalid_key_file_fails_closed(tmp_path, monkeypatch):
    monkeypatch.delenv("ULTRARAG_SESSION_SECRET", raising=False)
    path = tmp_path / "session.secret"
    path.write_text("")
    with pytest.raises(RuntimeError, match="Invalid session signing key"):
        load_session_secret(path)


def test_valid_login_survives_application_restart(tmp_path, monkeypatch):
    module = importlib.import_module("ui.backend.app")
    monkeypatch.delenv("ULTRARAG_SESSION_SECRET", raising=False)
    monkeypatch.setattr(module, "AUTH_DB_PATH", tmp_path / "users.sqlite3")
    first = module.create_app(admin_mode=True, no_mcp=True).test_client()
    response = first.post(
        "/api/auth/login", json={"username": "admin", "password": "12345678"}
    )
    assert response.status_code == 200
    cookie = first.get_cookie("session")
    restarted = module.create_app(admin_mode=True, no_mcp=True).test_client()
    restarted.set_cookie("session", cookie.value)
    assert restarted.get("/api/auth/me").get_json()["logged_in"] is True
