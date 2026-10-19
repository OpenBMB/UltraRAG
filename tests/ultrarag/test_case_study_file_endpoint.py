"""Regression tests for the case-study /file endpoint static-root guard."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "script"))

import case_study


@pytest.fixture
def client():
    return TestClient(case_study.app)


def file_outside_root(tmp_path: Path) -> Path:
    target = tmp_path / "secret.txt"
    target.write_text("top-secret", encoding="utf-8")
    return target


def test_empty_static_roots_forbids_any_path(client, tmp_path, monkeypatch):
    monkeypatch.setattr(case_study.STATE, "static_roots", [])
    target = file_outside_root(tmp_path)
    response = client.get("/file", params={"path": str(target)})
    assert response.status_code == 403
    assert "top-secret" not in response.text


def test_file_under_static_root_is_served(client, tmp_path, monkeypatch):
    target = tmp_path / "images" / "x.png"
    target.parent.mkdir()
    target.write_bytes(b"fake-png-bytes")
    monkeypatch.setattr(case_study.STATE, "static_roots", [str(tmp_path)])
    response = client.get("/file", params={"path": str(target)})
    assert response.status_code == 200
    assert response.content == b"fake-png-bytes"


def test_file_outside_static_root_is_forbidden(client, tmp_path, monkeypatch):
    target = file_outside_root(tmp_path)
    monkeypatch.setattr(case_study.STATE, "static_roots", [str(tmp_path / "other")])
    response = client.get("/file", params={"path": str(target)})
    assert response.status_code == 403


def test_missing_file_under_root_returns_404(client, tmp_path, monkeypatch):
    monkeypatch.setattr(case_study.STATE, "static_roots", [str(tmp_path)])
    response = client.get("/file", params={"path": str(tmp_path / "missing.txt")})
    assert response.status_code == 404
