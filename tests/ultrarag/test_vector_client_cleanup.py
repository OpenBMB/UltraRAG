"""Vector client ownership includes failures and local idle connections."""

from types import SimpleNamespace

import pytest

from ui.backend import pipeline_manager as pm


@pytest.mark.parametrize("backend", ["milvus", "qdrant"])
def test_listing_failure_closes_the_owned_database_client(
    tmp_path, monkeypatch, backend
):
    closed = []

    def fail():
        raise RuntimeError("database unavailable")

    client = SimpleNamespace(
        list_collections=fail, get_collections=fail, close=lambda: closed.append(True)
    )
    monkeypatch.setattr(pm, "load_kb_config", lambda: {"index_backend": backend})
    monkeypatch.setattr(pm, "_get_" + backend + "_client", lambda: client)
    for key in ("KB_RAW_DIR", "KB_CORPUS_DIR", "KB_CHUNKS_DIR"):
        monkeypatch.setattr(pm, key, tmp_path / key)
    assert pm.list_kb_files()["db_status"] == "error"
    assert closed == [True]


@pytest.mark.parametrize("backend", ["milvus", "qdrant"])
def test_deletion_failure_closes_the_owned_database_client(monkeypatch, backend):
    closed = []

    def fail(*args):
        raise RuntimeError("database unavailable")

    client = SimpleNamespace(
        has_collection=fail, collection_exists=fail, close=lambda: closed.append(True)
    )
    monkeypatch.setattr(pm, "_get_" + backend + "_client", lambda: client)
    with pytest.raises(RuntimeError):
        getattr(pm, "_delete_" + backend + "_collection")("example")
    assert closed == [True]


def test_local_milvus_disables_idle_pings_without_changing_remote_options(monkeypatch):
    calls = []
    monkeypatch.setattr(pm, "MilvusClient", lambda **kwargs: calls.append(kwargs))
    for uri in ("/tmp/test.db", "http://localhost:19530"):
        monkeypatch.setattr(pm, "load_kb_config", lambda: {"milvus": {"uri": uri}})
        pm._get_milvus_client()
    assert calls[0]["grpc_options"] == {"grpc.keepalive_permit_without_calls": False}
    assert "grpc_options" not in calls[1]
