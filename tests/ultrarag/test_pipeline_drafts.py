"""New builder pipelines remain selectable before their first build."""

import importlib
from pathlib import Path

import pytest
import yaml

from ui.backend import pipeline_manager as pm


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    demos, experiments = tmp_path / "demos", tmp_path / "experiments"
    demos.mkdir()
    experiments.mkdir()
    monkeypatch.setattr(pm, "DEMO_PIPELINES_DIR", demos)
    monkeypatch.setattr(pm, "EXPERIMENT_PIPELINES_DIR", experiments)
    monkeypatch.setattr(pm, "PIPELINE_SEARCH_DIRS", (demos, experiments))
    return demos, experiments


@pytest.mark.parametrize("name", ["NewPipeline", "my_pipeline"])
def test_new_unbuilt_pipeline_is_listed_as_a_draft(workspace, name):
    pm.save_pipeline({"name": name, "pipeline": ["sayhello.say_hello"]})
    entries = pm.list_pipelines()
    assert [entry["name"] for entry in entries] == [name]
    assert entries[0]["is_ready"] is False


def test_create_rejects_duplicate_without_overwriting(workspace):
    pm.save_pipeline({"name": "example", "pipeline": ["sayhello.say_hello"]})
    path = pm.pipeline_path("example")
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        pm.save_pipeline({"name": "example", "pipeline": []})
    assert path.read_bytes() == original


def test_create_preserves_explicit_server_aliases_and_mapped_steps(workspace):
    pm.save_pipeline(
        {
            "name": "alias",
            "servers": {"greeter": "servers/sayhello"},
            "pipeline": [{"greeter.say_hello": {"output": {"message": "answer"}}}],
        }
    )
    assert pm.load_pipeline("alias")["servers"] == {"greeter": "servers/sayhello"}


def test_saved_existing_pipeline_updates_in_its_original_directory(workspace):
    demos, _ = workspace
    path = demos / "lowercase.yaml"
    path.write_text("servers: {}\npipeline: []\n")
    pm.save_pipeline_yaml("lowercase", "servers: {}\npipeline: [sayhello.say_hello]\n")
    assert yaml.safe_load(path.read_text())["pipeline"] == ["sayhello.say_hello"]


def test_create_api_returns_conflict_for_existing_name(
    workspace, tmp_path, monkeypatch
):
    module = importlib.import_module("ui.backend.app")
    monkeypatch.setattr(module, "AUTH_DB_PATH", tmp_path / "db/users.sqlite3")
    app = module.create_app(admin_mode=True, no_mcp=True)
    with app.test_client() as client:
        with client.session_transaction() as session:
            session["auth_user_id"] = "admin"
        payload = {"name": "fresh", "pipeline": []}
        assert client.post("/api/pipelines", json=payload).status_code == 201
        assert client.post("/api/pipelines", json=payload).status_code == 409
        assert client.get("/api/pipelines").get_json()[0]["name"] == "fresh"


@pytest.mark.parametrize("name", ["../escape", "a/b", "", "a" * 129])
def test_create_rejects_unsafe_names(workspace, name):
    with pytest.raises(pm.PipelineManagerError):
        pm.save_pipeline({"name": name, "pipeline": []})


def test_create_rejects_absolute_names(workspace, tmp_path):
    destination = tmp_path / "outside" / "escape"
    with pytest.raises(pm.PipelineManagerError):
        pm.save_pipeline({"name": str(destination), "pipeline": []})
    assert not destination.with_suffix(".yaml").exists()


def test_rename_moves_both_build_artifacts_without_changing_directory(workspace):
    pm.save_pipeline({"name": "original", "pipeline": ["sayhello.say_hello"]})
    old = pm.pipeline_path("original")
    for folder, suffix in [("parameter", "parameter"), ("server", "server")]:
        (old.parent / folder).mkdir()
        (old.parent / folder / f"original_{suffix}.yaml").write_text("{}\n")
    pm.rename_pipeline("original", "Renamed")
    renamed = pm._find_pipeline_file("Renamed")
    assert renamed.parent == old.parent
    assert all(
        (old.parent / folder / f"Renamed_{folder}.yaml").exists()
        for folder in ["parameter", "server"]
    )
    assert pm.list_pipelines()[0]["is_ready"] is True


def test_deleted_pipeline_cannot_reuse_stale_build_artifacts(workspace):
    pm.save_pipeline({"name": "example", "pipeline": ["sayhello.say_hello"]})
    path = pm.pipeline_path("example")
    for folder in ["parameter", "server"]:
        (path.parent / folder).mkdir()
        (path.parent / folder / f"example_{folder}.yaml").write_text("{}\n")
    pm.delete_pipeline("example")
    pm.save_pipeline({"name": "example", "pipeline": []})
    assert pm.list_pipelines()[0]["is_ready"] is False


def test_editing_pipeline_invalidates_metadata_but_preserves_parameters(workspace):
    pm.save_pipeline({"name": "example", "pipeline": []})
    path = pm.pipeline_path("example")
    for folder in ["parameter", "server"]:
        (path.parent / folder).mkdir()
        (path.parent / folder / f"example_{folder}.yaml").write_text("custom: value\n")
    pm.save_pipeline_yaml("example", "servers: {}\npipeline: [sayhello.say_hello]\n")
    assert not (path.parent / "server/example_server.yaml").exists()
    assert (
        path.parent / "parameter/example_parameter.yaml"
    ).read_text() == "custom: value\n"
    assert pm.list_pipelines()[0]["is_ready"] is False


def test_build_returns_thread_scoped_logs_and_removes_handler(workspace, monkeypatch):
    import logging
    import threading

    pm.save_pipeline({"name": "example", "pipeline": []})
    logger = logging.getLogger("UltraRAG")
    handlers = list(logger.handlers)

    async def fake_build(*args, **kwargs):
        logger.warning("本次构建日志")
        other = threading.Thread(target=lambda: logger.warning("other-session"))
        other.start()
        other.join()

    monkeypatch.setattr(pm, "_ensure_client_funcs", lambda: {"build": fake_build})
    result = pm.build("example", include_logs=True)
    assert any("本次构建日志" in line for line in result["logs"])
    assert not any("other-session" in line for line in result["logs"])
    assert list(logger.handlers) == handlers

    async def fail_build(*args, **kwargs):
        logger.warning("failure details")
        raise ValueError("invalid tool")

    monkeypatch.setattr(pm, "_ensure_client_funcs", lambda: {"build": fail_build})
    with pytest.raises(pm.PipelineManagerError, match="invalid tool") as error:
        pm.build("example", include_logs=True)
    assert any("failure details" in line for line in error.value.logs)
    assert list(logger.handlers) == handlers
