"""Regression coverage for application logging and configuration boundaries."""

from pathlib import Path

import pytest
import yaml

from ultrarag import client
from ultrarag.configuration import Configuration
from ultrarag.mcp_exceptions import NodeNotInstalledError


@pytest.mark.parametrize("content", ["", "null", "# comment\n"])
def test_empty_yaml_is_a_mapping(tmp_path: Path, content: str) -> None:
    path = tmp_path / "parameter.yaml"
    path.write_text(content, encoding="utf-8")
    assert Configuration.load_config(path) == {}
    assert Configuration.load_parameter_config(path) == {}


@pytest.mark.parametrize("content", ["- value", "42", "false", "text"])
def test_invalid_yaml_shape_reports_file(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="must be a YAML mapping.*bad.yaml"):
        Configuration.load_config(path)


def test_missing_built_server_fails_with_rebuild_hint(tmp_path: Path) -> None:
    pipeline = tmp_path / "test.yaml"
    pipeline.write_text(
        "servers: {missing: servers/missing}\npipeline: []", encoding="utf-8"
    )
    (tmp_path / "server").mkdir()
    (tmp_path / "server" / "test_server.yaml").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing built server.*missing.*Rebuild"):
        client.load_pipeline_context(str(pipeline), no_mcp=True)


def test_remote_dependency_error_is_recoverable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client, "node_status", False)

    def unavailable(version: int) -> None:
        raise NodeNotInstalledError("Node.js is unavailable")

    monkeypatch.setattr(client, "check_node_version", unavailable)
    with pytest.raises(NodeNotInstalledError, match="unavailable"):
        client._server_launch_config("remote", "https://example.invalid/mcp")
    assert not client.node_status


def test_atomic_save_retains_previous_file_if_serialization_fails(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"
    Configuration.save_config(path, {"message": "中文", "value": 42})
    assert Configuration.load_config(path) == {"message": "中文", "value": 42}
    original = path.read_bytes()
    with pytest.raises(yaml.representer.RepresenterError):
        Configuration.save_config(path, {"unsupported": object()})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
