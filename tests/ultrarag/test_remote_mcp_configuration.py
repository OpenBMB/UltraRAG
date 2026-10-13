"""Remote MCP configs must pass FastMCP validation before transport starts."""

import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastmcp import Client

import ultrarag.client as client


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


@pytest.fixture
def remote_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, list[int]]:
    monkeypatch.setattr(client, "logger", logging.getLogger("remote-mcp-test"))
    monkeypatch.setattr(client, "node_status", False)
    checked_versions: list[int] = []
    monkeypatch.setattr(client, "check_node_version", checked_versions.append)
    directory = tmp_path / "remote"
    pipeline = tmp_path / "pipeline.yaml"
    write_yaml(
        pipeline,
        {"servers": {"remote": str(directory)}, "pipeline": ["remote.greet"]},
    )
    return pipeline, directory, checked_versions


def server_metadata(url: str) -> dict[str, Any]:
    return {
        "path": url,
        "tools": {"greet": {"input": {"name": "$name"}, "output": ["msg"]}},
    }


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_build_accepts_remote_configuration(
    remote_pipeline: tuple[Path, Path, list[int]],
    monkeypatch: pytest.MonkeyPatch,
    scheme: str,
) -> None:
    pipeline, directory, checked_versions = remote_pipeline
    url = f"{scheme}://example.invalid/mcp"
    write_yaml(directory / "parameter.yaml", {"path": url, "name": "UltraRAG"})
    captured: dict[str, Any] = {}

    class BuildClient:
        def __init__(self, config: dict[str, Any]) -> None:
            # Real FastMCP validation rejects the old one-element tuple here.
            self.validated_client = Client(config)
            captured.update(config)

        async def __aenter__(self) -> "BuildClient":
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> None:
            assert name == "build"
            parameter_file = Path(arguments["parameter_file"])
            write_yaml(parameter_file.with_name("server.yaml"), server_metadata(url))

    monkeypatch.setattr(client, "Client", BuildClient)
    asyncio.run(client.build(str(pipeline)))

    assert checked_versions == [20]
    assert captured["mcpServers"]["remote"]["args"] == ["-y", "mcp-remote", url]
    generated = yaml.safe_load(
        (pipeline.parent / "server" / "pipeline_server.yaml").read_text("utf-8")
    )
    assert generated["remote"]["path"] == url


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_runtime_accepts_remote_configuration(
    remote_pipeline: tuple[Path, Path, list[int]], scheme: str
) -> None:
    pipeline, directory, checked_versions = remote_pipeline
    url = f"{scheme}://example.invalid/mcp"
    write_yaml(
        pipeline.parent / "server" / "pipeline_server.yaml",
        {"remote": server_metadata(url)},
    )
    write_yaml(
        pipeline.parent / "parameter" / "pipeline_parameter.yaml",
        {"remote": {"name": "UltraRAG"}},
    )

    context = client.load_pipeline_context(str(pipeline))
    client.create_mcp_client(context["mcp_cfg"])

    assert checked_versions == [20]
    assert context["mcp_cfg"]["mcpServers"]["remote"]["args"] == [
        "-y",
        "mcp-remote",
        url,
    ]
