"""Local clients and nested pipelines must preserve their owning context."""

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ultrarag import client
from ultrarag.execution import LocalExecutionClient
from ultrarag.local_server import LocalServer


def test_nested_local_client_keeps_registrations_until_outer_exit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "server.py"
    source.write_text(
        "from ultrarag.server import UltraRAG_MCP_Server\n"
        "app = UltraRAG_MCP_Server('nested')\n"
        "@app.tool()\ndef hello() -> dict:\n    return {'hello': 'world'}\n",
        encoding="utf-8",
    )
    adapter = LocalExecutionClient(
        {"mcpServers": {}}, {"nested": {"path": str(source)}}
    )

    async def exercise() -> None:
        async with adapter:
            first = adapter.local["nested"]
            async with adapter:
                assert adapter.local["nested"] is first
            assert (await adapter.call_tool("hello", {})).data == {"hello": "world"}
        assert not adapter.local
        assert not adapter._module_names

    asyncio.run(exercise())


def test_demo_stream_failure_propagates_and_still_runs_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "servers" / "generation"
    (folder / "src").mkdir(parents=True)
    cleanup = tmp_path / "cleanup.txt"
    (folder / "src" / "generation.py").write_text(
        "from pathlib import Path\n"
        "from ultrarag.server import UltraRAG_MCP_Server\n"
        "app = UltraRAG_MCP_Server('generation')\n"
        "@app.tool(output='prompt_ls->ans_ls')\n"
        "def generate(prompt_ls: list[str]) -> dict:\n    return {'ans_ls': prompt_ls}\n"
        "@app.tool(output='->None')\n"
        f"def vllm_shutdown() -> None:\n    Path({str(cleanup)!r}).write_text('closed')\n",
        encoding="utf-8",
    )
    (folder / "parameter.yaml").write_text("prompt_ls: [hello]", encoding="utf-8")
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text(
        yaml.safe_dump(
            {
                "servers": {"generation": str(folder)},
                "pipeline": ["generation.generate"],
            }
        ),
        encoding="utf-8",
    )

    class FailingStream:
        def __init__(self, **kwargs):
            pass

        async def generate_stream(self, **kwargs):
            yield "partial output"
            raise RuntimeError("upstream stream disconnected")

    monkeypatch.setitem(
        sys.modules,
        "servers.generation.src.local_generation",
        SimpleNamespace(LocalGenerationService=FailingStream),
    )
    monkeypatch.chdir(tmp_path)

    async def exercise() -> None:
        await client.build(str(pipeline), no_mcp=True)
        with pytest.raises(RuntimeError, match="upstream stream disconnected"):
            await client.run(str(pipeline), no_mcp=True, is_demo=True)
        assert cleanup.read_text() == "closed"

    asyncio.run(exercise())


def test_failed_startup_passes_exception_to_entered_lifespans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []

    @asynccontextmanager
    async def lifespan(app: LocalServer):
        try:
            yield
        except RuntimeError as exc:
            observed.append(str(exc))
            raise

    first = LocalServer("first", lifespan=lifespan)
    adapter = LocalExecutionClient({"mcpServers": {}}, {"first": {}, "second": {}})

    def load(name: str, path: str) -> LocalServer:
        if name == "second":
            raise RuntimeError("second startup failed")
        return first

    adapter.server_cfg = {"first": {"path": "first"}, "second": {"path": "second"}}
    monkeypatch.setattr(adapter, "_load", load)

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="second startup failed"):
            async with adapter:
                pytest.fail("Startup should not complete")
        assert observed == ["second startup failed"]
        assert not adapter.local

    asyncio.run(exercise())


@pytest.mark.parametrize("fail", [False, True])
def test_pipeline_restores_callers_loop_state(
    monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    async def inner(*args, **kwargs):
        assert client._loop_terminal_var.get() == []
        client._loop_terminal_var.get().append(False)
        if fail:
            raise ValueError("pipeline failed")
        return "done"

    monkeypatch.setattr(client, "_execute_pipeline", inner)

    async def exercise() -> None:
        outer_state = [True]
        token = client._loop_terminal_var.set(outer_state)
        try:
            if fail:
                with pytest.raises(ValueError, match="pipeline failed"):
                    await client.execute_pipeline(None, {})
            else:
                assert await client.execute_pipeline(None, {}) == "done"
            assert client._loop_terminal_var.get() is outer_state
            assert outer_state == [True]
        finally:
            client._loop_terminal_var.reset(token)

    asyncio.run(exercise())
