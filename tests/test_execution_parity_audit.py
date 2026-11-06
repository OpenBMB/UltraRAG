"""Differential checks against a real FastMCP transport, plus session isolation."""

import asyncio
import json
import os
import shutil
import socket
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from ultrarag.execution import LocalExecutionClient, create_mcp_client

SOURCE = """from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
from ultrarag.server import UltraRAG_MCP_Server

@asynccontextmanager
async def lifespan(app):
    app.started = True
    yield
    Path(__file__).with_suffix('.closed').write_text('closed')

app = UltraRAG_MCP_Server('audit', lifespan=lifespan)

@app.tool()
def typed(value: int, payload: list[dict]) -> dict:
    payload[0]['changed'] = True
    return {'value': value + 1, 'payload': payload}

@app.tool()
def text() -> str:
    return 'hello'

@app.tool()
def empty() -> None:
    return None

@app.tool()
def sequence() -> list[int]:
    return [1, 2]

@app.tool()
def untyped():
    return 'untyped'

@app.tool()
def empty_mapping():
    return {}

@app.tool(enabled=False)
def disabled_tool() -> dict:
    return {'should_not_run': True}

@app.prompt(enabled=False)
def disabled_prompt() -> str:
    return 'should_not_run'

@app.tool()
def lifecycle() -> dict:
    return {'started': app.started}

@app.tool()
def process_id() -> dict:
    import os
    return {'pid': os.getpid()}

@app.prompt()
def welcome(names: list[str], label: str = 'hello') -> list[str]:
    return [label + ' ' + name for name in names]

@app.prompt()
def assistant_message():
    from fastmcp.prompts import Message
    from mcp.types import TextContent
    return [Message(TextContent(type='text', text='assistant reply', annotations={'audience': ['assistant'], 'priority': 0.5}), role='assistant')]

@app.prompt()
def invalid_prompt():
    return [{'role': 'assistant', 'content': {'type': 'text', 'text': 'bad shape'}}]

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        app.run(transport='http', host='127.0.0.1', port=int(sys.argv[1]))
    else:
        app.run(transport='stdio')
"""


def _source(tmp_path: Path) -> Path:
    path = tmp_path / "audit.py"
    path.write_text(SOURCE, encoding="utf-8")
    return path


def _mcp_config(path: Path) -> dict:
    return {
        "mcpServers": {
            "audit": {
                "command": sys.executable,
                "args": [str(path)],
                "env": {
                    "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
                },
            }
        }
    }


def _snapshot(result) -> dict:
    return {
        "data": getattr(result, "data", None),
        "content": [item.model_dump(exclude_none=True) for item in result.content],
    }


def _process_is_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            error = ctypes.get_last_error()
            if error == 87:  # ERROR_INVALID_PARAMETER: process no longer exists.
                return False
            raise ctypes.WinError(error)
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise ctypes.WinError(ctypes.get_last_error())
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.parametrize("aliases", [("one",), ("one", "two")])
def test_mcp_context_exit_stops_server_processes(
    tmp_path: Path, aliases: tuple[str, ...]
) -> None:
    pytest.importorskip("fastmcp")
    paths = []
    servers = {}
    for alias in aliases:
        folder = tmp_path / alias
        folder.mkdir()
        source = _source(folder)
        paths.append(source)
        servers[alias] = _mcp_config(source)["mcpServers"]["audit"]
    config = {"mcpServers": servers}
    original = json.loads(json.dumps(config))

    async def exercise() -> None:
        client = create_mcp_client(config)
        try:
            async with client:
                # Nested uses must retain the same process until the outer exit.
                async with client:
                    names = [
                        f"{alias}_process_id" if len(aliases) > 1 else "process_id"
                        for alias in aliases
                    ]
                    pids = [
                        (await client.call_tool(name, {})).data["pid"] for name in names
                    ]
                assert all(_process_is_alive(pid) for pid in pids)
                assert [
                    (await client.call_tool(name, {})).data["pid"] for name in names
                ] == pids
            # Do not rely on asyncio.run teardown or a local-mode marker to close MCP.
            async with asyncio.timeout(5):
                while any(_process_is_alive(pid) for pid in pids):
                    await asyncio.sleep(0.05)
            assert all(
                path.with_suffix(".closed").read_text() == "closed" for path in paths
            )
            assert config == original
        finally:
            await client.close()

    asyncio.run(exercise())


def test_values_validation_prompts_and_lifespan_match_mcp(tmp_path: Path) -> None:
    pytest.importorskip("fastmcp")
    from fastmcp.exceptions import ToolError
    from mcp.shared.exceptions import McpError

    path = _source(tmp_path)

    async def exercise() -> None:
        async with (
            LocalExecutionClient({}, {"audit": {"path": str(path)}}) as local,
            create_mcp_client(_mcp_config(path)) as remote,
        ):
            for name, arguments in (
                ("typed", {"value": "4", "payload": [{"original": True}]}),
                ("text", {}),
                ("empty", {}),
                ("sequence", {}),
                ("lifecycle", {}),
                ("untyped", {}),
                ("empty_mapping", {}),
            ):
                original = json.loads(json.dumps(arguments))
                left = await local.call_tool(name, arguments)
                assert arguments == original, "local execution mutated caller input"
                right = await remote.call_tool(name, arguments)
                assert _snapshot(left) == _snapshot(right)
            for arguments in (
                {"names": ["Ada"], "label": "123"},
                {"names": '["Ada"]'},
                {"names": ["Ada"], "ignored": "extra"},
            ):
                left = await local.get_prompt("welcome", arguments)
                right = await remote.get_prompt("welcome", arguments)
                assert [m.model_dump(exclude_none=True) for m in left.messages] == [
                    m.model_dump(exclude_none=True) for m in right.messages
                ]
            left = await local.get_prompt("assistant_message", {})
            right = await remote.get_prompt("assistant_message", {})
            assert left.messages[0].model_dump(exclude_none=True) == right.messages[
                0
            ].model_dump(exclude_none=True)
            assert left.messages[0].role == "assistant"
            with pytest.raises(TypeError):
                await local.get_prompt("invalid_prompt", {})
            with pytest.raises(McpError):
                await remote.get_prompt("invalid_prompt", {})
            for arguments in (
                {"value": "invalid", "payload": [{}]},
                {"payload": [{}]},
                {"value": 1, "payload": [{}], "extra": 2},
            ):
                with pytest.raises(ValidationError):
                    await local.call_tool("typed", arguments)
                with pytest.raises(ToolError):
                    await remote.call_tool("typed", arguments)
            for client in (local, remote):
                assert "disabled_tool" not in {
                    tool.name for tool in await client.list_tools()
                }
                with pytest.raises((ValueError, ToolError)):
                    await client.call_tool("disabled_tool", {})
                with pytest.raises((ValueError, McpError), match="disabled_prompt"):
                    await client.get_prompt("disabled_prompt", {})
        assert path.with_suffix(".closed").read_text() == "closed"

    asyncio.run(exercise())


def test_sibling_and_relative_modules_are_isolated(tmp_path: Path) -> None:
    for alias in ("one", "two"):
        folder = tmp_path / alias
        folder.mkdir()
        (folder / "state.py").write_text("counter = 0\n", encoding="utf-8")
        (folder / "nested").mkdir()
        (folder / "nested/__init__.py").write_text(
            "from .child import bump\n", encoding="utf-8"
        )
        (folder / "nested/child.py").write_text(
            "import importlib\nstate = importlib.import_module('state')\ndef bump():\n    state.counter += 1\n    return state.counter\n",
            encoding="utf-8",
        )
        (folder / "server.py").write_text(
            "from ultrarag.server import UltraRAG_MCP_Server\napp = UltraRAG_MCP_Server('state')\n@app.tool()\ndef tick() -> dict:\n    from nested import bump\n    return {'value': bump()}\n",
            encoding="utf-8",
        )
    cfg = {
        alias: {"path": str(tmp_path / alias / "server.py")} for alias in ("one", "two")
    }

    async def exercise() -> None:
        async with (
            LocalExecutionClient({}, cfg) as first,
            LocalExecutionClient({}, cfg) as second,
        ):
            for client in (first, second):
                for alias in cfg:
                    assert (await client.call_tool(f"{alias}_tick", {})).data == {
                        "value": 1
                    }
                    assert (await client.call_tool(f"{alias}_tick", {})).data == {
                        "value": 2
                    }
        assert not any(name.startswith("_ultrarag_modules_") for name in sys.modules)

    asyncio.run(exercise())


def test_sync_tool_does_not_block_other_sessions(tmp_path: Path) -> None:
    path = tmp_path / "waiter.py"
    path.write_text(
        "import time\nfrom ultrarag.server import UltraRAG_MCP_Server\napp = UltraRAG_MCP_Server('waiter')\n@app.tool()\ndef wait() -> None:\n    time.sleep(0.25)\n",
        encoding="utf-8",
    )

    async def exercise() -> None:
        async with LocalExecutionClient({}, {"waiter": {"path": str(path)}}) as client:
            pending = asyncio.create_task(client.call_tool("wait", {}))
            await asyncio.sleep(0.05)
            assert not pending.done()
            await pending

    asyncio.run(exercise())


def test_mcp_only_extensions_are_not_silently_ignored() -> None:
    from ultrarag.local_server import LocalServer

    with pytest.raises(ValueError, match="--no-mcp.*middleware"):
        LocalServer(middleware=[object()])
    LocalServer(middleware=[], include_tags=set(), auth=None)
    for key, value in (
        ("auth", object()),
        ("timeout", 2),
        ("exclude_args", ["secret"]),
        ("output_schema", None),
        ("run_in_thread", False),
    ):
        with pytest.raises(ValueError, match=key):
            LocalServer().tool(**{key: value})
    with pytest.raises(ValueError, match="strict_input_validation"):
        LocalServer(strict_input_validation=False)


def test_cancel_waits_for_sync_tool_before_releasing_its_resources(
    tmp_path: Path,
) -> None:
    import threading

    from ultrarag.registration import Registration

    started = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    def blocking() -> None:
        started.set()
        assert release.wait(5)
        completed.set()

    async def exercise() -> None:
        path = _source(tmp_path)
        async with LocalExecutionClient({}, {"audit": {"path": str(path)}}) as client:
            client.local["audit"].tools["blocking"] = Registration(
                "blocking", blocking, None
            )
            task = asyncio.create_task(client.call_tool("blocking", {}))
            assert await asyncio.to_thread(started.wait, 3)
            task.cancel()
            await asyncio.sleep(0.02)
            task.cancel()
            await asyncio.sleep(0.02)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert completed.is_set()

    try:
        asyncio.run(exercise())
    finally:
        release.set()


@pytest.mark.parametrize(
    "aliases,bridge,trailing_slash",
    [
        (("remote_one",), False, False),
        (("remote", "remote_one"), False, False),
        (("remote_one",), True, False),
        (("remote_one",), False, True),
    ],
)
def test_real_http_mixed_aliases(
    tmp_path: Path, aliases: tuple[str, ...], bridge: bool, trailing_slash: bool
) -> None:
    pytest.importorskip("fastmcp")
    if bridge and not shutil.which("npx"):
        pytest.skip("Node.js/npx is not installed")
    import subprocess
    import time

    path = _source(tmp_path)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    process = subprocess.Popen(
        [sys.executable, str(path), str(port)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            if process.poll() is not None:
                pytest.fail("HTTP MCP server exited")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            pytest.fail("HTTP MCP server did not start")
        url = f"http://127.0.0.1:{port}/mcp"
        if trailing_slash:
            import httpx

            url += "/"
            with httpx.stream(
                "GET", url, follow_redirects=False, timeout=3
            ) as response:
                assert response.status_code in (307, 308)
                assert response.headers["location"] == url.rstrip("/")
        cfg = {
            "local": {"path": str(path)},
            **{alias: {"path": url} for alias in aliases},
        }
        transport = {"mcpServers": {alias: {"url": url} for alias in aliases}}
        if bridge:
            from ultrarag.client import remote_server_launch_config

            transport = {
                "mcpServers": {
                    alias: remote_server_launch_config(url) for alias in aliases
                }
            }

        async def exercise() -> None:
            async with (
                asyncio.timeout(90),
                LocalExecutionClient(transport, cfg) as mixed,
            ):
                names = {tool.name for tool in await mixed.list_tools()}
                for alias in cfg:
                    assert f"{alias}_typed" in names
                    assert (
                        await mixed.call_tool(
                            f"{alias}_typed", {"value": 2, "payload": [{}]}
                        )
                    ).data["value"] == 3
                    prompt = await mixed.get_prompt(
                        f"{alias}_welcome", {"names": ["Ada"]}
                    )
                    assert prompt.messages[0].content.text == "hello Ada"

        asyncio.run(exercise())
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
