"""Integration checks for local execution and the existing MCP transport."""

from __future__ import annotations

import asyncio
import builtins
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

import pytest
import yaml

from ultrarag import execution
from ultrarag.client import build, create_execution_client, create_mcp_client, run


def _server(root: Path, name: str, source: str) -> Path:
    folder = root / name
    (folder / "src").mkdir(parents=True)
    (folder / "parameter.yaml").write_text("{}\n", encoding="utf-8")
    (folder / "src" / f"{name}.py").write_text(source, encoding="utf-8")
    return folder


def _pipeline(root: Path) -> Path:
    source = _server(
        root,
        "source",
        """from ultrarag.server import UltraRAG_MCP_Server
app = UltraRAG_MCP_Server('source')
@app.tool(output='->q_ls')
def emit():
    return {'q_ls': ['Ada']}
if __name__ == '__main__':
    app.run(transport='stdio')
""",
    )
    prompt = _server(
        root,
        "prompt",
        """from ultrarag.server import UltraRAG_MCP_Server
app = UltraRAG_MCP_Server('prompt')
@app.prompt(output='q_ls->prompt_ls')
def format_prompt(q_ls: list[str]) -> list[str]:
    return [f'Hello {q}' for q in q_ls]
if __name__ == '__main__':
    app.run(transport='stdio')
""",
    )
    echo = _server(
        root,
        "echo",
        """from ultrarag.server import UltraRAG_MCP_Server
app = UltraRAG_MCP_Server('echo')
@app.tool(output='prompt_ls->ans_ls')
def answer(prompt_ls: list) -> dict:
    texts = []
    for message in prompt_ls:
        content = message['content'] if isinstance(message, dict) else message.content
        text = content.get('text') if isinstance(content, dict) else content.text
        texts.append(text)
    return {'ans_ls': texts}
if __name__ == '__main__':
    app.run(transport='stdio')
""",
    )
    path = root / "pipeline.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "servers": {
                    "source": str(source),
                    "prompt": str(prompt),
                    "echo": str(echo),
                },
                "pipeline": [
                    "source.emit",
                    {"loop": {"times": 2, "steps": ["prompt.format_prompt"]}},
                    "echo.answer",
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_local_pipeline_and_session_isolation(tmp_path: Path) -> None:
    path = _pipeline(tmp_path)

    async def exercise() -> None:
        await build(str(path), no_mcp=True)
        result = await run(str(path), no_mcp=True, return_all=True)
        assert result["final_result"] == {"ans_ls": ["Hello Ada"]}

        from ultrarag.client import load_pipeline_context

        context = load_pipeline_context(str(path), no_mcp=True)
        first = create_execution_client(context["mcp_cfg"], context["server_cfg"], True)
        second = create_execution_client(
            context["mcp_cfg"], context["server_cfg"], True
        )
        async with first, second:
            assert first.local["source"] is not second.local["source"]
            assert (
                first.local["source"].tools["emit"].fn
                is not second.local["source"].tools["emit"].fn
            )
            assert (await first.call_tool("source_emit", {})).data == {"q_ls": ["Ada"]}
            assert (await second.call_tool("source_emit", {})).data == {"q_ls": ["Ada"]}

    asyncio.run(exercise())


def test_local_mode_never_imports_mcp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _pipeline(tmp_path)
    original_import = builtins.__import__

    def reject_mcp(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"fastmcp", "mcp"}:
            raise AssertionError(f"Unexpected MCP import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_mcp)

    async def exercise() -> None:
        await build(str(path), no_mcp=True)
        assert await run(str(path), no_mcp=True) == {"ans_ls": ["Hello Ada"]}

    asyncio.run(exercise())


def test_ui_sessions_use_separate_local_registries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ui.backend import pipeline_manager as pm

    path = _pipeline(tmp_path)
    parameter_path = tmp_path / "parameter" / "pipeline_parameter.yaml"
    monkeypatch.setattr(pm, "_find_pipeline_file", lambda name: path)
    monkeypatch.setattr(
        pm, "_resolve_parameter_path", lambda *args, **kwargs: parameter_path
    )
    pm.configure_execution_mode(True)
    assert pm.build("pipeline") == {"status": "ok"}
    first = pm.DemoSession("first")
    second = pm.DemoSession("second")
    try:
        first.start("pipeline")
        second.start("pipeline")
        assert first._client.local["source"] is not second._client.local["source"]
    finally:
        first.stop()
        second.stop()


def test_ui_starts_with_no_mcp_flag(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root)))
    env["ULTRARAG_UI_STORAGE_ROOT"] = str(tmp_path / "storage")
    command = [
        sys.executable,
        "-m",
        "ultrarag.client",
        "show",
        "ui",
        "--no-mcp",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    process = subprocess.Popen(
        command,
        cwd=root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        for _ in range(60):
            if process.poll() is not None:
                pytest.fail(f"UI exited during startup with code {process.returncode}")
            try:
                with urlopen(
                    f"http://127.0.0.1:{port}/api/pipelines", timeout=1
                ) as response:
                    assert response.status == 200
                    break
            except OSError:
                time.sleep(0.25)
        else:
            pytest.fail("UI route did not become ready")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_local_branch_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _pipeline(tmp_path)
    router = _server(
        tmp_path,
        "router",
        """from ultrarag.server import UltraRAG_MCP_Server
app = UltraRAG_MCP_Server('router')
@app.tool(output='q_ls->q_ls')
def route(q_ls: list[str]) -> dict:
    return {'q_ls': [{'data': q, 'state': 'selected'} for q in q_ls]}
if __name__ == '__main__':
    app.run(transport='stdio')
""",
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["servers"]["router"] = str(router)
    config["pipeline"] = [
        "source.emit",
        {
            "branch": {
                "router": ["router.route"],
                "branches": {"selected": ["prompt.format_prompt"], "other": []},
            }
        },
        "echo.answer",
    ]
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    monkeypatch.setenv(
        "PATH",
        str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
    )
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))

    async def exercise() -> None:
        await build(str(path), no_mcp=True)
        result = await run(str(path), no_mcp=True, return_all=True)
        assert result["final_result"] == {"ans_ls": ["Hello Ada"]}
        if importlib.util.find_spec("fastmcp") is not None:
            await build(str(path))
            mcp = await run(str(path), return_all=True)
            assert mcp["final_result"] == result["final_result"]

    asyncio.run(exercise())


def test_mixed_client_preserves_remote_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pipeline(tmp_path)
    source_path = str(tmp_path / "source" / "src" / "source.py")

    class FakeRemote:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def call_tool(self, name, arguments):
            self.calls.append(name)
            return {"name": name, "arguments": arguments}

        async def get_prompt(self, name, arguments):
            self.calls.append(name)
            return {"name": name, "arguments": arguments}

        async def list_tools(self):
            from types import SimpleNamespace

            return [SimpleNamespace(name="answer")]

    remote = FakeRemote()
    monkeypatch.setattr(execution, "create_mcp_client", lambda cfg: remote)
    server_cfg = {
        "source": {"path": source_path},
        "remote": {"path": "https://example.test/mcp"},
    }
    mcp_cfg = {"mcpServers": {"remote": {"command": "npx", "args": []}}}

    async def exercise() -> None:
        async with execution.LocalExecutionClient(mcp_cfg, server_cfg) as client:
            assert (await client.call_tool("source_emit", {})).data == {"q_ls": ["Ada"]}
            await client.call_tool("remote_answer", {"prompt": "Hi"})
            assert remote.calls == ["answer"]
            assert "remote_answer" in [tool.name for tool in await client.list_tools()]

    asyncio.run(exercise())


def test_mixed_pipeline_build_and_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ultrarag import client as client_module

    path = _pipeline(tmp_path)
    remote_folder = _server(tmp_path, "remote", "")
    remote_url = "https://example.test/mcp"
    (remote_folder / "parameter.yaml").write_text(
        yaml.safe_dump({"path": remote_url}), encoding="utf-8"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["servers"] = {
        "source": config["servers"]["source"],
        "remote": str(remote_folder),
    }
    config["pipeline"] = ["source.emit", "remote.answer"]
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    class FakeRemote:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def call_tool(self, name, arguments):
            if name == "build":
                parameter_file = Path(arguments["parameter_file"])
                server_data = {
                    "path": remote_url,
                    "parameter": str(parameter_file),
                    "tools": {
                        "answer": {
                            "input": {"q_ls": "q_ls"},
                            "output": ["ans_ls"],
                        }
                    },
                    "prompts": {},
                }
                (parameter_file.parent / "server.yaml").write_text(
                    yaml.safe_dump(server_data), encoding="utf-8"
                )
                value = None
            else:
                assert name == "answer"
                value = {
                    "ans_ls": [f"Remote {question}" for question in arguments["q_ls"]]
                }
            return SimpleNamespace(
                data=value,
                content=[SimpleNamespace(text=json.dumps(value))],
            )

        async def list_tools(self):
            return [SimpleNamespace(name="build"), SimpleNamespace(name="answer")]

    monkeypatch.setattr(execution, "create_mcp_client", lambda cfg: FakeRemote())
    monkeypatch.setattr(client_module, "check_node_version", lambda minimum: None)

    async def exercise() -> None:
        await build(str(path), no_mcp=True)
        result = await run(str(path), no_mcp=True)
        assert result == {"ans_ls": ["Remote Ada"]}

    asyncio.run(exercise())


def test_builtin_servers_register_for_local_execution() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = {
        name: root / "servers" / name / "src" / f"{name}.py"
        for name in ("generation", "retriever", "prompt")
    }
    server_cfg = {name: {"path": str(path)} for name, path in paths.items()}

    async def exercise() -> None:
        async with execution.LocalExecutionClient(
            {"mcpServers": {}}, server_cfg
        ) as client:
            assert "generate" in client.local["generation"].tools
            assert "retriever_search" in client.local["retriever"].tools
            assert "qa_boxed" in client.local["prompt"].prompts
            result = await client.get_prompt(
                "prompt_qa_boxed",
                {"q_ls": ["Ada"], "template": str(root / "prompt" / "qa_boxed.jinja")},
            )
            assert "Ada" in result.messages[0].content.text

    asyncio.run(exercise())


def test_main_guarded_class_registration_is_supported(tmp_path: Path) -> None:
    folder = _server(
        tmp_path,
        "legacy",
        """from ultrarag.server import UltraRAG_MCP_Server
app = UltraRAG_MCP_Server('legacy')
class Legacy:
    def __init__(self, server):
        server.tool(self.greet, output='name->msg')
    def greet(self, name: str) -> dict:
        return {'msg': f'Hi {name}'}
if __name__ == '__main__':
    Legacy(app)
    app.run(transport='stdio')
""",
    )
    cfg = {"legacy": {"path": str(folder / "src" / "legacy.py")}}

    async def exercise() -> None:
        async with execution.LocalExecutionClient({"mcpServers": {}}, cfg) as client:
            assert (await client.call_tool("greet", {"name": "Ada"})).data == {
                "msg": "Hi Ada"
            }

    asyncio.run(exercise())


def test_plain_fastmcp_server_has_migration_error(tmp_path: Path) -> None:
    folder = _server(
        tmp_path,
        "plain",
        """from fastmcp import FastMCP
app = FastMCP('plain')
""",
    )
    cfg = {"plain": {"path": str(folder / "src" / "plain.py")}}

    async def exercise() -> None:
        async with execution.LocalExecutionClient({"mcpServers": {}}, cfg):
            pass

    with pytest.raises((RuntimeError, ValueError), match="UltraRAG_MCP_Server"):
        asyncio.run(exercise())


@pytest.mark.skipif(
    importlib.util.find_spec("fastmcp") is None, reason="MCP extra not installed"
)
def test_builtin_prompt_still_works_over_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv(
        "PATH",
        str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
    )
    monkeypatch.setenv("PYTHONPATH", str(root / "src"))
    cfg = {
        "mcpServers": {
            "prompt": {
                "command": "python",
                "args": [str(root / "servers" / "prompt" / "src" / "prompt.py")],
                "env": os.environ.copy(),
            }
        }
    }

    async def exercise() -> None:
        async with create_mcp_client(cfg) as client:
            result = await client.get_prompt(
                "qa_boxed",
                {"q_ls": ["Ada"], "template": str(root / "prompt" / "qa_boxed.jinja")},
            )
            assert "Ada" in result.messages[0].content.text

    asyncio.run(exercise())


@pytest.mark.skipif(
    importlib.util.find_spec("fastmcp") is None, reason="MCP extra not installed"
)
def test_mcp_and_local_pipeline_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _pipeline(tmp_path)
    monkeypatch.setenv(
        "PATH",
        str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
    )
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))

    async def exercise() -> None:
        await build(str(path), no_mcp=True)
        server_config_path = tmp_path / "server" / "pipeline_server.yaml"
        parameter_path = tmp_path / "parameter" / "pipeline_parameter.yaml"
        local_server_config = yaml.safe_load(
            server_config_path.read_text(encoding="utf-8")
        )
        local_parameters = yaml.safe_load(parameter_path.read_text(encoding="utf-8"))
        local = await run(str(path), no_mcp=True, return_all=True)
        await build(str(path))
        mcp_server_config = yaml.safe_load(
            server_config_path.read_text(encoding="utf-8")
        )
        mcp_parameters = yaml.safe_load(parameter_path.read_text(encoding="utf-8"))
        for alias in local_server_config:
            assert local_server_config[alias].get("tools") == mcp_server_config[
                alias
            ].get("tools")
            assert local_server_config[alias].get("prompts") == mcp_server_config[
                alias
            ].get("prompts")
        assert local_parameters == mcp_parameters
        mcp = await run(str(path), return_all=True)
        assert local["final_result"] == mcp["final_result"]

    asyncio.run(exercise())
