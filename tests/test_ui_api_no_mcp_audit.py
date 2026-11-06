"""Exercise UI and Python API paths with actual local and stdio MCP services."""

import asyncio
import importlib.util
import json
import os
import shutil
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from ui.backend import pipeline_manager as pm
from ultrarag import api
from ultrarag.client import build

ROOT = Path(__file__).resolve().parents[1]
MODES = [
    True,
    pytest.param(
        False,
        marks=pytest.mark.skipif(
            importlib.util.find_spec("fastmcp") is None,
            reason="MCP extra not installed",
        ),
    ),
]


@pytest.fixture
def chat_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(ROOT / "src"), str(ROOT))))
    folder = tmp_path / "benchmark"
    (folder / "src").mkdir(parents=True)
    source = """import asyncio
import json
from pathlib import Path
from ultrarag.server import UltraRAG_MCP_Server
app = UltraRAG_MCP_Server("benchmark")
calls = 0
@app.tool(output="benchmark->q_ls")
async def get_data(benchmark: dict) -> dict:
    await asyncio.sleep(0.02)
    rows = [json.loads(line) for line in Path(benchmark["path"]).read_text(encoding="utf-8").splitlines()]
    return {"q_ls": [row["question"] for row in rows]}
@app.tool(output="q_ls->ans_ls")
async def reply(q_ls: list[str]) -> dict:
    global calls
    calls += 1
    await asyncio.sleep(0.02)
    if q_ls == ["wait"]:
        Path(__WAIT__).write_text("started")
        await asyncio.sleep(30)
    if q_ls == ["gate"]:
        Path(__WAIT__).write_text("started")
        while not Path(__WAIT__).with_suffix(".release").exists():
            await asyncio.sleep(0.01)
    if q_ls == ["fail"]:
        raise ValueError("intentional failure")
    return {"ans_ls": [f"{question}|{calls}" for question in q_ls]}
@app.tool(output="->None")
def vllm_shutdown() -> None:
    with Path(__CLEANUP__).open("a", encoding="utf-8") as stream:
        stream.write("closed\\n")
if __name__ == "__main__":
    app.run(transport="stdio")
""".replace("__CLEANUP__", repr(str(tmp_path / "cleanup.txt"))).replace(
        "__WAIT__", repr(str(tmp_path / "waiting.txt"))
    )
    (folder / "src/benchmark.py").write_text(source, encoding="utf-8")
    data = tmp_path / "default.jsonl"
    data.write_text(json.dumps({"question": "原始问题"}), encoding="utf-8")
    (folder / "parameter.yaml").write_text(
        yaml.safe_dump(
            {"benchmark": {"path": str(data), "label": "中文参数"}}, allow_unicode=True
        ),
        encoding="utf-8",
    )
    pipeline = tmp_path / "chat.yaml"
    pipeline.write_text(
        yaml.safe_dump(
            {
                "servers": {"benchmark": str(folder)},
                "pipeline": ["benchmark.get_data", "benchmark.reply"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pm, "_find_pipeline_file", lambda name: pipeline if name == "chat" else None
    )
    monkeypatch.setattr(
        pm,
        "_resolve_parameter_path",
        lambda *a, **k: tmp_path / "parameter/chat_parameter.yaml",
    )
    monkeypatch.setattr(pm, "CHAT_DATASET_DIR", tmp_path / "requests")
    monkeypatch.setattr(pm, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(pm, "SESSION_MANAGER", pm.SessionManager(None))
    monkeypatch.setattr(pm, "BACKGROUND_SESSION_MANAGER", pm.SessionManager(None))
    monkeypatch.setattr(pm, "BACKGROUND_TASK_MANAGER", pm.BackgroundTaskManager())
    monkeypatch.setattr(pm, "_find_memory_answer", lambda *a: (None, None))
    return pipeline


def _events(stream) -> list[dict]:
    return [json.loads(item.removeprefix("data: ").strip()) for item in stream]


def test_ui_request_respects_benchmark_alias(
    chat_pipeline: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = yaml.safe_load(chat_pipeline.read_text(encoding="utf-8"))
    config["servers"]["questions"] = config["servers"].pop("benchmark")
    config["pipeline"] = [
        step.replace("benchmark.", "questions.") for step in config["pipeline"]
    ]
    chat_pipeline.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(pm, "NO_MCP_MODE", True)
    pm.build("chat")
    pm.start_demo_session("chat", "alias")
    try:
        events = _events(pm.chat_demo_stream("chat", "aliased question", "alias"))
        answer = next(
            event["data"]["answer"] for event in events if event["type"] == "final"
        )
        assert answer == "aliased question|1"
    finally:
        pm.SESSION_MANAGER.remove("alias")


@pytest.mark.parametrize("no_mcp", MODES)
def test_session_stop_waits_for_pipeline_cleanup(
    chat_pipeline: Path, monkeypatch: pytest.MonkeyPatch, no_mcp: bool
) -> None:
    monkeypatch.setattr(pm, "NO_MCP_MODE", no_mcp)
    pm.configure_execution_mode(no_mcp)
    pm.build("chat")
    session = pm.DemoSession("cancel")
    session.start("chat")
    request, _ = pm._prepare_chat_context("chat", "wait")

    async def callback(event) -> None:
        pass

    with ThreadPoolExecutor(1) as executor:
        task = executor.submit(session.run_chat, callback, request)
        try:
            deadline = time.monotonic() + 15
            while not (chat_pipeline.parent / "waiting.txt").exists():
                assert time.monotonic() < deadline
                time.sleep(0.02)
            session.stop()
            with pytest.raises(CancelledError):
                task.result(timeout=5)
        finally:
            session.stop()
    assert session._loop.is_closed()
    assert (chat_pipeline.parent / "cleanup.txt").read_text().splitlines() == ["closed"]


@pytest.mark.parametrize("no_mcp", MODES)
def test_concurrent_ui_chats_background_and_failure_keep_inputs_isolated(
    chat_pipeline: Path, monkeypatch: pytest.MonkeyPatch, no_mcp: bool
) -> None:
    monkeypatch.setattr(pm, "NO_MCP_MODE", no_mcp)
    pm.configure_execution_mode(no_mcp)
    assert pm.build("chat") == {"status": "ok"}
    params = chat_pipeline.parent / "parameter/chat_parameter.yaml"
    original = params.read_bytes()
    sessions = [pm.SESSION_MANAGER.get_or_create(name) for name in ("first", "second")]
    for session in sessions:
        session.start("chat")

    prepare = pm._prepare_chat_context
    barrier = threading.Barrier(2)

    def synchronized_prepare(*args, **kwargs):
        result = prepare(*args, **kwargs)
        barrier.wait(timeout=15)
        return result

    monkeypatch.setattr(pm, "_prepare_chat_context", synchronized_prepare)
    try:
        with ThreadPoolExecutor(2) as pool:
            futures = [
                pool.submit(lambda q=q, s=s: _events(pm.chat_demo_stream("chat", q, s)))
                for q, s in (("甲", "first"), ("乙", "second"))
            ]
            results = [future.result(timeout=40) for future in futures]
        assert [
            next(
                event["data"]["answer"] for event in result if event["type"] == "final"
            )
            for result in results
        ] == ["甲|1", "乙|1"]
        assert sessions[0].get_conversation_history()[0]["content"] == "甲"
        assert sessions[1].get_conversation_history()[0]["content"] == "乙"
        assert params.read_bytes() == original

        tasks = [
            pm.run_background_chat("chat", question, "unused", user_id="test-user")
            for question in ("丙", "丁")
        ]
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            records = [
                pm.get_background_task(task, user_id="test-user") for task in tasks
            ]
            if all(record["status"] != "running" for record in records):
                break
            time.sleep(0.05)
        assert [record["status"] for record in records] == ["completed", "completed"], (
            records
        )
        assert [record["result"] for record in records] == ["丙|1", "丁|1"]
        assert params.read_bytes() == original

        monkeypatch.setattr(pm, "_prepare_chat_context", prepare)
        failure = _events(pm.chat_demo_stream("chat", "fail", "first"))
        assert any(event["type"] == "error" for event in failure)
        assert params.read_bytes() == original
    finally:
        for name in ("first", "second"):
            pm.SESSION_MANAGER.remove(name)
        for session_id in list(pm.BACKGROUND_SESSION_MANAGER._sessions):
            pm.BACKGROUND_SESSION_MANAGER.remove(session_id)
    assert all(session._loop.is_closed() for session in sessions)
    assert len((chat_pipeline.parent / "cleanup.txt").read_text().splitlines()) == 5


@pytest.mark.parametrize("no_mcp", MODES)
def test_python_api_reinitialize_cleanup_and_pipeline_after_asyncio_run(
    chat_pipeline: Path, no_mcp: bool
) -> None:
    asyncio.run(build(str(chat_pipeline), no_mcp=no_mcp))

    async def exercise() -> None:
        api.initialize(["benchmark"], str(chat_pipeline.parent), no_mcp=no_mcp)
        first = api._client
        cached_reply = api.ToolCall.benchmark.reply
        results = await asyncio.gather(
            api.ToolCall.benchmark.reply(["甲"]), api.ToolCall.benchmark.reply(["乙"])
        )
        assert all(result["ans_ls"] for result in results)
        api.initialize(["benchmark"], str(chat_pipeline.parent), no_mcp=no_mcp)
        assert await cached_reply(["新会话"]) == {"ans_ls": ["新会话|1"]}
        if no_mcp:
            assert not first.local
        await api.shutdown()

    asyncio.run(exercise())
    cleanup = chat_pipeline.parent / "cleanup.txt"
    assert len(cleanup.read_text().splitlines()) == 2
    # asyncio.run removes the current loop; the synchronous API must still work.
    result = api.PipelineCall(
        str(chat_pipeline),
        str(chat_pipeline.parent / "parameter/chat_parameter.yaml"),
        no_mcp=no_mcp,
    )
    assert result["final_result"] == {"ans_ls": ["原始问题|1"]}
    assert len(cleanup.read_text().splitlines()) == 3


def test_api_rejects_reinitialize_and_waits_for_active_calls_on_shutdown(
    chat_pipeline: Path,
) -> None:
    asyncio.run(build(str(chat_pipeline), no_mcp=True))

    async def exercise() -> None:
        api.initialize(["benchmark"], str(chat_pipeline.parent), no_mcp=True)
        call = api.ToolCall.benchmark.reply(["gate"])
        # A pending first call owns initialization even before __aenter__ runs.
        with pytest.raises(RuntimeError, match="calls or shutdown are active"):
            api.initialize(["benchmark"], str(chat_pipeline.parent), no_mcp=True)
        async with asyncio.timeout(10):
            while not (chat_pipeline.parent / "waiting.txt").exists():
                await asyncio.sleep(0.01)
        with pytest.raises(RuntimeError, match="calls or shutdown are active"):
            api.initialize(["benchmark"], str(chat_pipeline.parent), no_mcp=True)
        closing = api.shutdown()
        try:
            with pytest.raises(RuntimeError, match="shutting down"):
                api.ToolCall.benchmark.reply(["new"])
            await asyncio.sleep(0.03)
            assert not closing.done()
            assert not (chat_pipeline.parent / "cleanup.txt").exists()
        finally:
            (chat_pipeline.parent / "waiting.release").touch()
            assert await call == {"ans_ls": ["gate|1"]}
            await closing
        assert (chat_pipeline.parent / "cleanup.txt").read_text().splitlines() == [
            "closed"
        ]

    asyncio.run(exercise())


@pytest.mark.parametrize("no_mcp", MODES)
def test_ui_kb_text_ingest_uses_selected_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_mcp: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    monkeypatch.setattr(pm, "NO_MCP_MODE", no_mcp)
    # Copy the actual built-in server to keep generated configs out of the repo.
    shutil.copytree(
        ROOT / "servers/corpus",
        tmp_path / "corpus",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    pipeline = tmp_path / "build_text_corpus.yaml"
    pipeline.write_text(
        yaml.safe_dump(
            {
                "servers": {"corpus": str(tmp_path / "corpus")},
                "pipeline": ["corpus.build_text_corpus"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(pm, "_find_pipeline_file", lambda name: pipeline)
    monkeypatch.setattr(
        pm,
        "_resolve_parameter_path",
        lambda *a, **k: tmp_path / "parameter/build_text_corpus_parameter.yaml",
    )
    assert pm.build("build_text_corpus") == {"status": "ok"}
    source = tmp_path / "sample.txt"
    source.write_text("A real corpus document for UI ingestion.", encoding="utf-8")
    output = tmp_path / "parsed"
    result = pm.run_kb_pipeline_tool("build_text_corpus", str(source), str(output))
    assert result["status"] == "success"
    records = [
        json.loads(line)
        for line in (output / "sample.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any("A real corpus document" in row["contents"] for row in records)


@pytest.mark.parametrize("no_mcp", MODES)
def test_ui_multiturn_stream_keeps_each_conversation_separate(
    chat_pipeline: Path, monkeypatch: pytest.MonkeyPatch, no_mcp: bool
) -> None:
    """Use a controlled HTTP model endpoint, not a mocked generation method."""
    requests: list[dict] = []

    class Endpoint(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            messages = payload["messages"]
            answer = "/".join(
                message["content"] for message in messages if message["role"] == "user"
            )
            chunk = {
                "id": "test-stream",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {"index": 0, "delta": {"content": answer}, "finish_reason": None}
                ],
            }
            body = ("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            pass

    endpoint = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    worker = threading.Thread(target=endpoint.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setattr(pm, "NO_MCP_MODE", no_mcp)
    pm.configure_execution_mode(no_mcp)
    pm.build("chat")
    original_loader = pm.load_parameters

    def load_parameters(name: str) -> dict:
        parameters = original_loader(name)
        parameters["generation"] = {
            "sampling_params": {"temperature": 0},
            "backend_configs": {
                "openai": {
                    "model_name": "test-model",
                    "api_key": "test-only",
                    "base_url": f"http://127.0.0.1:{endpoint.server_port}/v1",
                }
            },
        }
        return parameters

    monkeypatch.setattr(pm, "load_parameters", load_parameters)
    try:
        for session_id in ("alpha", "beta"):
            pm.start_demo_session("chat", session_id)
            events = _events(pm.chat_demo_stream("chat", session_id, session_id))
            assert any(event["type"] == "final" for event in events)
        with ThreadPoolExecutor(2) as pool:
            futures = [
                pool.submit(
                    lambda sid=sid: _events(pm.chat_multiturn_stream(sid, "follow-up"))
                )
                for sid in ("alpha", "beta")
            ]
            events = [future.result(timeout=25) for future in futures]
        answers = [
            next(e["data"]["answer"] for e in stream if e["type"] == "final")
            for stream in events
        ]
        assert answers == ["alpha/follow-up", "beta/follow-up"]
        assert len(requests) == 2
        assert all(len(request["messages"]) == 3 for request in requests)
    finally:
        for session_id in ("alpha", "beta"):
            pm.SESSION_MANAGER.remove(session_id)
        endpoint.shutdown()
        endpoint.server_close()
        worker.join(timeout=2)
