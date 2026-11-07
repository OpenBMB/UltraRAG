"""Exercise actual model/tool/result iteration, proposal isolation and cancellation."""

import asyncio
import importlib
import json

import httpx
import pytest
import yaml

from ui.backend import builder_agent as agent
from ui.backend import pipeline_manager as pm
from ui.backend.agent_workspace import ToolWorkspace, parse_document

VALID = "servers:\n  hello: servers/sayhello\npipeline:\n- hello.greet\n"
SETTINGS = {
    "provider": "custom",
    "baseUrl": "https://model.example/v1",
    "apiKey": "test-secret",
    "model": "test",
}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    (tmp_path / "servers/sayhello").mkdir(parents=True)
    (tmp_path / "prompt").mkdir()
    (tmp_path / "servers/sayhello/server.yaml").write_text(
        yaml.safe_dump(
            {"tools": {"greet": {"input": {}, "output": {"answer": "answer"}}}}
        )
    )
    (tmp_path / "servers/sayhello/parameter.yaml").write_text(
        "api_key: secret\ncount: 1\n"
    )
    monkeypatch.setattr(pm, "SERVERS_DIR", tmp_path / "servers")
    monkeypatch.setattr(pm, "PROJECT_ROOT", tmp_path)
    return ToolWorkspace({"selectedPipeline": "draft", "pipelineYaml": VALID})


def test_tools_read_actual_contracts_and_redact_secrets(workspace):
    info = workspace.call("inspect_server", {"name": "sayhello"})
    assert "greet" in info["metadata"]["tools"]
    assert info["defaults"]["api_key"] == "[REDACTED]"
    assert workspace.call("validate_pipeline", {"yaml": VALID})["valid"]
    assert not workspace.call(
        "validate_pipeline", {"yaml": VALID.replace("greet", "invented")}
    )["valid"]


def test_proposals_are_validated_and_never_save_files(workspace, tmp_path):
    before = sorted(tmp_path.rglob("*"))
    assert workspace.call("propose_pipeline", {"yaml": VALID})["staged"]
    assert workspace.actions[0]["targetPipeline"] == "draft"
    assert sorted(tmp_path.rglob("*")) == before
    assert not workspace.call("propose_pipeline", {"yaml": "servers: {}"})["staged"]
    assert workspace.actions == []


@pytest.mark.parametrize(
    "path,value",
    [
        ("hello.api_key", "new"),
        ("hello.options", {"apiKey": "new"}),
        ("hello.__proto__.x", 1),
        ("unknown.count", 1),
    ],
)
def test_parameter_proposals_reject_credentials_and_invalid_paths(
    workspace, path, value
):
    with pytest.raises(ValueError):
        workspace.call("propose_parameter", {"path": path, "value": value})


def test_prompt_tools_block_traversal_and_symlinks(workspace, tmp_path):
    outside = tmp_path / "outside.jinja"
    outside.write_text("secret")
    (tmp_path / "prompt/link.jinja").symlink_to(outside)
    for filename in ["../outside.jinja", "link.jinja", str(outside)]:
        with pytest.raises(ValueError):
            workspace.call("read_prompt", {"filename": filename})


def test_recursive_yaml_is_rejected(workspace):
    with pytest.raises(ValueError, match="Recursive"):
        parse_document("a: &a [*a]")


def stream_message(*, text="", calls=None):
    delta = {"content": text}
    if calls:
        delta["tool_calls"] = [
            {
                "index": i,
                "id": f"call{i}",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
            for i, (name, arguments) in enumerate(calls)
        ]
    chunks = [
        {"choices": [{"delta": delta, "finish_reason": None}]},
        {
            "choices": [
                {"delta": {}, "finish_reason": "tool_calls" if calls else "stop"}
            ]
        },
    ]
    return httpx.Response(
        200,
        text="".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
        + "data: [DONE]\n\n",
    )


def test_real_tool_result_loop_corrects_invalid_proposal(workspace, monkeypatch):
    monkeypatch.setattr(agent, "ToolWorkspace", lambda _: workspace)
    requests, events = [], []

    async def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return stream_message(
                calls=[
                    ("propose_pipeline", {"yaml": VALID.replace("greet", "invented")})
                ]
            )
        if len(requests) == 2:
            assert "unknown tool" in body["messages"][-1]["content"]
            return stream_message(
                calls=[
                    ("inspect_server", {"name": "sayhello"}),
                    ("propose_pipeline", {"yaml": VALID}),
                ]
            )
        assert body["messages"][-1]["role"] == "tool"
        assert json.loads(body["messages"][-1]["content"])["staged"]
        return stream_message(text="已校验，请应用并 Build。")

    asyncio.run(
        agent.run_agent(
            SETTINGS,
            [{"role": "user", "content": "build"}],
            {},
            events.append,
            transport=httpx.MockTransport(respond),
        )
    )
    assert len(requests) == 3
    assert events[-1]["type"] == "final"
    assert events[-1]["actions"][0]["content"] == VALID
    assert any(event.get("ok") is False for event in events)


def test_cancellation_closes_model_request_and_prevents_tools(workspace, monkeypatch):
    monkeypatch.setattr(agent, "ToolWorkspace", lambda _: workspace)
    events = []

    async def check():
        entered = asyncio.Event()
        closed = asyncio.Event()

        async def respond(request):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()

        task = asyncio.create_task(
            agent.run_agent(
                SETTINGS, [], {}, events.append, transport=httpx.MockTransport(respond)
            )
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()

    asyncio.run(check())
    assert not any(event["type"].startswith("tool") for event in events)


def test_stop_registry_checks_owner_and_disconnect_releases_run(monkeypatch):
    async def idle(*args):
        await asyncio.Event().wait()

    monkeypatch.setattr(agent, "run_agent", idle)
    run = agent.create_run(
        "admin",
        {"settings": SETTINGS, "messages": [{"role": "user", "content": "hello"}]},
    )
    stream = run.stream()
    assert run.id in next(stream)
    assert not agent.stop_run("someone-else", run.id)
    assert agent.stop_run("admin", run.id)
    stream.close()
    assert run.id not in agent.RUNS


def test_agent_api_requires_admin_and_validates_requests(tmp_path, monkeypatch):
    module = importlib.import_module("ui.backend.app")
    monkeypatch.setattr(module, "AUTH_DB_PATH", tmp_path / "users.sqlite3")
    app = module.create_app(admin_mode=True, no_mcp=True)
    with app.test_client() as client:
        assert client.post("/api/ai/agent", json={}).status_code == 403
        with client.session_transaction() as session:
            session["auth_user_id"] = "admin"
        assert client.post("/api/ai/agent", json={}).status_code == 400
        assert client.post("/api/ai/agent/missing/stop").get_json() == {
            "stopped": False
        }


def test_parameter_proposals_require_actual_parameter_paths(workspace):
    assert workspace.call("propose_parameter", {"path": "hello.count", "value": 2})[
        "staged"
    ]
    with pytest.raises(ValueError, match="Unknown parameter"):
        workspace.call("propose_parameter", {"path": "hello.misspelled", "value": 2})
    with pytest.raises(ValueError):
        parse_document("number: .nan")


def test_pipeline_tool_never_echoes_raw_yaml_credentials(workspace, monkeypatch):
    monkeypatch.setattr(
        pm,
        "load_pipeline",
        lambda _: {
            "servers": {"hello": "servers/sayhello"},
            "apiKey": "private-marker",
            "_raw_yaml": "apiKey: private-marker",
        },
    )
    result = workspace.call("read_pipeline", {"name": "saved"})
    assert "private-marker" not in json.dumps(result)
    assert "_raw_yaml" not in result["pipeline"]


def test_parameter_proposals_reject_nested_unsafe_keys(workspace):
    with pytest.raises(ValueError, match="safe strings"):
        workspace.call(
            "propose_parameter",
            {"path": "hello.count", "value": {"__proto__": {"x": 1}}},
        )
