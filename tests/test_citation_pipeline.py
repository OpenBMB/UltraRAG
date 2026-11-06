"""Exercise the real executor and UI answer extraction with offline MCP doubles."""

import asyncio
import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from ui.backend.pipeline_manager import _extract_result
from ultrarag import client as engine

ROOT = Path(__file__).resolve().parents[1]


def load_server(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / f"servers/{name}/src/{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pipeline_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine, "logger", logging.getLogger("citation-tests"))
    custom = load_server("custom")
    router = load_server("router")
    configs = {}
    for name, module in (("custom", custom), ("router", router)):
        module.app._refresh_registered_meta()
        configs[name] = {
            "path": str(ROOT / f"servers/{name}/src/{name}.py"),
            "tools": {
                key: module.app._build_entry(meta, {})
                for key, meta in module.app.fn_meta.items()
            },
        }
    configs["benchmark"] = {
        "path": "benchmark.py",
        "tools": {"get_data": {"input": {}, "output": ["q_ls", "ret_psg", "page_ls"]}},
    }
    configs["generation"] = {
        "path": str(ROOT / "servers/generation/src/generation.py"),
        "tools": {"generate": {"input": {"ret_psg": "ret_psg"}, "output": ["ans_ls"]}},
    }

    class Generation:
        def __init__(self, **kwargs):
            pass

        async def generate_stream(self, **kwargs):
            yield "Answer [1]"

    generation_module = ModuleType("servers.generation.src.local_generation")
    generation_module.LocalGenerationService = Generation
    monkeypatch.setitem(sys.modules, generation_module.__name__, generation_module)

    class Client:
        failure = None

        async def list_tools(self):
            return []

        async def call_tool(self, name, args):
            if name == "benchmark_get_data":
                payload = {
                    "q_ls": ["question"],
                    "ret_psg": [["Document A"]],
                    "page_ls": ["to be filled"],
                }
            elif name == "generation_generate":
                if self.failure:
                    raise self.failure
                payload = {"ans_ls": ["Answer [1]"]}
            else:
                prefix, tool = name.split("_", 1)
                payload = getattr({"custom": custom, "router": router}[prefix], tool)(
                    **args
                )
            return SimpleNamespace(
                content=[SimpleNamespace(text=json.dumps(payload))], data=payload
            )

    def context(steps):
        path = tmp_path / "pipeline.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "servers": {key: str(tmp_path / key) for key in configs},
                    "pipeline": steps,
                }
            )
        )
        return {
            "config_path": str(path),
            "server_cfg": configs,
            "param_config_path": str(tmp_path / "params.yaml"),
            "pipeline_cfg": steps,
            "cfg_name": "citation-test",
        }

    return custom, router, Client, context


def answer_steps():
    demo = yaml.safe_load((ROOT / "examples/demos/LightResearch.yaml").read_text())[
        "pipeline"
    ]
    steps = [
        "benchmark.get_data",
        "custom.init_citation_registry",
        "custom.assign_citation_ids_stateful",
        "generation.generate",
    ]
    if demo[-1] == "custom.clear_citation_registry":
        steps.append(demo[-1])
    return steps


def test_final_answer_and_citations_survive_pipeline_and_ui(pipeline_env):
    _, _, Client, context = pipeline_env
    events = []

    async def capture(event):
        events.append(event)

    result = asyncio.run(
        engine.execute_pipeline(
            Client(),
            context(answer_steps()),
            is_demo=True,
            return_all=True,
            stream_callback=capture,
        )
    )
    tokens = [event for event in events if event["type"] == "token"]
    assert tokens and all(event["is_final"] for event in tokens)
    assert _extract_result(result) == "Answer [1]"
    assert json.loads(result["final_result"])["ans_ls"] == ["Answer [1]"]
    sources = [event for event in events if event["type"] == "sources"]
    assert sources
    assert sources[-1]["data"][0]["id"] == 1
    assert sources[-1]["data"][0]["content"] == "Document A"


@pytest.mark.parametrize(
    "failure", [RuntimeError("retrieval failed"), asyncio.CancelledError()]
)
def test_failure_and_cancellation_retain_no_server_registry(pipeline_env, failure):
    custom, _, Client, context = pipeline_env
    client = Client()
    client.failure = failure
    with pytest.raises(type(failure)):
        asyncio.run(engine.execute_pipeline(client, context(answer_steps())))
    assert (
        not hasattr(custom, "CitationRegistry")
        or not custom.CitationRegistry._instances
    )


def test_mixed_branches_keep_query_state_across_loop_iterations(pipeline_env):
    _, _, BaseClient, context = pipeline_env
    calls = 0

    class Client(BaseClient):
        async def call_tool(self, name, args):
            nonlocal calls
            if name == "benchmark_get_data":
                payload = {
                    "q_ls": ["completed query", "active query"],
                    "page_ls": ["finished", "to be filled"],
                    "ret_psg": [[], []],
                }
            elif name == "retriever_search":
                assert args["page_ls"] == ["to be filled"]
                calls += 1
                payload = {"ret_psg": [["A", "B"]] if calls == 1 else [["B", "C"]]}
            else:
                return await super().call_tool(name, args)
            return SimpleNamespace(
                content=[SimpleNamespace(text=json.dumps(payload))], data=payload
            )

    steps = [
        "benchmark.get_data",
        "custom.init_citation_registry",
        {
            "loop": {
                "times": 2,
                "steps": [
                    {
                        "branch": {
                            "router": ["router.webnote_check_page_with_citations"],
                            "branches": {
                                "complete": [],
                                "incomplete": [
                                    {
                                        "retriever.search": {
                                            "output": {"ret_psg": "psg_ls"}
                                        }
                                    },
                                    {
                                        "custom.assign_citation_ids_stateful": {
                                            "input": {"ret_psg": "psg_ls"},
                                            "output": {
                                                "ret_psg": "psg_ls",
                                                "citation_state": "citation_state",
                                            },
                                        }
                                    },
                                ],
                            },
                        }
                    }
                ],
            }
        },
    ]
    ctx = context(steps)
    ctx["server_cfg"]["retriever"] = {
        "path": "retriever.py",
        "tools": {"search": {"input": {"page_ls": "page_ls"}, "output": ["ret_psg"]}},
    }
    result = asyncio.run(engine.execute_pipeline(Client(), ctx, return_all=True))
    assert calls == 2
    final = result["final_result"]
    assert final["ret_psg"] == [["[2] B", "[3] C"]]
    snapshot = result["all_results"][-1]["memory"]["memory_citation_state"]
    assert snapshot[0] == {"registry": {}, "counter": 0}
    assert snapshot[1] == {"registry": {"A": 1, "B": 2, "C": 3}, "counter": 3}


def test_lightresearch_keeps_answer_last_and_saves_state_mapping():
    steps = yaml.safe_load((ROOT / "examples/demos/LightResearch.yaml").read_text())[
        "pipeline"
    ]
    assert steps[-1] == "generation.generate"
    loop = next(
        step["loop"] for step in steps if isinstance(step, dict) and "loop" in step
    )
    branch = loop["steps"][0]["branch"]
    assert branch["router"] == ["router.webnote_check_page_with_citations"]
    assignment = next(
        step["custom.assign_citation_ids_stateful"]
        for step in branch["branches"]["incomplete"]
        if isinstance(step, dict) and "custom.assign_citation_ids_stateful" in step
    )
    assert assignment["output"]["citation_state"] == "citation_state"


def test_documented_lightresearch_pipeline_matches_executable_example():
    documentation = (ROOT / "docs/llms.txt").read_text()
    marker = '```yaml examples/LightResearch.yaml icon="/images/yaml.svg"\n'
    documented_yaml = documentation.split(marker, 1)[1].split("```", 1)[0]
    executable_yaml = (ROOT / "examples/demos/LightResearch.yaml").read_text()
    assert yaml.safe_load(documented_yaml) == yaml.safe_load(executable_yaml)


def test_citation_tools_round_trip_over_mcp(pipeline_env):
    from fastmcp import Client

    custom, router, _, _ = pipeline_env

    async def exercise():
        async with Client(custom.app) as client:
            initial = await client.call_tool("init_citation_registry", {"q_ls": ["q"]})
            state = json.loads(initial.content[0].text)["citation_state"]
            first = await client.call_tool(
                "assign_citation_ids_stateful",
                {"ret_psg": [["A"]], "citation_state": state},
            )
            state = json.loads(first.content[0].text)["citation_state"]
        async with Client(router.app) as client:
            routed = await client.call_tool(
                "webnote_check_page_with_citations",
                {"page_ls": ["to be filled"], "citation_state": state},
            )
            payload = json.loads(routed.content[0].text)
            assert payload["citation_state"] == [
                {"state": "incomplete", "data": state[0]}
            ]
        async with Client(custom.app) as client:
            continued = await client.call_tool(
                "assign_citation_ids_stateful",
                {"ret_psg": [["A", "B"]], "citation_state": state},
            )
            assert json.loads(continued.content[0].text)["ret_psg"] == [
                ["[1] A", "[2] B"]
            ]

    asyncio.run(exercise())


def test_ui_sse_final_event_preserves_answer_and_source_ids(
    pipeline_env, tmp_path, monkeypatch
):
    from ui.backend import pipeline_manager as ui

    _, _, Client, context = pipeline_env
    ctx = context(answer_steps())
    parameter_path = tmp_path / "ui-parameters.yaml"
    parameter_path.write_text("{}")
    history = []

    class Session:
        _pipeline_name = "citation-test"

        def run_chat(self, callback, params):
            return asyncio.run(
                engine.execute_pipeline(
                    Client(),
                    ctx,
                    is_demo=True,
                    return_all=True,
                    stream_callback=callback,
                )
            )

        def add_to_history(self, role, content):
            history.append((role, content))

        def mark_first_turn_done(self):
            pass

        def init_multiturn_client(self):
            pass

    monkeypatch.setattr(ui, "SESSION_MANAGER", SimpleNamespace(get=lambda _: Session()))
    monkeypatch.setattr(
        ui, "_prepare_chat_context", lambda *args: (parameter_path, "{}", None)
    )
    monkeypatch.setattr(
        ui, "_find_memory_answer", lambda *args: ("incorrect memory fallback", None)
    )
    monkeypatch.setattr(ui, "trigger_memory_sync_for_pipeline", lambda *args: None)
    monkeypatch.setattr(ui, "OUTPUT_DIR", tmp_path)
    events = [
        json.loads(event.removeprefix("data: "))
        for event in ui.chat_demo_stream("citation-test", "question", "test-session")
    ]
    assert not any(event["type"] == "error" for event in events)
    answer = next(
        event["data"]["answer"] for event in events if event["type"] == "final"
    )
    assert answer == "Answer [1]"
    assert ("assistant", answer) in history
    tokens = [event for event in events if event["type"] == "token"]
    assert tokens and all(event["is_final"] for event in tokens)
    sources = next(event["data"] for event in events if event["type"] == "sources")
    assert sources[0]["id"] == 1 and sources[0]["content"] == "Document A"
