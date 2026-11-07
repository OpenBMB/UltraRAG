"""Session cancellation must own both pipeline and direct multi-turn work."""

import asyncio
import json
import sys
import threading
from concurrent.futures import (
    CancelledError as FutureCancelledError,
)
from concurrent.futures import (
    ThreadPoolExecutor,
)
from pathlib import Path
from types import SimpleNamespace

import pytest

from ui.backend import pipeline_manager as pm


def outcome(action):
    try:
        return action()
    except BaseException as exc:
        return exc


@pytest.fixture
def session(monkeypatch):
    manager = pm.SessionManager(None)
    monkeypatch.setattr(pm, "SESSION_MANAGER", manager)
    instance = manager.get_or_create("owned")
    instance._active = True
    instance._pipeline_name = "test"
    monkeypatch.setattr(instance, "_build_demo_backend_overrides", lambda _: {})
    yield instance
    manager.remove("owned")


def test_interrupt_waits_for_pipeline_cleanup(session, monkeypatch):
    started, closed = threading.Event(), threading.Event()

    async def execute(*args, **kwargs):
        started.set()
        try:
            await asyncio.sleep(30)
        finally:
            await asyncio.sleep(0.03)
            closed.set()

    monkeypatch.setattr(pm, "_ensure_client_funcs", lambda: {"exec_pipe": execute})
    with ThreadPoolExecutor() as pool:
        result = pool.submit(outcome, lambda: session.run_chat(None))
        assert started.wait(2)
        try:
            assert session.interrupt_task()
            assert closed.is_set(), (
                "Stop returned before the previous tool finished cleanup"
            )

            async def next_turn(*args, **kwargs):
                return "recovered"

            monkeypatch.setattr(
                pm, "_ensure_client_funcs", lambda: {"exec_pipe": next_turn}
            )
            assert session.run_chat(None) == "recovered"
        finally:
            session.stop()
        assert isinstance(result.result(timeout=3), FutureCancelledError)


def test_concurrent_interrupts_do_not_cancel_cleanup_twice(session, monkeypatch):
    started, closed = threading.Event(), threading.Event()

    async def execute(*args, **kwargs):
        started.set()
        try:
            await asyncio.sleep(30)
        finally:
            await asyncio.sleep(0.1)
            closed.set()

    monkeypatch.setattr(pm, "_ensure_client_funcs", lambda: {"exec_pipe": execute})
    with ThreadPoolExecutor(3) as pool:
        request = pool.submit(outcome, lambda: session.run_chat(None))
        assert started.wait(2)
        stops = [pool.submit(session.interrupt_task) for _ in range(2)]
        try:
            assert all(stop.result(timeout=3) for stop in stops)
            assert closed.is_set()
            assert isinstance(request.result(timeout=3), FutureCancelledError)
        finally:
            session.stop()


def test_concurrent_session_starts_enter_one_client(session, monkeypatch):
    session._active = False
    entered, release = threading.Event(), threading.Event()
    clients = []

    class Client:
        async def __aenter__(self):
            entered.set()
            while not release.is_set():
                await asyncio.sleep(0.01)

        async def __aexit__(self, *args):
            pass

    def create(*args):
        instance = Client()
        clients.append(instance)
        return instance

    monkeypatch.setattr(pm, "_find_pipeline_file", lambda _: Path("test.yaml"))
    monkeypatch.setattr(
        pm, "_resolve_parameter_path", lambda *args, **kwargs: Path("params.yaml")
    )
    monkeypatch.setattr(
        pm,
        "_ensure_client_funcs",
        lambda: {
            "load_ctx": lambda *args, **kwargs: {"mcp_cfg": {}, "server_cfg": {}},
            "create_client": create,
        },
    )
    with ThreadPoolExecutor() as pool:
        first = pool.submit(session.start, "test")
        assert entered.wait(2)
        second = pool.submit(session.start, "test")
        try:
            threading.Event().wait(0.03)
        finally:
            release.set()
        first.result(timeout=3)
        second.result(timeout=3)
    assert len(clients) == 1


def test_timed_out_multiturn_startup_is_cancelled(session, monkeypatch):
    cancelled = []

    class Client:
        async def __aenter__(self):
            pass

        async def __aexit__(self, *args):
            pass

    class Future:
        def result(self, timeout):
            raise TimeoutError("startup timed out")

        def cancel(self):
            cancelled.append(True)

    def submit(coroutine, loop):
        coroutine.close()
        return Future()

    with monkeypatch.context() as patch:
        patch.setattr(pm, "_find_pipeline_file", lambda _: Path("multiturn.yaml"))
        patch.setattr(
            pm, "_resolve_parameter_path", lambda *args, **kwargs: Path("params.yaml")
        )
        patch.setattr(
            pm,
            "_ensure_client_funcs",
            lambda: {
                "load_ctx": lambda *args, **kwargs: {"mcp_cfg": {}, "server_cfg": {}},
                "create_client": lambda *args: Client(),
            },
        )
        patch.setattr(pm.asyncio, "run_coroutine_threadsafe", submit)
        session.init_multiturn_client()
    assert cancelled == [True]
    assert session._multiturn_active is False


def test_overlapping_requests_cannot_replace_the_running_task(session, monkeypatch):
    started, release = threading.Event(), threading.Event()

    async def execute(*args, **kwargs):
        started.set()
        while not release.is_set():
            await asyncio.sleep(0.01)
        return "completed"

    monkeypatch.setattr(pm, "_ensure_client_funcs", lambda: {"exec_pipe": execute})
    with ThreadPoolExecutor() as pool:
        first = pool.submit(outcome, lambda: session.run_chat(None))
        assert started.wait(2)
        second = pool.submit(outcome, lambda: session.run_chat(None))
        try:
            error = second.result(timeout=0.5)
            assert isinstance(error, pm.PipelineManagerError)
            assert "already running" in str(error)
            assert session.interrupt_task()
        finally:
            release.set()
        assert isinstance(first.result(timeout=3), FutureCancelledError)


def test_multiturn_interrupt_closes_client_without_saving_partial_history(
    session, monkeypatch
):
    started, release, closed = threading.Event(), threading.Event(), threading.Event()
    saved = []

    class Service:
        def __init__(self, **kwargs):
            self.client = self

        async def close(self):
            closed.set()

        async def multiturn_generate_stream(self, **kwargs):
            yield "partial"
            started.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
            yield "complete"

    monkeypatch.setattr(pm, "load_parameters", lambda _: {})
    monkeypatch.setattr(pm, "_resolve_generation_server_alias", lambda _: "generation")
    monkeypatch.setattr(
        pm, "_auto_save_memory_chat_turn", lambda *args: saved.append(args)
    )
    monkeypatch.setitem(
        sys.modules,
        "servers.generation.src.local_generation",
        SimpleNamespace(LocalGenerationService=Service),
    )
    with ThreadPoolExecutor() as pool:
        result = pool.submit(
            lambda: [
                json.loads(frame.removeprefix("data: "))
                for frame in pm.chat_multiturn_stream(
                    "owned", "question", conversation_history=[]
                )
            ]
        )
        assert started.wait(2)
        try:
            assert session.interrupt_task(), (
                "Direct multi-turn generation was not registered for cancellation"
            )
            assert closed.is_set()
        finally:
            release.set()
        events = result.result(timeout=3)
    assert not any(event["type"] in {"error", "final"} for event in events)
    assert session.get_conversation_history() == []
    assert saved == []
