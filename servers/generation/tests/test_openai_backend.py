"""Unit tests for the OpenAI generation backend's retry loop. No network needed."""

import asyncio
import os
import sys
import types

import httpx
import openai
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import generation as gen_mod


def _make(base_delay=0.001):
    g = gen_mod.Generation.__new__(gen_mod.Generation)
    g.generation_init(
        backend_configs={
            "openai": {
                "model_name": "gpt-x",
                "base_url": "http://localhost:1/v1",
                "api_key": "sk-test",
                "base_delay": base_delay,
            }
        },
        sampling_params={"temperature": 0.1, "max_tokens": 8},
        backend="openai",
    )
    return g


def _rate_limit_error():
    request = httpx.Request("POST", "http://localhost:1/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return openai.RateLimitError("slow down", response=response, body=None)


def test_rate_limit_is_retried_instead_of_aborting():
    g = _make()
    calls = []

    async def _create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise _rate_limit_error()
        message = types.SimpleNamespace(content="ok")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    g.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=_create))
    )

    out = asyncio.run(g.generate(prompt_ls=["hi"]))

    assert out == {"ans_ls": ["ok"]}
    assert len(calls) == 2


def test_exhausted_retries_raise_instead_of_returning_error_as_an_answer():
    g = _make()
    g._retries = 1

    async def limited(**kwargs):
        raise _rate_limit_error()

    g.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=limited))
    )
    with pytest.raises(gen_mod.ToolError, match="failed after 1 attempts"):
        asyncio.run(g.generate(prompt_ls=["hi"]))


def test_failed_generation_cancels_and_drains_its_sibling_requests():
    g = _make()
    g._max_concurrency = 2

    async def exercise():
        sibling_started = asyncio.Event()
        sibling_finished = asyncio.Event()

        async def create(**kwargs):
            if kwargs["messages"][-1]["content"] == "fail":
                await sibling_started.wait()
                raise ValueError("failed request")
            sibling_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                sibling_finished.set()

        g.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
        )
        with pytest.raises(gen_mod.ToolError, match="failed request"):
            await g.generate(prompt_ls=["fail", "pending"])
        # Assert inside the live loop; asyncio.run teardown would mask a leak.
        assert sibling_finished.is_set()

    asyncio.run(exercise())


def test_generation_init_does_not_mutate_shared_sampling_parameters():
    g = gen_mod.Generation.__new__(gen_mod.Generation)
    parameters = {"temperature": 0}
    g.generation_init(
        {"openai": {"model_name": "test", "api_key": "test"}},
        parameters,
        extra_params={"seed": 1}, backend="openai",
    )
    assert parameters == {"temperature": 0}
