"""Unit tests for the OpenAI generation backend's retry loop. No network needed."""

import asyncio
import os
import sys
import types

import httpx
import openai

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
