"""Unit tests for the LiteLLM generation backend.

`litellm` is an optional extra, so these stub it in ``sys.modules`` before the
lazy ``import litellm`` inside ``Generation._generate`` runs. No network needed.

Run with the litellm extra installed:
    uv sync --extra litellm
    uv run pytest servers/generation/tests/test_litellm_backend.py -v
"""

import asyncio
import os
import sys
import types

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import generation as gen_mod


def _stub_litellm(content: str = "ok"):
    """Install a fake ``litellm`` module and return it (with a .acompletion spy)."""
    fake = types.ModuleType("litellm")
    for name in (
        "AuthenticationError",
        "RateLimitError",
        "Timeout",
        "APIConnectionError",
        "InternalServerError",
        "ServiceUnavailableError",
    ):
        setattr(fake, name, type(name, (Exception,), {}))

    async def _acompletion(**kwargs):
        _acompletion.calls.append(kwargs)
        message = types.SimpleNamespace(content=content)
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice])

    _acompletion.calls = []
    fake.acompletion = _acompletion
    sys.modules["litellm"] = fake
    return fake


def _make(cfg):
    os.environ.pop("LLM_API_KEY", None)
    g = gen_mod.Generation.__new__(gen_mod.Generation)
    g.generation_init(
        backend_configs={"litellm": cfg},
        sampling_params={"temperature": 0.1, "max_tokens": 8},
        backend="litellm",
    )
    return g


def test_dispatch_forwards_model_creds_and_drop_params():
    fake = _stub_litellm("4")
    g = _make(
        {"model_name": "openai/gpt-4o-mini", "api_key": "sk-x", "base_url": "http://proxy"}
    )
    out = asyncio.run(g.generate(prompt_ls=["What is 2+2?"]))
    assert out == {"ans_ls": ["4"]}
    call = fake.acompletion.calls[0]
    assert call["model"] == "openai/gpt-4o-mini"
    assert call["drop_params"] is True
    assert call["api_key"] == "sk-x"
    assert call["api_base"] == "http://proxy"
    assert call["temperature"] == 0.1


def test_credentials_omitted_when_blank():
    # No api_key / base_url -> LiteLLM should fall back to the provider's env vars,
    # so we must NOT pass empty strings through.
    fake = _stub_litellm("ok")
    g = _make({"model_name": "anthropic/claude-3-5-sonnet"})
    asyncio.run(g.generate(prompt_ls=["hi"]))
    call = fake.acompletion.calls[0]
    assert "api_key" not in call
    assert "api_base" not in call
    assert call["drop_params"] is True


def test_drop_params_opt_out():
    fake = _stub_litellm("ok")
    g = _make({"model_name": "openai/gpt-4o-mini", "drop_params": False})
    asyncio.run(g.generate(prompt_ls=["hi"]))
    assert fake.acompletion.calls[0]["drop_params"] is False


def test_missing_model_name_raises():
    _stub_litellm("ok")
    try:
        _make({"api_key": "sk-x"})
    except ValueError:
        return
    raise AssertionError("expected ValueError when model_name is missing")


def test_multiturn_routes_through_litellm():
    fake = _stub_litellm("Ada")
    g = _make({"model_name": "gemini/gemini-1.5-pro"})
    out = asyncio.run(
        g.multiturn_generate(
            messages=[
                {"role": "user", "content": "My name is Ada."},
                {"role": "assistant", "content": "Hi Ada."},
                {"role": "user", "content": "What is my name?"},
            ]
        )
    )
    assert out == {"ans_ls": ["Ada"]}
    assert fake.acompletion.calls[0]["model"] == "gemini/gemini-1.5-pro"
