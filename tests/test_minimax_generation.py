"""Unit tests for the MiniMax generation backend."""

import asyncio
import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: add the generation server source to sys.path so we can import
# the Generation class without installing the full package.
# ---------------------------------------------------------------------------
_GEN_SRC = os.path.join(
    os.path.dirname(__file__), os.pardir, "servers", "generation", "src"
)
sys.path.insert(0, os.path.abspath(_GEN_SRC))

# We need to stub out ``ultrarag.server`` before importing generation.py
# because it tries to create an MCP server instance at module level.
_stub_server = types.ModuleType("ultrarag.server")


class _StubMCPServer:
    """Minimal stand-in for UltraRAG_MCP_Server."""

    def __init__(self, *a, **kw):
        self.logger = MagicMock()

    def tool(self, *a, **kw):  # noqa: D401
        pass

    def run(self, *a, **kw):
        pass


_stub_server.UltraRAG_MCP_Server = _StubMCPServer  # type: ignore[attr-defined]

# Also stub ``ultrarag`` parent package
_stub_ultrarag = types.ModuleType("ultrarag")
sys.modules.setdefault("ultrarag", _stub_ultrarag)
sys.modules["ultrarag.server"] = _stub_server

# Stub fastmcp.exceptions
_stub_fastmcp = types.ModuleType("fastmcp")
_stub_fastmcp_exc = types.ModuleType("fastmcp.exceptions")


class _ToolError(Exception):
    pass


_stub_fastmcp_exc.ToolError = _ToolError  # type: ignore[attr-defined]
sys.modules.setdefault("fastmcp", _stub_fastmcp)
sys.modules["fastmcp.exceptions"] = _stub_fastmcp_exc

# Now we can import
from generation import Generation  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gen() -> Generation:
    """Create a Generation instance without triggering tool registration."""
    mcp = _StubMCPServer()
    return Generation(mcp)


# ---------------------------------------------------------------------------
# Tests: _clamp_temperature
# ---------------------------------------------------------------------------

class TestClampTemperature:
    def test_clamp_below_min(self):
        result = Generation._clamp_temperature({"temperature": 0.0})
        assert result["temperature"] == 0.01

    def test_clamp_above_max(self):
        result = Generation._clamp_temperature({"temperature": 1.5})
        assert result["temperature"] == 1.0

    def test_within_range(self):
        result = Generation._clamp_temperature({"temperature": 0.7})
        assert result["temperature"] == 0.7

    def test_no_temperature_key(self):
        result = Generation._clamp_temperature({"top_p": 0.9})
        assert "temperature" not in result
        assert result["top_p"] == 0.9

    def test_exact_min(self):
        result = Generation._clamp_temperature({"temperature": 0.01})
        assert result["temperature"] == 0.01

    def test_exact_max(self):
        result = Generation._clamp_temperature({"temperature": 1.0})
        assert result["temperature"] == 1.0

    def test_preserves_other_keys(self):
        result = Generation._clamp_temperature(
            {"temperature": 0.5, "max_tokens": 100, "top_p": 0.9}
        )
        assert result == {"temperature": 0.5, "max_tokens": 100, "top_p": 0.9}

    def test_custom_range(self):
        result = Generation._clamp_temperature(
            {"temperature": 0.5}, low=0.6, high=0.8
        )
        assert result["temperature"] == 0.6


# ---------------------------------------------------------------------------
# Tests: _strip_think_tags
# ---------------------------------------------------------------------------

class TestStripThinkTags:
    def test_no_tags(self):
        assert Generation._strip_think_tags("Hello world") == "Hello world"

    def test_single_tag(self):
        text = "<think>internal reasoning</think>Final answer."
        assert Generation._strip_think_tags(text) == "Final answer."

    def test_multiline_tag(self):
        text = "<think>\nstep 1\nstep 2\n</think>\nHere is the answer."
        assert Generation._strip_think_tags(text) == "Here is the answer."

    def test_multiple_tags(self):
        text = "<think>a</think>Hello <think>b</think>world"
        assert Generation._strip_think_tags(text) == "Hello world"

    def test_empty_tag(self):
        text = "<think></think>Answer"
        assert Generation._strip_think_tags(text) == "Answer"

    def test_no_content_after_strip(self):
        text = "<think>all thinking</think>"
        assert Generation._strip_think_tags(text) == ""

    def test_nested_angle_brackets(self):
        text = "<think>x > 0 and y < 1</think>Result"
        assert Generation._strip_think_tags(text) == "Result"


# ---------------------------------------------------------------------------
# Tests: generation_init for minimax backend
# ---------------------------------------------------------------------------

class TestMinimaxInit:
    def test_init_with_api_key_in_config(self):
        gen = _make_gen()
        gen.generation_init(
            backend_configs={
                "minimax": {
                    "model_name": "MiniMax-M2.7",
                    "api_key": "test-key-123",
                }
            },
            sampling_params={"temperature": 0.7, "max_tokens": 100},
            backend="minimax",
        )
        assert gen.backend == "minimax"
        assert gen.model_name == "MiniMax-M2.7"
        assert gen._strip_think is True

    def test_init_with_env_api_key(self):
        gen = _make_gen()
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-key-456"}):
            gen.generation_init(
                backend_configs={"minimax": {}},
                sampling_params={"temperature": 0.5},
                backend="minimax",
            )
        assert gen.backend == "minimax"
        assert gen.model_name == "MiniMax-M2.7"  # default

    def test_init_missing_api_key(self):
        gen = _make_gen()
        with patch.dict(os.environ, {}, clear=True):
            # Remove both env vars
            env = os.environ.copy()
            env.pop("MINIMAX_API_KEY", None)
            env.pop("LLM_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(ValueError, match="api_key is required"):
                    gen.generation_init(
                        backend_configs={"minimax": {}},
                        sampling_params={},
                        backend="minimax",
                    )

    def test_temperature_clamped(self):
        gen = _make_gen()
        gen.generation_init(
            backend_configs={
                "minimax": {"api_key": "k", "model_name": "MiniMax-M2.5"}
            },
            sampling_params={"temperature": 0.0, "max_tokens": 512},
            backend="minimax",
        )
        assert gen.sampling_params["temperature"] == 0.01
        assert gen.sampling_params["max_tokens"] == 512

    def test_temperature_high_clamped(self):
        gen = _make_gen()
        gen.generation_init(
            backend_configs={"minimax": {"api_key": "k"}},
            sampling_params={"temperature": 2.0},
            backend="minimax",
        )
        assert gen.sampling_params["temperature"] == 1.0

    def test_default_model_name(self):
        gen = _make_gen()
        gen.generation_init(
            backend_configs={"minimax": {"api_key": "k"}},
            sampling_params={},
            backend="minimax",
        )
        assert gen.model_name == "MiniMax-M2.7"

    def test_custom_model_name(self):
        gen = _make_gen()
        gen.generation_init(
            backend_configs={
                "minimax": {"api_key": "k", "model_name": "MiniMax-M2.5-highspeed"}
            },
            sampling_params={},
            backend="minimax",
        )
        assert gen.model_name == "MiniMax-M2.5-highspeed"

    def test_strip_think_disabled(self):
        gen = _make_gen()
        gen.generation_init(
            backend_configs={
                "minimax": {"api_key": "k", "strip_think_tags": False}
            },
            sampling_params={},
            backend="minimax",
        )
        assert gen._strip_think is False

    def test_concurrency_retries_defaults(self):
        gen = _make_gen()
        gen.generation_init(
            backend_configs={"minimax": {"api_key": "k"}},
            sampling_params={},
            backend="minimax",
        )
        assert gen._max_concurrency == 1
        assert gen._retries == 3
        assert gen._base_delay == 1.0

    def test_custom_concurrency(self):
        gen = _make_gen()
        gen.generation_init(
            backend_configs={
                "minimax": {"api_key": "k", "concurrency": 8, "retries": 5}
            },
            sampling_params={},
            backend="minimax",
        )
        assert gen._max_concurrency == 8
        assert gen._retries == 5

    def test_extra_params_set(self):
        gen = _make_gen()
        gen.generation_init(
            backend_configs={"minimax": {"api_key": "k"}},
            sampling_params={"temperature": 0.7},
            extra_params={"response_format": {"type": "json_object"}},
            backend="minimax",
        )
        assert gen.sampling_params["extra_body"] == {
            "response_format": {"type": "json_object"}
        }

    def test_default_base_url(self):
        gen = _make_gen()
        gen.generation_init(
            backend_configs={"minimax": {"api_key": "k"}},
            sampling_params={},
            backend="minimax",
        )
        assert str(gen.client.base_url).rstrip("/").endswith("api.minimax.io/v1")

    def test_custom_base_url(self):
        gen = _make_gen()
        gen.generation_init(
            backend_configs={
                "minimax": {
                    "api_key": "k",
                    "base_url": "https://custom.example.com/v1",
                }
            },
            sampling_params={},
            backend="minimax",
        )
        assert "custom.example.com" in str(gen.client.base_url)


# ---------------------------------------------------------------------------
# Tests: _generate with minimax backend (mocked API calls)
# ---------------------------------------------------------------------------

class TestMinimaxGenerate:
    def _setup_gen(self, strip_think: bool = True) -> Generation:
        gen = _make_gen()
        gen.backend = "minimax"
        gen.model_name = "MiniMax-M2.7"
        gen.sampling_params = {"temperature": 0.7}
        gen._max_concurrency = 2
        gen._retries = 1
        gen._base_delay = 0.01
        gen._strip_think = strip_think

        # Mock AsyncOpenAI client
        gen.client = MagicMock()
        return gen

    def _mock_response(self, text: str):
        choice = MagicMock()
        choice.message.content = text
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_single_message(self):
        gen = self._setup_gen()
        gen.client.chat.completions.create = AsyncMock(
            return_value=self._mock_response("Hello!")
        )

        msg_ls = [[{"role": "user", "content": "Hi"}]]
        result = asyncio.get_event_loop().run_until_complete(gen._generate(msg_ls))
        assert result == ["Hello!"]

    def test_multiple_messages(self):
        gen = self._setup_gen()
        responses = [
            self._mock_response("Answer 1"),
            self._mock_response("Answer 2"),
        ]
        gen.client.chat.completions.create = AsyncMock(side_effect=responses)

        msg_ls = [
            [{"role": "user", "content": "Q1"}],
            [{"role": "user", "content": "Q2"}],
        ]
        result = asyncio.get_event_loop().run_until_complete(gen._generate(msg_ls))
        assert len(result) == 2
        assert "Answer 1" in result
        assert "Answer 2" in result

    def test_think_tags_stripped(self):
        gen = self._setup_gen(strip_think=True)
        gen.client.chat.completions.create = AsyncMock(
            return_value=self._mock_response(
                "<think>reasoning here</think>The final answer."
            )
        )

        msg_ls = [[{"role": "user", "content": "Q"}]]
        result = asyncio.get_event_loop().run_until_complete(gen._generate(msg_ls))
        assert result == ["The final answer."]

    def test_think_tags_preserved(self):
        gen = self._setup_gen(strip_think=False)
        gen.client.chat.completions.create = AsyncMock(
            return_value=self._mock_response(
                "<think>reasoning</think>Answer"
            )
        )

        msg_ls = [[{"role": "user", "content": "Q"}]]
        result = asyncio.get_event_loop().run_until_complete(gen._generate(msg_ls))
        assert result == ["<think>reasoning</think>Answer"]

    def test_empty_response(self):
        gen = self._setup_gen()
        choice = MagicMock()
        choice.message.content = None
        resp = MagicMock()
        resp.choices = [choice]
        gen.client.chat.completions.create = AsyncMock(return_value=resp)

        msg_ls = [[{"role": "user", "content": "Q"}]]
        result = asyncio.get_event_loop().run_until_complete(gen._generate(msg_ls))
        assert result == [""]

    def test_system_prompt_included(self):
        gen = self._setup_gen()
        gen.client.chat.completions.create = AsyncMock(
            return_value=self._mock_response("OK")
        )

        msg_ls = [
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ]
        ]
        result = asyncio.get_event_loop().run_until_complete(gen._generate(msg_ls))
        assert result == ["OK"]

        # Verify system message was passed
        call_args = gen.client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
        assert messages[0]["role"] == "system"


# ---------------------------------------------------------------------------
# Tests: generate() and multiturn_generate() with minimax backend
# ---------------------------------------------------------------------------

class TestMinimaxHighLevelGenerate:
    def _setup_gen(self) -> Generation:
        gen = _make_gen()
        gen.backend = "minimax"
        gen.model_name = "MiniMax-M2.7"
        gen.sampling_params = {"temperature": 0.7}
        gen._max_concurrency = 1
        gen._retries = 1
        gen._base_delay = 0.01
        gen._strip_think = True
        gen.client = MagicMock()
        return gen

    def _mock_response(self, text: str):
        choice = MagicMock()
        choice.message.content = text
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_generate_single_prompt(self):
        gen = self._setup_gen()
        gen.client.chat.completions.create = AsyncMock(
            return_value=self._mock_response("Response")
        )

        result = asyncio.get_event_loop().run_until_complete(
            gen.generate(["What is RAG?"], system_prompt="Be concise.")
        )
        assert result == {"ans_ls": ["Response"]}

    def test_generate_empty_prompt_list(self):
        gen = self._setup_gen()
        result = asyncio.get_event_loop().run_until_complete(
            gen.generate([], system_prompt="")
        )
        assert result == {"ans_ls": []}

    def test_multiturn_generate(self):
        gen = self._setup_gen()
        gen.client.chat.completions.create = AsyncMock(
            return_value=self._mock_response("Sure, I can help.")
        )

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "What is MiniMax?"},
        ]
        result = asyncio.get_event_loop().run_until_complete(
            gen.multiturn_generate(messages, system_prompt="You are helpful.")
        )
        assert result == {"ans_ls": ["Sure, I can help."]}

    def test_multiturn_empty_messages(self):
        gen = self._setup_gen()
        result = asyncio.get_event_loop().run_until_complete(
            gen.multiturn_generate([], system_prompt="")
        )
        assert result == {"ans_ls": []}
