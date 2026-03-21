"""Integration tests for the MiniMax generation backend.

These tests call the real MiniMax API and are skipped when
MINIMAX_API_KEY is not set in the environment.
"""

import asyncio
import os
import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Bootstrap (same stub approach as unit tests)
# ---------------------------------------------------------------------------
_GEN_SRC = os.path.join(
    os.path.dirname(__file__), os.pardir, "servers", "generation", "src"
)
sys.path.insert(0, os.path.abspath(_GEN_SRC))

from unittest.mock import MagicMock

_stub_server = types.ModuleType("ultrarag.server")


class _StubMCPServer:
    def __init__(self, *a, **kw):
        self.logger = MagicMock()

    def tool(self, *a, **kw):
        pass

    def run(self, *a, **kw):
        pass


_stub_server.UltraRAG_MCP_Server = _StubMCPServer  # type: ignore[attr-defined]
_stub_ultrarag = types.ModuleType("ultrarag")
sys.modules.setdefault("ultrarag", _stub_ultrarag)
sys.modules["ultrarag.server"] = _stub_server

_stub_fastmcp = types.ModuleType("fastmcp")
_stub_fastmcp_exc = types.ModuleType("fastmcp.exceptions")


class _ToolError(Exception):
    pass


_stub_fastmcp_exc.ToolError = _ToolError  # type: ignore[attr-defined]
sys.modules.setdefault("fastmcp", _stub_fastmcp)
sys.modules["fastmcp.exceptions"] = _stub_fastmcp_exc

from generation import Generation  # noqa: E402

# ---------------------------------------------------------------------------

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
skip_no_key = pytest.mark.skipif(
    not MINIMAX_API_KEY,
    reason="MINIMAX_API_KEY not set – skipping live integration tests",
)


def _make_gen() -> Generation:
    mcp = _StubMCPServer()
    return Generation(mcp)


@skip_no_key
class TestMinimaxLiveGeneration:
    """Integration tests that hit the real MiniMax API."""

    def _init_gen(self, model: str = "MiniMax-M2.7") -> Generation:
        gen = _make_gen()
        gen.generation_init(
            backend_configs={
                "minimax": {
                    "model_name": model,
                    "api_key": MINIMAX_API_KEY,
                    "concurrency": 1,
                }
            },
            sampling_params={"temperature": 0.7, "max_tokens": 128},
            backend="minimax",
        )
        return gen

    def test_simple_generation(self):
        gen = self._init_gen()
        result = asyncio.get_event_loop().run_until_complete(
            gen.generate(
                ["What is the capital of France? Reply with just the city name."],
                system_prompt="Answer concisely.",
            )
        )
        assert "ans_ls" in result
        assert len(result["ans_ls"]) == 1
        assert len(result["ans_ls"][0]) > 0

    def test_multiturn_conversation(self):
        gen = self._init_gen()
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user", "content": "And what is 3+3?"},
        ]
        result = asyncio.get_event_loop().run_until_complete(
            gen.multiturn_generate(messages)
        )
        assert "ans_ls" in result
        assert len(result["ans_ls"]) == 1
        assert "6" in result["ans_ls"][0]

    def test_system_prompt(self):
        gen = self._init_gen()
        result = asyncio.get_event_loop().run_until_complete(
            gen.generate(
                ["What are you?"],
                system_prompt="You are a pirate. Always reply in pirate speak.",
            )
        )
        assert "ans_ls" in result
        assert len(result["ans_ls"][0]) > 0
