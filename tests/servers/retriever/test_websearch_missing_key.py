"""Missing Tavily credentials should produce an actionable tool error."""

import importlib
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ultrarag.errors import ToolError


def test_tavily_missing_key_is_a_tool_error(monkeypatch: pytest.MonkeyPatch) -> None:
    source = Path(__file__).resolve().parents[3] / "servers/retriever/src"
    monkeypatch.syspath_prepend(str(source))
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    class MissingAPIKeyError(Exception):
        def __init__(self) -> None:
            super().__init__("Missing API key")

    monkeypatch.setitem(
        sys.modules,
        "tavily",
        SimpleNamespace(
            AsyncTavilyClient=lambda **kwargs: None,
            BadRequestError=Exception,
            UsageLimitExceededError=Exception,
            InvalidAPIKeyError=Exception,
            MissingAPIKeyError=MissingAPIKeyError,
        ),
    )
    backend = importlib.import_module("websearch_backends.tavily_backend")
    with pytest.raises(ToolError, match="TAVILY_API_KEY"):
        backend.TavilyWebSearchBackend({}, logging.getLogger("test"))
