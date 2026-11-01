"""Tests for the Outcrawl web search backend, run against a local aiohttp server."""

import asyncio
import logging
import sys
from pathlib import Path

import pytest
from aiohttp import web
from fastmcp.exceptions import ToolError

RETRIEVER_SRC = Path(__file__).resolve().parents[3] / "servers" / "retriever" / "src"
sys.path.insert(0, str(RETRIEVER_SRC))

from websearch_backends import create_websearch_backend


def _search(handler, config, queries, top_k):
    async def main():
        app = web.Application()
        app.router.add_post("/v1/search", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        port = runner.addresses[0][1]
        config["base_url"] = f"http://127.0.0.1:{port}/v1/search"
        try:
            backend = create_websearch_backend(
                "outcrawl", logging.getLogger("outcrawl"), config=config
            )
            return await backend.search(queries, top_k=top_k)
        finally:
            await runner.cleanup()

    return asyncio.run(main())


def test_search_returns_snippets_with_key_and_limit(monkeypatch):
    monkeypatch.delenv("OUTCRAWL_API_KEY", raising=False)
    seen = []

    async def handler(request):
        body = await request.json()
        seen.append(
            (
                body["query"],
                body["limit"],
                request.headers["Authorization"],
                request.headers["User-Agent"],
            )
        )
        hits = [
            {"url": "https://a", "title": "A", "snippet": "snip-a", "rank": 1},
            {"url": "https://b", "title": "B", "snippet": "", "rank": 2},
            {"url": "https://c", "title": "C", "snippet": "snip-c", "rank": 3},
            {"url": "https://d", "title": "D", "snippet": "snip-d", "rank": 4},
        ]
        return web.json_response({"hits": hits, "usage": {"credits": "1"}})

    assert _search(handler, {"api_key": "k"}, ["q"], 2) == [["snip-a", "snip-c"]]
    assert seen == [("q", 2, "Bearer k", "UltraRAG")]


def test_scraped_markdown_is_preferred_over_snippet(monkeypatch):
    monkeypatch.delenv("OUTCRAWL_API_KEY", raising=False)
    seen = []

    async def handler(request):
        seen.append(await request.json())
        hits = [
            {"url": "https://a", "snippet": "snip-a", "markdown": "# Page A"},
            {"url": "https://b", "snippet": "snip-b"},
        ]
        return web.json_response({"hits": hits})

    config = {
        "api_key": "k",
        "search_kwargs": {"scrape": True, "formats": ["markdown"]},
    }
    assert _search(handler, config, ["q"], 5) == [["# Page A", "snip-b"]]
    assert seen == [{"query": "q", "scrape": True, "formats": ["markdown"], "limit": 5}]


def test_env_key_and_empty_results(monkeypatch):
    monkeypatch.setenv("OUTCRAWL_API_KEY", "env-key")
    seen = []

    async def handler(request):
        seen.append(request.headers["Authorization"])
        return web.json_response({"hits": []})

    assert _search(handler, {}, ["a", "b"], 3) == [[], []]
    assert seen == ["Bearer env-key", "Bearer env-key"]


def test_missing_key_raises_tool_error(monkeypatch):
    monkeypatch.delenv("OUTCRAWL_API_KEY", raising=False)
    with pytest.raises(ToolError):
        create_websearch_backend("outcrawl", logging.getLogger("outcrawl"), config={})


@pytest.mark.parametrize("status", [401, 403, 402])
def test_auth_and_credit_errors_raise_without_retry(monkeypatch, status):
    monkeypatch.delenv("OUTCRAWL_API_KEY", raising=False)
    calls = []

    async def handler(request):
        calls.append(1)
        return web.json_response({"code": "error"}, status=status)

    with pytest.raises(ToolError):
        _search(handler, {"api_key": "bad", "retries": 3, "base_delay": 0}, ["a"], 3)
    assert len(calls) == 1


def test_rate_limit_honors_retry_after_then_succeeds(monkeypatch):
    monkeypatch.delenv("OUTCRAWL_API_KEY", raising=False)
    calls = []

    async def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return web.json_response({}, status=429, headers={"Retry-After": "0"})
        return web.json_response({"hits": [{"snippet": "ok"}]})

    # base_delay is large so the test only finishes quickly if Retry-After is used.
    config = {"api_key": "k", "retries": 2, "base_delay": 30}
    assert _search(handler, config, ["a"], 3) == [["ok"]]
    assert len(calls) == 2


def test_server_errors_exhaust_retries_and_return_empty(monkeypatch):
    monkeypatch.delenv("OUTCRAWL_API_KEY", raising=False)
    calls = []

    async def handler(request):
        calls.append(1)
        return web.json_response({}, status=500)

    config = {"api_key": "k", "retries": 2, "base_delay": 0}
    assert _search(handler, config, ["a"], 3) == [[]]
    assert len(calls) == 2
