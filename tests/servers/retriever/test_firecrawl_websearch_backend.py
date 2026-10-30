"""Tests for the Firecrawl web search backend, run against a local aiohttp server."""

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
        app.router.add_post("/v2/search", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        port = runner.addresses[0][1]
        config["base_url"] = f"http://127.0.0.1:{port}/v2/search"
        try:
            backend = create_websearch_backend(
                "firecrawl", logging.getLogger("firecrawl"), config=config
            )
            return await backend.search(queries, top_k=top_k)
        finally:
            await runner.cleanup()

    return asyncio.run(main())


def test_search_sends_key_limit_and_integration(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    seen = []

    async def handler(request):
        body = await request.json()
        seen.append(
            (
                body,
                request.headers["Authorization"],
                request.headers["User-Agent"],
            )
        )
        web_results = [
            {"url": "https://a", "title": "A", "description": "desc-a"},
            {"url": "https://b", "title": "B", "description": ""},
            {"url": "https://c", "title": "C", "description": "desc-c"},
            {"url": "https://d", "title": "D", "description": "desc-d"},
        ]
        return web.json_response({"success": True, "data": {"web": web_results}})

    assert _search(handler, {"api_key": "k"}, ["q"], 2) == [["desc-a", "desc-c"]]
    assert seen == [
        (
            {"query": "q", "integration": "_ultrarag", "limit": 2},
            "Bearer k",
            "UltraRAG",
        )
    ]


def test_markdown_preferred_and_search_kwargs_passed(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    seen = []

    async def handler(request):
        seen.append(await request.json())
        web_results = [
            {"url": "https://a", "description": "desc-a", "markdown": "# Page A"},
            {"url": "https://b", "description": "desc-b"},
        ]
        return web.json_response({"success": True, "data": {"web": web_results}})

    scrape = {"scrapeOptions": {"formats": ["markdown"]}}
    config = {"api_key": "k", "search_kwargs": scrape}
    assert _search(handler, config, ["q"], 5) == [["# Page A", "desc-b"]]
    assert seen[0]["scrapeOptions"] == {"formats": ["markdown"]}


def test_env_key_and_empty_results(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "env-key")
    seen = []

    async def handler(request):
        seen.append(request.headers["Authorization"])
        return web.json_response({"success": True, "data": {}})

    assert _search(handler, {}, ["a", "b"], 3) == [[], []]
    assert seen == ["Bearer env-key", "Bearer env-key"]


def test_missing_key_raises_tool_error(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    with pytest.raises(ToolError):
        create_websearch_backend("firecrawl", logging.getLogger("firecrawl"), config={})


@pytest.mark.parametrize("status", [401, 403, 402])
def test_auth_and_credit_errors_raise_without_retry(monkeypatch, status):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    calls = []

    async def handler(request):
        calls.append(1)
        return web.json_response({}, status=status)

    with pytest.raises(ToolError):
        _search(handler, {"api_key": "bad", "retries": 3, "base_delay": 0}, ["a"], 3)
    assert len(calls) == 1


def test_rate_limit_honors_retry_after_then_succeeds(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    calls = []

    async def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return web.json_response({}, status=429, headers={"Retry-After": "0"})
        return web.json_response(
            {"success": True, "data": {"web": [{"description": "ok"}]}}
        )

    # base_delay is large so the test only finishes quickly if Retry-After is used.
    config = {"api_key": "k", "retries": 2, "base_delay": 30}
    assert _search(handler, config, ["a"], 3) == [["ok"]]
    assert len(calls) == 2


def test_rate_limit_exhausts_retries_and_returns_empty(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    calls = []

    async def handler(request):
        calls.append(1)
        return web.json_response({}, status=429)

    config = {"api_key": "k", "retries": 2, "base_delay": 0}
    assert _search(handler, config, ["a"], 3) == [[]]
    assert len(calls) == 2
