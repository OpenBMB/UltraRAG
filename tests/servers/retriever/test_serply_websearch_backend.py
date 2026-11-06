"""Tests for the Serply web search backend, run against a local aiohttp server."""

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
        app.router.add_get("/v1/search", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        port = runner.addresses[0][1]
        config["base_url"] = f"http://127.0.0.1:{port}/v1/search"
        try:
            backend = create_websearch_backend(
                "serply", logging.getLogger("serply"), config=config
            )
            return await backend.search(queries, top_k=top_k)
        finally:
            await runner.cleanup()

    return asyncio.run(main())


def test_search_returns_descriptions_with_key_and_user_agent(monkeypatch):
    monkeypatch.delenv("SERPLY_API_KEY", raising=False)
    seen = []

    async def handler(request):
        seen.append(
            (request.query["num"], request.headers["X-Api-Key"], request.headers["User-Agent"])
        )
        results = [{"description": f"{request.query['q']}-{i}"} for i in range(5)]
        return web.json_response({"results": results})

    assert _search(handler, {"api_key": "k"}, ["a"], 2) == [["a-0", "a-1"]]
    assert seen == [("2", "k", "UltraRAG")]


def test_missing_key_and_unauthorized_raise_tool_error(monkeypatch):
    monkeypatch.delenv("SERPLY_API_KEY", raising=False)

    async def handler(request):
        return web.json_response({}, status=401)

    for config in ({}, {"api_key": "bad"}):
        with pytest.raises(ToolError):
            _search(handler, config, ["a"], 3)
