"""The openai reranker backend sends the configured api_key."""

import asyncio
import importlib.util
from pathlib import Path

import pytest

from ultrarag.server import local_server_mode

web = pytest.importorskip("aiohttp.web")


def _load_module(monkeypatch: pytest.MonkeyPatch):
    source = Path(__file__).resolve().parents[3] / "servers/reranker/src"
    monkeypatch.syspath_prepend(str(source))
    spec = importlib.util.spec_from_file_location(
        "reranker_api_key_test", source / "reranker.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with local_server_mode():
        spec.loader.exec_module(module)
    return module


async def _run(module, api_key):
    seen = []

    async def handler(request):
        seen.append(request.headers.get("Authorization"))
        return web.json_response(
            {"results": [{"document": {"text": "b"}}, {"document": "a"}]}
        )

    app = web.Application()
    app.router.add_post("/rerank", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        cfg = {"model_name": "m", "base_url": f"http://127.0.0.1:{port}/rerank"}
        if api_key is not None:
            cfg["api_key"] = api_key
        await module.provider.reranker_init(
            model_name_or_path="m",
            backend_configs={"openai": cfg},
            batch_size=1,
            backend="openai",
        )
        out = await module.provider.reranker_rerank(["q"], [["a", "b"]], top_k=2)
    finally:
        await runner.cleanup()
    return seen, out


def test_api_key_sent_as_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    module = _load_module(monkeypatch)
    seen, out = asyncio.run(_run(module, "secret"))
    assert seen == ["Bearer secret"]
    assert out == {"rerank_psg": [["b", "a"]]}


def test_no_header_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    module = _load_module(monkeypatch)
    seen, _ = asyncio.run(_run(module, None))
    assert seen == [None]
