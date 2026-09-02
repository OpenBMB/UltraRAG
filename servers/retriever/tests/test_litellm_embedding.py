"""Unit tests for the LiteLLM embedding backend in the retriever server.

`litellm` is an optional extra, so these stub it in ``sys.modules`` before the
lazy ``import litellm`` inside ``Retriever._api_embed_texts`` runs. No network.

Model names below are arbitrary fixtures — the backend never hardcodes a model,
it always reads ``model_name`` from config.

    uv sync --extra litellm
    uv run pytest servers/retriever/tests/test_litellm_embedding.py -v
"""

import asyncio
import os
import sys
import types

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import retriever as r_mod


def _stub_litellm(dim: int = 3, as_object: bool = False):
    """Install a fake litellm whose aembedding echoes deterministic vectors."""
    fake = types.ModuleType("litellm")

    async def _aembedding(**kwargs):
        _aembedding.calls.append(kwargs)
        n = len(kwargs["input"])
        vecs = [[float(i)] * dim for i in range(n)]
        if as_object:
            data = [types.SimpleNamespace(embedding=v) for v in vecs]
        else:
            data = [{"embedding": v, "index": i} for i, v in enumerate(vecs)]
        return types.SimpleNamespace(data=data)

    _aembedding.calls = []
    fake.aembedding = _aembedding
    sys.modules["litellm"] = fake
    return fake


def _make(model_name="provider/embed-model", api_key=None, base_url=None):
    os.environ.pop("RETRIEVER_API_KEY", None)
    r = r_mod.Retriever.__new__(r_mod.Retriever)
    r.backend = "litellm"
    r.model_name = model_name
    r.litellm_api_key = api_key
    r.litellm_base_url = base_url
    r.litellm_drop_params = True
    r.batch_size = 2
    r.openai_concurrency = 2
    return r


def _embed(r, texts):
    return asyncio.run(
        r._api_embed_texts(texts, batch_size=r.batch_size, concurrency=2, desc="t")
    )


def test_dispatch_and_drop_params_and_creds():
    fake = _stub_litellm(dim=4)
    r = _make(model_name="cohere/embed-english-v3.0", api_key="sk-x", base_url="http://proxy")
    out = _embed(r, ["a", "b", "c"])
    assert len(out) == 3 and all(len(v) == 4 for v in out)
    call = fake.aembedding.calls[0]
    assert call["model"] == "cohere/embed-english-v3.0"
    assert call["drop_params"] is True
    assert call["api_key"] == "sk-x"
    assert call["api_base"] == "http://proxy"


def test_credentials_omitted_when_blank():
    fake = _stub_litellm()
    r = _make()  # no key / base_url
    _embed(r, ["a"])
    call = fake.aembedding.calls[0]
    assert "api_key" not in call
    assert "api_base" not in call


def test_parses_object_style_response():
    _stub_litellm(dim=5, as_object=True)
    r = _make()
    out = _embed(r, ["a", "b"])
    assert len(out) == 2 and all(len(v) == 5 for v in out)


def test_batching_covers_all_inputs():
    _stub_litellm(dim=2)
    r = _make()
    texts = [f"t{i}" for i in range(5)]  # batch_size=2 -> 3 batches
    out = _embed(r, texts)
    assert len(out) == 5


def test_size_mismatch_raises():
    fake = _stub_litellm()

    async def _bad(**kwargs):
        return types.SimpleNamespace(data=[{"embedding": [0.0]}])  # 1 vec for N inputs

    fake.aembedding = _bad
    r = _make()
    try:
        _embed(r, ["a", "b"])
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError on size mismatch")
