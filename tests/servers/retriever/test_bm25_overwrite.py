"""A persisted BM25 index can be rebuilt with a different corpus."""

import asyncio
import importlib.util
from pathlib import Path

import pytest

from ultrarag.server import local_server_mode

bm25s = pytest.importorskip("bm25s")


def test_overwrite_existing_bm25_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Path(__file__).resolve().parents[3] / "servers/retriever/src"
    monkeypatch.syspath_prepend(str(source))
    spec = importlib.util.spec_from_file_location(
        "bm25_overwrite_test", source / "retriever.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with local_server_mode():
        spec.loader.exec_module(module)

    provider = module.provider
    index_path = tmp_path / "bm25"
    provider.cfg = {"save_path": str(index_path)}
    provider.model = bm25s.BM25(backend="numpy")
    provider.tokenizer = bm25s.tokenization.Tokenizer(stopwords=[])
    provider.contents = ["obsolete document", "old passage"]
    asyncio.run(provider.bm25_index())
    assert index_path.is_dir()

    provider.contents = ["quasar telescope", "comet ice", "planet orbit"]
    asyncio.run(provider.bm25_index(overwrite=True))

    provider.model = bm25s.BM25.load(str(index_path))
    provider.model.corpus = provider.contents
    provider.tokenizer = bm25s.tokenization.Tokenizer()
    provider.tokenizer.load_vocab(str(index_path))
    provider.tokenizer.load_stopwords(str(index_path))
    result = asyncio.run(provider.bm25_search(["quasar"], top_k=3))["ret_psg"][0]
    assert result[0] == "quasar telescope"
    assert set(result) == set(provider.contents)
