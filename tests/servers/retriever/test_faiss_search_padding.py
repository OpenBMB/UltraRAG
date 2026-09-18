"""Tests for FAISS result handling in the ``retriever`` MCP server.

FAISS pads its result with ``-1`` when the index holds fewer vectors than the
requested ``top_k``. ``BaseIndexBackend.contents`` is a plain list, so indexing
it with ``-1`` returns the *last* document instead of signalling a miss: the
padding used to surface as duplicated passages, and in a RAG pipeline those
passages feed the generator as if they were retrieved evidence.

``faiss`` is an optional extra, so the module is skipped when it is missing.

    uv sync --extra retriever
    uv run pytest tests/servers/retriever/test_faiss_search_padding.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

RETRIEVER_SRC = Path(__file__).resolve().parents[3] / "servers" / "retriever" / "src"
sys.path.insert(0, str(RETRIEVER_SRC))

faiss = pytest.importorskip("faiss")

from index_backends.faiss_backend import FaissIndexBackend


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


CONTENTS = ["DOC_A", "DOC_B", "DOC_C"]

# Points straight at DOC_A so the expected top hit is unambiguous.
QUERY = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)


def _backend(tmp_path, contents=CONTENTS):
    """A backend holding one orthogonal embedding per passage."""
    backend = FaissIndexBackend(
        contents=contents,
        config={"index_path": str(tmp_path / "index.index")},
        logger=_Logger(),
    )
    dim = len(contents)
    backend.build_index(
        embeddings=np.eye(dim, dtype=np.float32),
        ids=np.array(list(range(dim)), dtype=np.int64),
    )
    return backend


def test_padding_slots_are_not_returns_as_the_last_document(tmp_path):
    backend = _backend(tmp_path)

    hits = backend.search(QUERY, top_k=5)[0]

    # Before the fix FAISS returned [0, 2, 1, -1, -1]; the two -1 slots were
    # read as contents[-1], so three passages came back as five, two of them
    # copies of the last document.
    assert hits == ["DOC_A", "DOC_C", "DOC_B"], hits
    assert len(set(hits)) == len(hits), f"duplicated passages returned: {hits}"


def test_top_k_below_index_size_still_returns_top_k(tmp_path):
    backend = _backend(tmp_path)

    hits = backend.search(QUERY, top_k=2)[0]

    assert len(hits) == 2
    assert hits[0] == "DOC_A"


def test_top_k_equal_to_index_size_returns_every_passage_once(tmp_path):
    backend = _backend(tmp_path)

    hits = backend.search(QUERY, top_k=3)[0]

    assert sorted(hits) == sorted(CONTENTS)


def test_single_passage_index_does_not_repeat_padding(tmp_path):
    backend = _backend(tmp_path, contents=["ONLY_DOC"])

    hits = backend.search(np.array([[1.0]], dtype=np.float32), top_k=3)[0]

    assert hits == ["ONLY_DOC"]
