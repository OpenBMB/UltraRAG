"""Tests for the query-extraction tools in the ``custom`` MCP server.

The ``custom`` server lives under ``servers/custom/src`` rather than inside the
installable ``ultrarag`` package, so its module is made importable here without
installing each server separately.
"""

import sys
from pathlib import Path

CUSTOM_SRC = Path(__file__).resolve().parents[3] / "servers" / "custom" / "src"
sys.path.insert(0, str(CUSTOM_SRC))

import custom  # noqa: E402


def test_r1_searcher_query_extract_returns_tagged_query():
    answers = [
        "Let me reason about this. "
        "<|begin_of_query|>capital of France<|end_of_query|> then continue."
    ]
    result = custom.r1_searcher_query_extract(answers)
    assert result == {"extract_query_list": ["capital of France?"]}


def test_r1_searcher_query_extract_uses_last_query():
    answers = [
        "<|begin_of_query|>first question<|end_of_query|> ... "
        "<|begin_of_query|>second question<|end_of_query|> done."
    ]
    result = custom.r1_searcher_query_extract(answers)
    assert result == {"extract_query_list": ["second question?"]}


def test_r1_searcher_query_extract_appends_question_mark():
    answers = ["<|begin_of_query|>already a question?<|end_of_query|>"]
    result = custom.r1_searcher_query_extract(answers)
    assert result == {"extract_query_list": ["already a question?"]}


def test_r1_searcher_query_extract_without_tag():
    result = custom.r1_searcher_query_extract(["no query tags in this text"])
    assert result == {"extract_query_list": ["There is no query."]}
