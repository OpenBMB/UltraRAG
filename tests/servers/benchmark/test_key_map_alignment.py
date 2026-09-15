"""Tests for benchmark key mapping in the ``benchmark`` MCP server.

The ``benchmark`` server lives under ``servers/benchmark/src`` rather than
inside the installable ``ultrarag`` package, and importing it pulls in the MCP
app and pandas. The pure helper under test is lifted out of the source with
``ast`` instead, the way ``tests/servers/evaluation`` does it.
"""

import ast
import random
from pathlib import Path
from typing import Any, Dict, List

BENCHMARK_SRC = (
    Path(__file__).resolve().parents[3] / "servers" / "benchmark" / "src" / "benchmark.py"
)


class _Logger:
    def __init__(self):
        self.warnings = []

    def warning(self, msg):
        self.warnings.append(msg)

    def debug(self, msg):
        pass

    def info(self, msg):
        pass


class _App:
    def __init__(self):
        self.logger = _Logger()


def _load(data):
    """Compile ``_load_from_local`` with a stub app and a fixed record list."""
    tree = ast.parse(BENCHMARK_SRC.read_text())
    module = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_load_from_local"
        ],
        type_ignores=[],
    )
    app = _App()
    namespace = {
        "random": random,
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "app": app,
        # _load_from_local reads the file through this helper; hand it the
        # records directly so the test does not need a fixture on disk.
        "_load_data_from_file": lambda path, limit: (
            data if limit == -1 else data[:limit]
        ),
    }
    exec(compile(module, str(BENCHMARK_SRC), "exec"), namespace)
    return namespace["_load_from_local"], app


KEY_MAP = {"q_ls": "question", "gt_ls": "answer"}

RAGGED = [
    {"question": "q1", "answer": "a1"},
    {"question": "q2"},
    {"question": "q3", "answer": "a3"},
]

COMPLETE = [
    {"question": "q1", "answer": "a1"},
    {"question": "q2", "answer": "a2"},
    {"question": "q3", "answer": "a3"},
]


def test_a_record_missing_a_key_does_not_shift_the_other_columns():
    load, _ = _load(RAGGED)
    ret = load("data.jsonl", KEY_MAP, -1)
    # Before the fix q_ls kept q2 while gt_ls did not, so q2 was scored
    # against a3 and q3 was dropped by zip().
    assert ret["q_ls"] == ["q1", "q3"]
    assert ret["gt_ls"] == ["a1", "a3"]


def test_columns_are_the_same_length():
    load, _ = _load(RAGGED)
    ret = load("data.jsonl", KEY_MAP, -1)
    assert len(set(len(v) for v in ret.values())) == 1


def test_skipped_records_are_reported():
    load, app = _load(RAGGED)
    load("data.jsonl", KEY_MAP, -1)
    assert len(app.logger.warnings) == 1
    assert "Skipped 1 of 3" in app.logger.warnings[0]


def test_shuffling_a_ragged_file_does_not_raise():
    load, _ = _load(RAGGED)
    # Before the fix the shuffle branch indexed every column with the first
    # column's range, so a shorter column raised IndexError.
    ret = load("data.jsonl", KEY_MAP, -1, is_shuffle=True)
    assert sorted(ret["q_ls"]) == ["q1", "q3"]
    assert sorted(ret["gt_ls"]) == ["a1", "a3"]


def test_shuffling_keeps_each_question_with_its_own_answer():
    load, _ = _load(COMPLETE)
    ret = load("data.jsonl", KEY_MAP, -1, is_shuffle=True)
    pairs = list(zip(ret["q_ls"], ret["gt_ls"]))
    assert sorted(pairs) == [("q1", "a1"), ("q2", "a2"), ("q3", "a3")]


def test_complete_records_are_unchanged():
    load, app = _load(COMPLETE)
    ret = load("data.jsonl", KEY_MAP, -1)
    assert ret == {"q_ls": ["q1", "q2", "q3"], "gt_ls": ["a1", "a2", "a3"]}
    assert app.logger.warnings == []


def test_limit_still_applies_after_skipping():
    load, _ = _load(RAGGED)
    ret = load("data.jsonl", KEY_MAP, 1)
    assert ret == {"q_ls": ["q1"], "gt_ls": ["a1"]}
