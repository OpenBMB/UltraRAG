"""Tests for evaluation-metric normalization in the ``evaluation`` MCP server.

The ``evaluation`` server lives under ``servers/evaluation/src`` rather than
inside the installable ``ultrarag`` package. Importing the module directly would
pull in the MCP app and the rouge scorer, so the pure functions under test are
lifted out of the source with ``ast`` instead.
"""

import ast
import re
import string
from pathlib import Path
from typing import List

EVALUATION_SRC = (
    Path(__file__).resolve().parents[3] / "servers" / "evaluation" / "src" / "evaluation.py"
)

_WANTED = {
    "normalize_text",
    "accuracy_score",
    "exact_match_score",
    "cover_exact_match_score",
}


def _load():
    tree = ast.parse(EVALUATION_SRC.read_text())
    module = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in _WANTED
        ],
        type_ignores=[],
    )
    namespace = {"re": re, "string": string, "List": List}
    exec(compile(module, str(EVALUATION_SRC), "exec"), namespace)
    return namespace


_NS = _load()
normalize_text = _NS["normalize_text"]
accuracy_score = _NS["accuracy_score"]
exact_match_score = _NS["exact_match_score"]
cover_exact_match_score = _NS["cover_exact_match_score"]


def test_single_letter_answer_survives_normalization():
    # qa_boxed_multiple_choice labels options with string.ascii_uppercase, so
    # "A" is a real gold value. Stripping it to "" makes it match everything.
    assert normalize_text("A") == "a"


def test_article_only_answer_survives_normalization():
    assert normalize_text("the") == "the"


def test_accuracy_rejects_a_wrong_multiple_choice_answer():
    assert accuracy_score(["A"], "D") == 0.0


def test_accuracy_accepts_a_correct_multiple_choice_answer():
    assert accuracy_score(["A"], "A") == 1.0


def test_cover_exact_match_rejects_a_wrong_multiple_choice_answer():
    assert cover_exact_match_score(["A"], "D") == 0.0


def test_cover_exact_match_accepts_a_correct_multiple_choice_answer():
    # Regression guard: passes with and without the fix, because an empty
    # token list also matched. It pins the behaviour the fix must not break.
    assert cover_exact_match_score(["A"], "A") == 1.0


def test_articles_are_still_stripped_inside_a_longer_answer():
    # Regression guard: passes with and without the fix.
    assert normalize_text("The Beatles") == "beatles"
    assert accuracy_score(["the beatles"], "Beatles") == 1.0


def test_unaffected_option_letters_are_unchanged():
    # Regression guard: passes with and without the fix.
    assert accuracy_score(["B"], "D") == 0.0
    assert accuracy_score(["B"], "B") == 1.0
