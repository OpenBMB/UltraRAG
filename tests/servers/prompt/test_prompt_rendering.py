"""Prompt values reach the model as plain text, not HTML."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from jinja2.exceptions import SecurityError

from ultrarag.server import local_server_mode

ROOT = Path(__file__).resolve().parents[3]
TEMPLATES = ROOT / "prompt"

QUESTION = 'What does the "R&D" team\'s report say about x < y?'
PASSAGES = [
    'The R&D team\'s report says x < y holds in "strict" mode.',
    "Budgets > $5k need sign-off.",
]


@pytest.fixture(scope="module")
def prompt_server() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "prompt_rendering_test", ROOT / "servers/prompt/src/prompt.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with local_server_mode():
        spec.loader.exec_module(module)
    return module


def test_question_and_passages_are_rendered_verbatim(
    prompt_server: ModuleType,
) -> None:
    [prompt] = prompt_server.qa_rag_boxed(
        [QUESTION], [PASSAGES], TEMPLATES / "qa_rag_boxed.jinja"
    )
    assert f"Question: {QUESTION}" in prompt
    assert "\n".join(PASSAGES) in prompt


@pytest.mark.parametrize(
    ("tool", "init_template", "append_template", "answer"),
    [
        (
            "search_r1_gen",
            "search_r1_init.jinja",
            "search_r1_append.jinja",
            "<think>I need the report.</think>\n<search> R&D team's report on x < y",
        ),
        (
            "r1_searcher_gen",
            "r1_searcher_init.jinja",
            "r1_searcher_append.jinja",
            "I need the report. <|begin_of_query|> R&D team's report on x < y",
        ),
    ],
    ids=["search_r1", "r1_searcher"],
)
def test_each_turn_extends_the_history_unchanged(
    prompt_server: ModuleType,
    tool: str,
    init_template: str,
    append_template: str,
    answer: str,
) -> None:
    [history] = prompt_server.qa_boxed([QUESTION], TEMPLATES / init_template)
    for _ in range(3):
        [prompt] = getattr(prompt_server, tool)(
            [history], [answer], [PASSAGES], TEMPLATES / append_template
        )
        assert prompt.startswith(history)
        assert answer in prompt
        history = prompt


def test_template_source_still_renders_in_the_sandbox(
    prompt_server: ModuleType, tmp_path: Path
) -> None:
    template = tmp_path / "unsafe.jinja"
    template.write_text("{{ question.__class__.__mro__ }}", encoding="utf-8")
    with pytest.raises(SecurityError):
        prompt_server.qa_boxed(["q"], template)
