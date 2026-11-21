"""Ensure prompt server template loading is restricted to the workspace prompt/ directory."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from ultrarag.server import local_server_mode

ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = (ROOT / "prompt").resolve()


@pytest.fixture(scope="module")
def prompt_server() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "prompt_path_validation_test", ROOT / "servers/prompt/src/prompt.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with local_server_mode():
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "valid_path",
    [
        "prompt/qa_boxed.jinja",
        "qa_boxed.jinja",
        PROMPTS_DIR / "qa_boxed.jinja",
        str(PROMPTS_DIR / "qa_boxed.jinja"),
    ],
)
def test_validate_template_path_accepts_workspace_prompts(
    prompt_server: ModuleType, valid_path: str | Path
) -> None:
    resolved = prompt_server._validate_template_path(valid_path)
    assert resolved == PROMPTS_DIR / "qa_boxed.jinja"
    template = prompt_server.load_prompt_template(valid_path)
    assert template is not None


@pytest.mark.parametrize(
    "invalid_path",
    [
        "/etc/passwd",
        "/etc/hosts",
        "../pyproject.toml",
        "prompt/../pyproject.toml",
        str(ROOT / "pyproject.toml"),
        str(ROOT / "README.md"),
    ],
)
def test_validate_template_path_rejects_outside_or_non_jinja_paths(
    prompt_server: ModuleType, invalid_path: str
) -> None:
    with pytest.raises(ValueError):
        prompt_server._validate_template_path(invalid_path)
    with pytest.raises(ValueError):
        prompt_server.load_prompt_template(invalid_path)


def test_validate_template_path_rejects_symlinks_escaping_prompt_dir(
    prompt_server: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_prompts = tmp_path / "prompt"
    fake_prompts.mkdir()
    outside = tmp_path / "outside.jinja"
    outside.write_text("secret", encoding="utf-8")
    symlink = fake_prompts / "escape.jinja"
    symlink.symlink_to(outside)

    monkeypatch.setattr(prompt_server, "_PROMPTS_DIR", fake_prompts.resolve())
    with pytest.raises(ValueError):
        prompt_server._validate_template_path("prompt/escape.jinja")
    with pytest.raises(ValueError):
        prompt_server._validate_template_path(symlink)
