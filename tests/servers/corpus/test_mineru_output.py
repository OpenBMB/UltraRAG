"""MinerU output folders must not silently produce an empty corpus."""

import asyncio
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from ultrarag.errors import ToolError
from ultrarag.server import local_server_mode


def _load_corpus() -> ModuleType:
    source = Path(__file__).resolve().parents[3] / "servers/corpus/src/corpus.py"
    spec = importlib.util.spec_from_file_location("corpus_output_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with local_server_mode():
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("method", ["auto", "txt", "ocr", "vlm"])
def test_reads_selected_mineru_method(tmp_path: Path, method: str) -> None:
    corpus = _load_corpus()
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    parsed = tmp_path / "parsed/paper" / method
    parsed.mkdir(parents=True)
    (parsed / "paper.md").write_text("Parsed document text.", encoding="utf-8")
    text_output = tmp_path / "text.jsonl"
    asyncio.run(
        corpus.build_mineru_corpus(
            str(tmp_path / "parsed"),
            str(pdf),
            str(text_output),
            str(tmp_path / "image.jsonl"),
        )
    )
    record = json.loads(text_output.read_text(encoding="utf-8"))
    assert record["contents"] == "Parsed document text."


def test_missing_mineru_output_fails_without_writing_empty_corpus(
    tmp_path: Path,
) -> None:
    corpus = _load_corpus()
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    text_output = tmp_path / "text.jsonl"
    with pytest.raises(ToolError, match="No corpus records"):
        asyncio.run(
            corpus.build_mineru_corpus(
                str(parsed), str(pdf), str(text_output), str(tmp_path / "image.jsonl")
            )
        )
    assert not text_output.exists()


def test_image_only_mineru_output_is_preserved(tmp_path: Path) -> None:
    from PIL import Image

    corpus = _load_corpus()
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    images = tmp_path / "parsed/paper/vlm/images"
    images.mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(images / "figure.png")
    image_output = tmp_path / "image.jsonl"
    asyncio.run(
        corpus.build_mineru_corpus(
            str(tmp_path / "parsed"),
            str(pdf),
            str(tmp_path / "text.jsonl"),
            str(image_output),
        )
    )
    record = json.loads(image_output.read_text(encoding="utf-8"))
    assert (tmp_path / record["image_path"]).is_file()


def test_blank_markdown_is_not_a_corpus_record(tmp_path: Path) -> None:
    corpus = _load_corpus()
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    parsed = tmp_path / "parsed/paper/auto"
    parsed.mkdir(parents=True)
    (parsed / "paper.md").write_text(" \n", encoding="utf-8")
    with pytest.raises(ToolError, match="No corpus records"):
        asyncio.run(
            corpus.build_mineru_corpus(
                str(tmp_path / "parsed"),
                str(pdf),
                str(tmp_path / "text.jsonl"),
                str(tmp_path / "image.jsonl"),
            )
        )
