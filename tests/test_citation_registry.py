from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

CUSTOM_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "servers" / "custom" / "src" / "custom.py"
)


def _load_custom_module():
    spec = spec_from_file_location("ultrarag_custom", CUSTOM_MODULE_PATH)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_citation_registries_are_isolated_between_pipeline_runs() -> None:
    custom = _load_custom_module()

    registry_a = custom.init_citation_registry(["request-a"])["citation_registry_id"]
    first_a = custom.assign_citation_ids_stateful([["doc-a"]], registry_a)

    registry_b = custom.init_citation_registry(["request-b"])["citation_registry_id"]
    first_b = custom.assign_citation_ids_stateful([["doc-b"]], registry_b)
    continued_a = custom.assign_citation_ids_stateful(
        [["doc-a", "doc-c"]],
        registry_a,
    )

    assert first_a["ret_psg"] == [["[1] doc-a"]]
    assert first_b["ret_psg"] == [["[1] doc-b"]]
    assert continued_a["ret_psg"] == [["[1] doc-a", "[2] doc-c"]]

    custom.clear_citation_registry(registry_a)
    custom.clear_citation_registry(registry_b)
    with pytest.raises(ValueError, match="Unknown citation registry"):
        custom.assign_citation_ids_stateful([["doc-a"]], registry_a)
