import json
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


def test_citation_state_is_isolated_and_survives_json_round_trips() -> None:
    custom = _load_custom_module()
    state_a = custom.init_citation_registry(["request-a"])["citation_state"]
    first_a = custom.assign_citation_ids_stateful([["doc-a"]], state_a)
    state_b = custom.init_citation_registry(["request-b"])["citation_state"]
    first_b = custom.assign_citation_ids_stateful([["doc-b"]], state_b)
    continued_a = custom.assign_citation_ids_stateful(
        [[" doc-a ", "doc-c"]], json.loads(json.dumps(first_a["citation_state"]))
    )
    assert first_a["ret_psg"] == [["[1] doc-a"]]
    assert first_b["ret_psg"] == [["[1] doc-b"]]
    assert continued_a["ret_psg"] == [["[1] doc-a", "[2] doc-c"]]
    assert state_a == state_b == [{"registry": {}, "counter": 0}]
    assert first_a["citation_state"][0]["counter"] == 1
    assert not hasattr(custom, "CitationRegistry")


def test_queries_keep_independent_counters_and_empty_passages() -> None:
    custom = _load_custom_module()
    initial = custom.init_citation_registry(["a", "b"])["citation_state"]
    first = custom.assign_citation_ids_stateful([["A", "B"], []], initial)
    second = custom.assign_citation_ids_stateful(
        [["B"], ["B"]], first["citation_state"]
    )
    assert second["ret_psg"] == [["[2] B"], ["[1] B"]]
    with pytest.raises(ValueError, match="same queries"):
        custom.assign_citation_ids_stateful([["A"]], initial)
