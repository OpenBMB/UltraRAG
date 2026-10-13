"""Tests for the ``output=`` I/O spec parsing in ``ultrarag.server``.

``UltraRAG_MCP_Server._make_io_mapping`` / ``_build_entry`` translate a tool's
``output="a,b->c"`` declaration into the ``input`` / ``output`` blocks of the
generated ``server.yaml``. That generated file is what the pipeline engine
resolves variables from, so a declaration that mangles silently surfaces much
later as a "missing variable in pipeline execution".

A malformed declaration used to be accepted silently: the input list was
zipped against the function parameters, and ``zip()`` truncates to the shorter
side, so a typo that dropped one name from the input list simply omitted the
trailing parameters from the mapping instead of failing the build.
"""

import pytest

from ultrarag.server import UltraRAG_MCP_Server

# _build_entry / _make_io_mapping are pure helpers that never touch instance
# state, so build the instance without running the FastMCP __init__.
SERVER = UltraRAG_MCP_Server.__new__(UltraRAG_MCP_Server)


def build(output, params, param_cfg=None):
    meta = {"output": output, "params": params}
    return SERVER._build_entry(meta, param_cfg or {})


def test_canonical_spec_maps_inputs_and_outputs():
    assert build("a,b->c,d", ["a", "b"]) == {
        "input": {"a": "a", "b": "b"},
        "output": ["c", "d"],
    }


def test_parameter_backed_names_get_dollar_prefix_on_both_sides():
    # A name that exists in the server's parameter.yaml is read from there,
    # which the ``$`` prefix signals to the runtime.
    assert build("a,p->p", ["a", "p"], {"p": 1}) == {
        "input": {"a": "a", "p": "$p"},
        "output": ["$p"],
    }


def test_already_prefixed_names_are_not_prefixed_twice():
    assert build("$a->b", ["a"], {"a": 1}) == {
        "input": {"a": "$a"},
        "output": ["b"],
    }


def test_none_output_declares_inputs_only():
    ret = build("a,b->None", ["a", "b"])
    assert ret == {"input": {"a": "a", "b": "b"}}
    assert "output" not in ret


def test_missing_output_spec_falls_back_to_parameter_names():
    assert build(None, ["a", "b"]) == {"input": {"a": "a", "b": "b"}}


def test_fewer_declared_inputs_than_parameters_raises():
    # Before the fix this returned input={"a": "a"} and dropped "b" entirely.
    with pytest.raises(ValueError, match="Input count mismatch"):
        build("a->out", ["a", "b"])


def test_more_declared_inputs_than_parameters_raises():
    with pytest.raises(ValueError, match="Input count mismatch"):
        build("a,b,c->out", ["a", "b"])


def test_more_than_one_arrow_raises():
    with pytest.raises(ValueError, match="Output format error"):
        build("a->b->c", ["a"])


def test_empty_output_name_raises():
    # A trailing comma produces an empty name that can never match a returned
    # key, so the tool would silently lose that output at runtime.
    with pytest.raises(ValueError, match="Output format error"):
        build("a->b,", ["a"])
