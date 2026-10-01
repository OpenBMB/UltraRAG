"""I/O metadata shared by MCP and direct local server registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Registration:
    name: str
    fn: Callable[..., Any]
    output: str | None


def make_io_mapping(
    params: list[str], io_spec: str | None, parameter_config: dict[str, Any]
) -> dict[str, str]:
    specs = [part.strip() for part in io_spec.split(",")] if io_spec else params
    mapping = {}
    for key, spec in zip(params, specs):
        mapping[key] = (
            f"${spec}"
            if spec in parameter_config and not spec.startswith("$")
            else spec
        )
    return mapping


def build_io_entry(
    params: list[str], output: str | None, parameter_config: dict[str, Any]
) -> dict[str, Any]:
    if not output:
        return {"input": make_io_mapping(params, None, parameter_config)}
    parts = [part.strip() for part in output.split("->")]
    if len(parts) > 2:
        raise ValueError(f"Output format error: {output}")
    entry: dict[str, Any] = {
        "input": make_io_mapping(
            params, parts[0] if len(parts) == 2 else None, parameter_config
        )
    }
    if parts[-1] and parts[-1].lower() != "none":
        entry["output"] = [
            f"${part.strip()}"
            if part.strip() in parameter_config and not part.strip().startswith("$")
            else part.strip()
            for part in parts[-1].split(",")
        ]
    return entry
