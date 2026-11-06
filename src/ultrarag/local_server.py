"""Transport-independent registration and config generation for local servers."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from ultrarag.mcp_logging import get_logger
from ultrarag.registration import Registration, build_io_entry


def _reject_behavior_options(options: dict[str, Any], allowed: set[str]) -> None:
    unsupported = []
    for key, value in options.items():
        if key in allowed or value is Ellipsis:
            continue
        if key != "output_schema" and (
            value is None
            or (isinstance(value, (str, list, tuple, dict, set)) and not value)
        ):
            continue
        unsupported.append(key)
    if unsupported:
        raise ValueError(
            "--no-mcp does not support MCP-specific options: "
            + ", ".join(unsupported)
            + ". Use MCP mode for these extensions."
        )


class LocalServer:
    def __init__(self, name: str = "UltraRAG", *args: Any, **kwargs: Any) -> None:
        _reject_behavior_options(
            kwargs,
            {
                "log_level",
                "lifespan",
                "tools",
                "instructions",
                "version",
                "host",
                "port",
                "sse_path",
                "message_path",
                "streamable_http_path",
                "json_response",
                "stateless_http",
                "debug",
                "include_fastmcp_meta",
            },
        )
        self.name = name
        self.logger = get_logger(name, kwargs.get("log_level") or "warn")
        self.tools: dict[str, Registration] = {}
        self.prompts: dict[str, Registration] = {}
        self.lifespan = kwargs.get("lifespan")
        for tool in kwargs.get("tools") or []:
            if callable(tool):
                self.tool(tool)
            else:
                self.add_tool(tool)

    def _register(
        self,
        registry: dict[str, Registration],
        name_or_fn: str | Callable[..., Any] | None,
        *,
        name: str | None,
        output: str | None,
        enabled: bool = True,
    ) -> Any:
        if isinstance(name_or_fn, str):
            name = name_or_fn
            name_or_fn = None

        def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
            registry[name or fn.__name__] = Registration(
                name or fn.__name__, fn, output, enabled
            )
            return fn

        return decorate(name_or_fn) if callable(name_or_fn) else decorate

    def tool(
        self,
        name_or_fn: str | Callable[..., Any] | None = None,
        *,
        output: str | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> Any:
        _reject_behavior_options(
            kwargs,
            {"enabled", "description", "title", "icons", "tags", "annotations", "meta"},
        )
        if output is None:
            annotations = kwargs.get("annotations")
            output = (
                annotations.get("output")
                if isinstance(annotations, dict)
                else getattr(annotations, "output", None)
            )
        return self._register(
            self.tools,
            name_or_fn,
            name=name,
            output=output,
            enabled=kwargs.get("enabled") is not False,
        )

    def prompt(
        self,
        name_or_fn: str | Callable[..., Any] | None = None,
        *,
        output: str | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> Any:
        _reject_behavior_options(
            kwargs, {"enabled", "description", "title", "icons", "tags", "meta"}
        )
        return self._register(
            self.prompts,
            name_or_fn,
            name=name,
            output=output,
            enabled=kwargs.get("enabled") is not False,
        )

    def add_tool(self, tool: Any) -> Any:
        self.tool(tool.fn, name=tool.name, output=getattr(tool, "output", None))
        return tool

    def add_prompt(self, prompt: Any) -> Any:
        self.prompt(prompt.fn, name=prompt.name, output=getattr(prompt, "output", None))
        return prompt

    @staticmethod
    def _entry(
        registration: Registration, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        args = [
            p.name
            for p in inspect.signature(registration.fn).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        ]
        return build_io_entry(args, registration.output, parameters)

    def build(self, parameter_file: str) -> None:
        path = Path(parameter_file)
        parameters = (
            yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if path.exists()
            else {}
        )
        server_path = parameters.get(
            "path", str(path.parent / "src" / f"{path.parent.name}.py")
        )
        if not Path(server_path).exists():
            raise FileNotFoundError(f"Server code not found: {server_path}")
        data = {
            "path": server_path,
            "parameter": parameter_file,
            "tools": {
                name: self._entry(reg, parameters) for name, reg in self.tools.items()
            },
            "prompts": {
                name: self._entry(reg, parameters) for name, reg in self.prompts.items()
            },
        }
        with (path.parent / "server.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, allow_unicode=True, sort_keys=False)

    def run(self, *args: Any, **kwargs: Any) -> None:
        # Main-guarded server registration can be evaluated locally; starting a
        # transport is intentionally a no-op in that case.
        return None
