from __future__ import annotations

import asyncio
import inspect
import logging
import os
from contextlib import AbstractAsyncContextManager
from functools import partial
from pathlib import Path
from types import EllipsisType, SimpleNamespace
from typing import Any, Callable, List, Literal, Optional, Union

import yaml
from fastmcp import FastMCP
from fastmcp.client import Client
from fastmcp.prompts import Prompt
from fastmcp.server.auth.auth import OAuthProvider
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.tools.tool import Tool
from fastmcp.tools.tool_transform import ToolTransformConfig
from mcp.server.lowlevel.server import LifespanResultT
from mcp.types import AnyFunction, ToolAnnotations, TypeAlias

from ultrarag.mcp_logging import get_logger

NotSet = ...
NotSetT: TypeAlias = EllipsisType

DuplicateBehavior = Literal["warn", "error", "replace", "ignore"]
Transport = Literal["stdio", "http", "sse", "streamable-http"]


class UltraRAG_MCP_Server(FastMCP):
    """Extended FastMCP server with UltraRAG-specific features.

    Provides additional functionality for tool/prompt metadata tracking,
    configuration loading, and server.yaml generation.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        instructions: Optional[str] = None,
        *,
        version: Optional[str] = None,
        auth: Optional[OAuthProvider] = None,
        middleware: Optional[List[Middleware]] = None,
        lifespan: Optional[
            Callable[
                [FastMCP[LifespanResultT]],
                AbstractAsyncContextManager[LifespanResultT],
            ]
        ] = None,
        tool_serializer: Optional[Callable[[Any], str]] = None,
        on_duplicate_tools: Optional[DuplicateBehavior] = None,
        on_duplicate_resources: Optional[DuplicateBehavior] = None,
        on_duplicate_prompts: Optional[DuplicateBehavior] = None,
        resource_prefix_format: Optional[Literal["protocol", "path"]] = None,
        tool_transformations: Optional[dict[str, ToolTransformConfig]] = None,
        mask_error_details: Optional[bool] = None,
        tools: Optional[List[Union[Tool, Callable[..., Any]]]] = None,
        dependencies: Optional[List[str]] = None,
        include_tags: Optional[set[str]] = None,
        exclude_tags: Optional[set[str]] = None,
        include_fastmcp_meta: Optional[bool] = None,
        log_level: Optional[str] = None,
        debug: Optional[bool] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        sse_path: Optional[str] = None,
        message_path: Optional[str] = None,
        streamable_http_path: Optional[str] = None,
        json_response: Optional[bool] = None,
        stateless_http: Optional[bool] = None,
    ) -> None:
        """Initialize UltraRAG MCP server.

        Args:
            name: Server name (default: "UltraRAG")
            instructions: Server instructions
            version: Server version
            auth: OAuth provider for authentication
            middleware: List of middleware to apply
            lifespan: Lifespan context manager
            tool_serializer: Custom tool serializer function
            on_duplicate_tools: Behavior for duplicate tools
            on_duplicate_resources: Behavior for duplicate resources
            on_duplicate_prompts: Behavior for duplicate prompts
            resource_prefix_format: Resource prefix format
            tool_transformations: Tool transformation configurations
            mask_error_details: Whether to mask error details
            tools: List of tools to register
            dependencies: List of dependencies
            include_tags: Tags to include
            exclude_tags: Tags to exclude
            include_fastmcp_meta: Whether to include FastMCP metadata
            log_level: Logging level
            debug: Enable debug mode
            host: Server host
            port: Server port
            sse_path: SSE path
            message_path: Message path
            streamable_http_path: Streamable HTTP path
            json_response: Use JSON response format
            stateless_http: Enable stateless HTTP mode
        """
        name = name or "UltraRAG"
        level = os.environ.get("log_level", "warn")
        self.logger = get_logger(name, level)
        # FastMCP 3.x removed/renamed some __init__ parameters (e.g., resource_prefix_format)
        # Dynamically filter to keep only parameters supported by current version for compatibility
        super_kwargs = {
            "version": version,
            "auth": auth,
            "middleware": middleware,
            "lifespan": lifespan,
            "tool_serializer": tool_serializer,
            "on_duplicate_tools": on_duplicate_tools,
            "on_duplicate_resources": on_duplicate_resources,
            "on_duplicate_prompts": on_duplicate_prompts,
            "resource_prefix_format": resource_prefix_format,
            "tool_transformations": tool_transformations,
            "mask_error_details": mask_error_details,
            "tools": tools,
            "dependencies": dependencies,
            "include_tags": include_tags,
            "exclude_tags": exclude_tags,
            "include_fastmcp_meta": include_fastmcp_meta,
            "log_level": log_level,
            "debug": debug,
            "host": host,
            "port": port,
            "sse_path": sse_path,
            "message_path": message_path,
            "streamable_http_path": streamable_http_path,
            "json_response": json_response,
            "stateless_http": stateless_http,
        }

        init_params = inspect.signature(super().__init__).parameters
        filtered_kwargs = {}
        unsupported = []
        for key, value in super_kwargs.items():
            if key not in init_params:
                unsupported.append(key)
                continue
            # Avoid explicitly passing None to override FastMCP defaults
            if value is None and init_params[key].default is not inspect._empty:
                continue
            filtered_kwargs[key] = value

        if unsupported:
            self.logger.debug("Ignoring unsupported FastMCP args: %s", unsupported)

        super().__init__(name, instructions, **filtered_kwargs)
        self.output = {}
        self.fn_meta: dict[str, dict[str, Any]] = {}
        self.prompt_meta: dict[str, dict[str, Any]] = {}
        self.tool(self.build, name="build")

    def load_config(self, file_path: str) -> dict[str, Any]:
        """Load YAML configuration file.

        Args:
            file_path: Path to YAML configuration file

        Returns:
            Dictionary containing configuration (empty dict if file is empty)
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def tool(
        self,
        name_or_fn: Optional[Union[str, AnyFunction]] = None,
        *,
        output: Optional[str] = None,
        name: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        icons: Optional[List] = None,
        tags: Optional[set[str]] = None,
        output_schema: Union[dict[str, Any], None, NotSetT] = NotSet,
        annotations: Optional[Union[ToolAnnotations, dict[str, Any]]] = None,
        exclude_args: Optional[List[str]] = None,
        meta: Optional[dict[str, Any]] = None,
        enabled: Optional[bool] = None,
        **extra_kwargs: Any,
    ) -> Any:
        """Register a tool with UltraRAG-specific output annotation support.

        Args:
            name_or_fn: Tool name or function
            output: Output specification string (e.g., "input->output")
            name: Tool name
            title: Tool title
            description: Tool description
            icons: Tool icons
            tags: Tool tags
            output_schema: Output schema
            annotations: Tool annotations
            exclude_args: Arguments to exclude
            meta: Additional metadata
            enabled: Whether tool is enabled
            **extra_kwargs: Additional keyword arguments

        Returns:
            Tool registration result
        """
        if output is not None:
            if annotations is None:
                annotations = {"output": output}
            elif isinstance(annotations, dict):
                annotations = annotations | {"output": output}
            else:
                annotations.output = output

        tool_kwargs = {
            "name": name,
            "title": title,
            "output_schema": output_schema,
            "description": description,
            "icons": icons,
            "tags": tags,
            "annotations": annotations,
            "exclude_args": exclude_args,
            "meta": meta,
            "enabled": enabled,
            **extra_kwargs,
        }

        tool_params = inspect.signature(super().tool).parameters
        filtered_kwargs = {}
        unsupported = []
        for key, value in tool_kwargs.items():
            if key not in tool_params:
                unsupported.append(key)
                continue
            if value is None and tool_params[key].default is not inspect._empty:
                continue
            filtered_kwargs[key] = value

        if unsupported:
            self.logger.debug("Ignoring unsupported FastMCP.tool args: %s", unsupported)

        return super().tool(name_or_fn, **filtered_kwargs)

    def prompt(
        self,
        name_or_fn: Optional[Union[str, AnyFunction]] = None,
        *,
        output: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[set[str]] = None,
        enabled: Optional[bool] = None,
    ) -> Any:
        """Register a prompt with UltraRAG-specific output annotation support.

        Args:
            name_or_fn: Prompt name or function
            output: Output specification string
            name: Prompt name
            description: Prompt description
            tags: Prompt tags
            enabled: Whether prompt is enabled

        Returns:
            Prompt registration result or partial function for decorator usage

        Raises:
            ValueError: If classmethod is used as decorator
            TypeError: If invalid argument combination is provided
        """
        if isinstance(name_or_fn, classmethod):
            raise ValueError(
                inspect.cleandoc(
                    """
                    To decorate a classmethod, first define the method and then call
                    prompt() directly on the method instead of using it as a
                    decorator. See https://gofastmcp.com/patterns/decorating-methods
                    for examples and more information.
                    """
                )
            )

        # Determine the actual name and function based on the calling pattern
        if inspect.isroutine(name_or_fn):
            # Case 1: @prompt (without parens) - function passed directly as decorator
            # Case 2: direct call like prompt(fn, name="something")
            fn = name_or_fn
            prompt_name = name  # Use keyword name if provided, otherwise None

            if not hasattr(self, "_pending_output"):
                self._pending_output: dict[int, str] = {}
            self._pending_output[name or fn.__name__] = output

            # Register the prompt immediately
            prompt = Prompt.from_function(
                fn=fn,
                name=prompt_name,
                description=description,
                tags=tags,
                enabled=enabled,
            )
            self.add_prompt(prompt)

            return prompt

        elif isinstance(name_or_fn, str):
            # Case 3: @prompt("custom_name") - name passed as first argument
            if name is not None:
                raise TypeError(
                    "Cannot specify both a name as first argument and as keyword argument. "
                    f"Use either @prompt('{name_or_fn}') or @prompt(name='{name}'), not both."
                )
            prompt_name = name_or_fn
        elif name_or_fn is None:
            # Case 4: @prompt() or @prompt(name="something") - use keyword name
            prompt_name = name
        else:
            raise TypeError(
                f"First argument to @prompt must be a function, string, or None, got {type(name_or_fn)}"
            )

        # Return partial for cases where we need to wait for the function
        return partial(
            self.prompt,
            name=prompt_name,
            description=description,
            tags=tags,
            enabled=enabled,
            output=output,
        )

    def add_prompt(self, prompt: Prompt) -> None:
        """Add a prompt and track its metadata.

        Args:
            prompt: Prompt instance to add
        """
        fn = prompt.fn
        fn_name = fn.__name__

        sig = inspect.signature(fn)
        param_names = [
            p.name
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]
        output_val = self._pending_output.pop(fn_name, None)
        self.prompt_meta[prompt.name or fn_name] = {
            "fn_name": fn_name,
            "params": param_names,
            "output": output_val,
        }

        super().add_prompt(prompt)

    def add_tool(self, tool: Tool) -> None:
        """Add a tool and track its metadata.

        Args:
            tool: Tool instance to add
        """
        fn = tool.fn
        fn_name = fn.__name__
        if fn_name != "build":
            sig = inspect.signature(fn)
            param_names = [
                p.name
                for p in sig.parameters.values()
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            ]
            try:
                output = tool.annotations.output
            except (AttributeError, TypeError):
                output = None
            self.fn_meta[tool.name or fn_name] = {
                "fn_name": fn_name,
                "params": param_names,
                "output": output,
            }

        super().add_tool(tool)

    def _make_io_mapping(
        self, params: List[str], io_spec: Optional[str], param_cfg: dict[str, Any]
    ) -> dict[str, str]:
        """Create input/output mapping from parameter specification.

        Args:
            params: List of parameter names
            io_spec: Optional I/O specification string
            param_cfg: Parameter configuration dictionary

        Returns:
            Dictionary mapping parameter names to I/O specifications
        """
        if io_spec:
            in_specs = [p.strip() for p in io_spec.split(",")]
        else:
            in_specs = params

        mapping = {}
        for key, spec in zip(params, in_specs):
            if spec in param_cfg and not spec.startswith("$"):
                spec = "$" + spec
            mapping[key] = spec
        return mapping

    def _build_entry(
        self, meta: dict[str, Any], param_cfg: dict[str, Any]
    ) -> dict[str, Any]:
        """Build a tool/prompt entry for server.yaml.

        Args:
            meta: Metadata dictionary containing function info
            param_cfg: Parameter configuration dictionary

        Returns:
            Dictionary entry for server.yaml

        Raises:
            AssertionError: If output format is invalid
        """
        entry: dict[str, Any] = {}
        if meta["output"]:
            parts = [span.strip() for span in meta["output"].split("->")]
            assert len(parts) <= 2, f"Output format error: {meta['output']}"
            entry["input"] = self._make_io_mapping(
                meta["params"], parts[0] if len(parts) == 2 else None, param_cfg
            )
            if parts[-1].strip() and not parts[-1].strip().lower() == "none":
                entry["output"] = [
                    (
                        "$" + p.strip()
                        if p.strip() in param_cfg and not p.strip().startswith("$")
                        else p.strip()
                    )
                    for p in parts[-1].split(",")
                ]
        else:
            entry["input"] = self._make_io_mapping(meta["params"], None, param_cfg)
        return entry

    def build(self, parameter_file: str) -> None:
        """Build server.yaml configuration file from registered tools and prompts.

        Args:
            parameter_file: Path to parameter YAML file

        Raises:
            FileNotFoundError: If server code file does not exist
        """
        cfg_path = Path(parameter_file)
        base_dir = cfg_path.parent
        srv_name = base_dir.name
        self.param_cfg = self.load_config(str(cfg_path)) if cfg_path.exists() else {}
        out_path = base_dir / "server.yaml"
        build_yaml = {
            "path": self.param_cfg.get(
                "path", str(base_dir / "src" / f"{srv_name}.py")
            ),
            "parameter": parameter_file,
            "tools": {
                name: self._build_entry(self.fn_meta[name], self.param_cfg)
                for name in self.fn_meta
            },
            "prompts": {
                name: self._build_entry(self.prompt_meta[name], self.param_cfg)
                for name in self.prompt_meta
            },
        }

        if not Path(build_yaml["path"]).exists():
            raise FileNotFoundError(f"Server code not found: {build_yaml['path']}")

        yaml.safe_dump(
            build_yaml, out_path.open("w"), allow_unicode=True, sort_keys=False
        )

    def run(
        self,
        transport: Optional[Transport] = None,
        show_banner: bool = False,
        **transport_kwargs: Any,
    ) -> None:
        """Run the MCP server with specified transport.

        Args:
            transport: Transport type (default: None, uses FastMCP default)
            show_banner: Whether to show server banner
            **transport_kwargs: Additional transport-specific arguments
        """
        super().run(
            transport=transport,
            show_banner=show_banner,
            **transport_kwargs,
        )


# Suppress MCP library logging at WARNING level and below
logging.getLogger("mcp").setLevel(logging.WARNING)
