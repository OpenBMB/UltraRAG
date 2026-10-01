"""Common execution client for direct Python calls and remote MCP services."""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import threading
import tokenize
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ultrarag.local_server import LocalServer
from ultrarag.server import local_server_mode

_path_lock = threading.Lock()
_path_users: dict[str, tuple[int, bool]] = {}


def create_mcp_client(mcp_cfg: dict[str, Any]) -> Any:
    try:
        from fastmcp import Client
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == "fastmcp" or exc.name.startswith("mcp")):
            raise RuntimeError(
                "MCP execution requires the optional dependency: "
                "pip install 'ultrarag[mcp]'"
            ) from exc
        raise
    return Client(mcp_cfg)


class LocalExecutionClient:
    """Client-shaped adapter that imports local registrars once per session."""

    def __init__(self, mcp_cfg: dict[str, Any], server_cfg: dict[str, Any]) -> None:
        self.server_cfg = server_cfg
        self.local: dict[str, Any] = {}
        self._module_names: list[str] = []
        self._source_paths: set[str] = set()
        remote_names = [
            name
            for name, cfg in server_cfg.items()
            if str(cfg.get("path", "")).startswith(("http://", "https://"))
        ]
        self.remote_names = set(remote_names)
        remote_cfg = {
            "mcpServers": {name: mcp_cfg["mcpServers"][name] for name in remote_names}
        }
        self.remote = create_mcp_client(remote_cfg) if remote_names else None
        self._entered = False

    def _load(self, name: str, path: str) -> Any:
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Local server file not found: {source}")
        source_dir = str(source.parent)
        if source_dir not in self._source_paths:
            with _path_lock:
                count, added = _path_users.get(source_dir, (0, False))
                if count == 0 and source_dir not in sys.path:
                    sys.path.append(source_dir)
                    added = True
                _path_users[source_dir] = (count + 1, added)
            self._source_paths.add(source_dir)

        def import_source(as_main: bool = False) -> Any:
            module_name = f"_ultrarag_local_{name}_{uuid.uuid4().hex}"
            spec = importlib.util.spec_from_file_location(module_name, source)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load local server: {source}")
            module = importlib.util.module_from_spec(spec)
            if as_main:
                module.__name__ = "__main__"
            sys.modules[module_name] = module
            self._module_names.append(module_name)
            try:
                with local_server_mode():
                    if as_main:
                        with tokenize.open(source) as stream:
                            code = compile(stream.read(), str(source), "exec")
                        exec(code, module.__dict__)
                    else:
                        spec.loader.exec_module(module)
            except ModuleNotFoundError as exc:
                if exc.name in {"fastmcp", "mcp"}:
                    raise RuntimeError(
                        f"Local server '{name}' imports MCP directly. "
                        "Register it through UltraRAG_MCP_Server for --no-mcp."
                    ) from exc
                raise
            return module

        module = import_source()
        app = getattr(module, "app", None)
        if not isinstance(app, LocalServer):
            raise ValueError(
                f"Local server '{name}' must expose "
                "an UltraRAG_MCP_Server instance as app"
            )
        if not app.tools and not app.prompts:
            # Older class-bound servers register their provider inside the main guard.
            module = import_source(as_main=True)
            app = module.app
        return app

    async def __aenter__(self) -> "LocalExecutionClient":
        try:
            for name, cfg in self.server_cfg.items():
                if name not in self.remote_names:
                    self.local[name] = self._load(name, str(cfg["path"]))
            if self.remote is not None:
                await self.remote.__aenter__()
            self._entered = True
            return self
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.remote is not None and self._entered:
            await self.remote.__aexit__(exc_type, exc, tb)
        self.local.clear()
        for name in self._module_names:
            sys.modules.pop(name, None)
        self._module_names.clear()
        with _path_lock:
            for source_dir in self._source_paths:
                count, added = _path_users[source_dir]
                if count == 1:
                    _path_users.pop(source_dir)
                    if added and source_dir in sys.path:
                        sys.path.remove(source_dir)
                else:
                    _path_users[source_dir] = (count - 1, added)
        self._source_paths.clear()
        self._entered = False

    def _split(self, full_name: str) -> tuple[str, str]:
        names = list(self.server_cfg)
        if len(names) == 1:
            return names[0], full_name
        for name in sorted(names, key=len, reverse=True):
            prefix = f"{name}_"
            if full_name.startswith(prefix):
                return name, full_name[len(prefix) :]
        raise ValueError(f"Unknown tool or prompt: {full_name}")

    def _remote_name(self, alias: str, tool: str) -> str:
        return tool if len(self.remote_names) == 1 else f"{alias}_{tool}"

    async def call_tool(self, full_name: str, arguments: dict[str, Any]) -> Any:
        alias, tool = self._split(full_name)
        if alias in self.remote_names:
            return await self.remote.call_tool(
                self._remote_name(alias, tool), arguments
            )
        app = self.local[alias]
        if tool == "build":
            value = app.build(**arguments)
        else:
            registration = app.tools.get(tool)
            if registration is None:
                raise ValueError(f"Unknown local tool: {alias}.{tool}")
            value = registration.fn(**arguments)
        if inspect.isawaitable(value):
            value = await value
        serialized = json.dumps(value, ensure_ascii=False, default=str)
        return SimpleNamespace(data=value, content=[SimpleNamespace(text=serialized)])

    async def get_prompt(self, full_name: str, arguments: dict[str, Any]) -> Any:
        alias, prompt = self._split(full_name)
        if alias in self.remote_names:
            return await self.remote.get_prompt(
                self._remote_name(alias, prompt), arguments
            )
        registration = self.local[alias].prompts.get(prompt)
        if registration is None:
            raise ValueError(f"Unknown local prompt: {alias}.{prompt}")
        value = registration.fn(**arguments)
        if inspect.isawaitable(value):
            value = await value
        if not isinstance(value, list):
            value = [value]
        messages = [
            item
            if hasattr(item, "content") and hasattr(item.content, "text")
            else SimpleNamespace(content=SimpleNamespace(text=str(item)))
            for item in value
        ]
        return SimpleNamespace(messages=messages)

    async def list_tools(self) -> list[Any]:
        tools: list[Any] = []
        multiple = len(self.server_cfg) > 1
        for alias, app in self.local.items():
            for name in ["build", *app.tools]:
                tools.append(
                    SimpleNamespace(name=f"{alias}_{name}" if multiple else name)
                )
        if self.remote is not None:
            remote_tools = await self.remote.list_tools()
            for tool in remote_tools:
                if len(self.remote_names) == 1:
                    alias = next(iter(self.remote_names))
                    name = f"{alias}_{tool.name}" if multiple else tool.name
                else:
                    name = tool.name
                tools.append(SimpleNamespace(name=name))
        return tools
