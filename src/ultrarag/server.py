"""Server registration entry point with an optional MCP transport."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_local_mode: ContextVar[bool] = ContextVar("ultrarag_local_server_mode", default=False)


@contextmanager
def local_server_mode() -> Iterator[None]:
    """Select the dependency-free registrar while loading a local server."""
    token = _local_mode.set(True)
    try:
        yield
    finally:
        _local_mode.reset(token)


def __getattr__(name: str):
    """Resolve the public server class for the current import context."""
    if name != "UltraRAG_MCP_Server":
        raise AttributeError(name)
    if _local_mode.get():
        from ultrarag.local_server import LocalServer

        return LocalServer
    try:
        from ultrarag.mcp_server import UltraRAG_MCP_Server as MCPServer
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == "fastmcp" or exc.name.startswith("mcp")):
            raise RuntimeError(
                "MCP execution requires the optional dependency: "
                "pip install 'ultrarag[mcp]'"
            ) from exc
        raise
    return MCPServer
