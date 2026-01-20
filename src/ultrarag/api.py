import asyncio
import os
from types import SimpleNamespace
from typing import List, Optional

import yaml
from fastmcp import Client

from .mcp_logging import get_logger
from . import client as _client_mod

_client: Optional[Client] = None
_servers: Optional[List[str]] = None
SERVER_ROOT = ""
logger = None


class _CallWrapper:
    """Wraps a MCP tool so it can be called like a normal Python function."""

    def __init__(self, client: Client, server: str, tool: str, multi: bool):
        self._client = client
        self._server = server
        self._tool = tool
        self._multi = multi

    async def _ensure_client(self):
        global _client, logger
        if _client is None:
            raise RuntimeError(
                "[UltraRAG Error] ToolCall was used before `initialize()` was called."
            )
        try:
            _ = _client.session
        except RuntimeError:
            await _client.__aenter__()
            tools = await _client.list_tools()
            tool_name_lst = [
                tool.name
                for tool in tools
                if not tool.name.endswith("_build" if "_" in tool.name else "build")
            ]
            logger.info(f"Available tools: {tool_name_lst}")

    async def _async_call(self, *args, **kwargs):
        global _client, SERVER_ROOT
        await self._ensure_client()
        concated = f"{self._server}_{self._tool}" if self._multi else self._tool
        param_file = os.path.join(SERVER_ROOT, self._server, "parameter.yaml")
        if os.path.exists(param_file):
            with open(param_file, "r") as f:
                parameter = yaml.safe_load(f)
        else:
            parameter = {}

        with open(os.path.join(SERVER_ROOT, self._server, "server.yaml"), "r") as f:
            try:
                input_param = yaml.safe_load(f)["tools"][self._tool]["input"]
                input_keys = list(input_param.keys())
            except (KeyError, TypeError) as e:
                raise ValueError(
                    f"[UltraRAG Error] Tool {self._tool} not found in server {self._server} configuration!"
                ) from e

        for k, v in list(input_param.items()):
            if isinstance(v, str) and v.startswith("$"):
                key = v[1:]
                if key not in parameter:
                    continue
                input_param[k] = parameter[key]
            else:
                input_param[k] = None

        if len(args) > len(input_param):
            raise ValueError(
                f"[UltraRAG Error] Expected at most {len(input_param)} positional args, got {len(args)}"
            )
        for pos, value in enumerate(args):
            key = input_keys[pos]
            input_param[key] = value

        for k, v in kwargs.items():
            if k not in input_param:
                raise ValueError(f"[UltraRAG Error] Unexpected keyword arg: {k!r}")
            input_param[k] = v

        missing = [k for k, v in input_param.items() if v is None]
        if missing:
            raise ValueError(f"[UltraRAG Error] Missing value for key(s): {missing}")
        result = await self._client.call_tool(concated, input_param)
        return result.data if result else None

    def __call__(self, *args, **kwargs):
        """Call the tool synchronously or asynchronously based on event loop state.

        Args:
            *args: Positional arguments for the tool
            **kwargs: Keyword arguments for the tool

        Returns:
            Tool result (synchronous) or Task (asynchronous if loop is running)
        """
        loop = asyncio.get_event_loop_policy().get_event_loop()
        if loop.is_running():
            return loop.create_task(self._async_call(*args, **kwargs))
        else:
            return loop.run_until_complete(self._async_call(*args, **kwargs))


class _ServerProxy(SimpleNamespace):
    """Proxy for a specific server, e.g. `ToolCall.retriever`.

    Accessing an attribute on this object (e.g. `.retriever_search`) returns a
    `_CallWrapper` bound to that (server, tool) pair.
    """

    def __init__(self, client: Client, name: str, multi: bool):
        """Initialize server proxy.

        Args:
            client: MCP client instance
            name: Server name
            multi: Whether multiple servers are configured
        """
        super().__init__()
        self._client = client
        self._name = name
        self._multi = multi

    def __getattr__(self, tool_name: str):
        """Get a callable wrapper for the specified tool.

        Args:
            tool_name: Name of the tool to access

        Returns:
            _CallWrapper instance for the tool
        """
        return _CallWrapper(self._client, self._name, tool_name, self._multi)


class _Router(SimpleNamespace):
    """Top-level router for ToolCall.

    Example:
        ToolCall.retriever.retriever_search(...)
        ToolCall.benchmark.get_data(...)
    """

    def __getattr__(self, server: str):
        """Get a proxy for the specified server.

        Args:
            server: Name of the server to access

        Returns:
            _ServerProxy instance for the server

        Raises:
            AttributeError: If server has not been initialized
        """
        global _client, _servers
        if _servers is None or server not in _servers:
            raise AttributeError(f"Server {server} has not been initialized!")
        return _ServerProxy(_client, server, len(_servers) > 1)


def initialize(servers: List[str], server_root: str, log_level: str = "info") -> None:
    """Initialize MCP servers so they can be accessed via ToolCall.

    Args:
        servers: List of server names to initialize
        server_root: Root directory containing server directories
        log_level: Logging level (default: "info")

    Raises:
        ValueError: If server path does not exist
    """
    global _client, _servers, SERVER_ROOT, logger
    logger = get_logger("Client", log_level)
    SERVER_ROOT = server_root
    mcp_cfg = {"mcpServers": {}}
    for server_name in servers:
        path = os.path.join(server_root, server_name, "src", f"{server_name}.py")
        if not os.path.exists(path):
            raise ValueError(f"Server path {path} does not exist!")
        mcp_cfg["mcpServers"][server_name] = {
            "command": "python",
            "args": [path],
            "env": os.environ.copy(),
        }

    _client = Client(mcp_cfg)
    _servers = servers


ToolCall = _Router()


async def _pipeline_async(
    pipeline_file: str,
    parameter_file: str,
    log_level: str = "error",
):
    """Internal async helper that runs a full UltraRAG pipeline.

    Args:
        pipeline_file: Path to pipeline YAML file
        parameter_file: Path to parameter YAML file
        log_level: Logging level (default: "error")

    Returns:
        Pipeline execution results
    """
    _client_mod.logger = get_logger("Client", log_level)

    return await _client_mod.run(pipeline_file, parameter_file, return_all=True)


def PipelineCall(
    pipeline_file: str,
    parameter_file: str,
    log_level: str = "error",
):
    """Run a full UltraRAG pipeline from Python.

    Similar to `ultrarag run`, but with an explicitly provided parameter file.
    Can be called synchronously or asynchronously based on event loop state.

    Args:
        pipeline_file: Path to pipeline YAML file
        parameter_file: Path to parameter YAML file
        log_level: Logging level (default: "error")

    Returns:
        Pipeline execution results (synchronous) or Task (asynchronous if loop is running)
    """
    loop = asyncio.get_event_loop_policy().get_event_loop()
    if loop.is_running():
        return loop.create_task(
            _pipeline_async(pipeline_file, parameter_file, log_level)
        )
    else:
        return loop.run_until_complete(
            _pipeline_async(pipeline_file, parameter_file, log_level)
        )
