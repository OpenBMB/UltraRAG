import argparse
import asyncio
import contextvars
import copy
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

import yaml
from dotenv import load_dotenv
from fastmcp import Client

from ultrarag.cli import log_server_banner
from ultrarag.mcp_exceptions import (
    NodeNotInstalledError,
    NodeVersionTooLowError,
    check_node_version,
)
from ultrarag.mcp_logging import get_logger

log_level = ""
logger = None
PipelineStep = Union[str, Dict[str, Any]]
node_status = False


class MockContent:
    """Mock content object for testing."""

    def __init__(self, text: str) -> None:
        """Initialize mock content.

        Args:
            text: Text content
        """
        self.text = text


class MockResult:
    """Mock result object for testing."""

    def __init__(self, text_content: str) -> None:
        """Initialize mock result.

        Args:
            text_content: Text content
        """
        self.content = [MockContent(text_content)]
        self.data = text_content


def launch_ui(
    host: str = "127.0.0.1", port: int = 5050, admin_mode: bool = False
) -> None:
    """Launch UltraRAG UI server.

    Args:
        host: Server host address (default: "127.0.0.1")
        port: Server port (default: 5050)
        admin_mode: Whether to run in admin mode (default: False)

    Raises:
        RuntimeError: If UI backend cannot be loaded or server fails to start
    """
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from ui.backend.app import create_app
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the UI backend. "
            "Please ensure the `ui/backend` directory exists and is importable."
        ) from exc

    app = create_app(admin_mode=admin_mode)
    ui_logger = logging.getLogger("UltraRAG-UI")
    mode_str = "Admin" if admin_mode else "Chat"
    ui_logger.info("UltraRAG UI (%s mode) started: http://%s:%d", mode_str, host, port)

    try:
        app.run(host=host, port=port, debug=False)
    except OSError as exc:
        raise RuntimeError(
            f"Failed to start UltraRAG UI (host={host}, port={port}): {exc}"
        ) from exc


class Configuration:
    """Configuration manager for loading environment variables and YAML files."""

    def __init__(self) -> None:
        """Initialize configuration and load environment variables."""
        self.load_env()

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """Load YAML configuration file.

        Args:
            file_path: Path to YAML file

        Returns:
            Dictionary containing configuration (None if file is empty)
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def load_parameter_config(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load parameter configuration from YAML file.

        Args:
            file_path: Path to parameter YAML file

        Returns:
            Dictionary containing parameters (empty dict if file doesn't exist)
        """
        path = Path(file_path)
        if not path.is_file():
            return {}
        return yaml.safe_load(path.read_text())


ROOT = "BASE"
SEP = "/"
# Use ContextVar to implement coroutine-safe loop termination flag
# Each coroutine (user request) has an independent LoopTerminal copy
_loop_terminal_var: contextvars.ContextVar[List[bool]] = contextvars.ContextVar(
    "loop_terminal", default=[]
)


# Sentinel value: used to distinguish "data not set" from "data is None"
class _Unset:
    """Sentinel value to indicate data has not been set by any branch."""

    __slots__ = ()

    def __repr__(self):
        return "<UNSET>"


UNSET = _Unset()


def parse_path(path: str) -> List[Tuple[int, str]]:
    """Parse branch path string into list of (depth, state) tuples.

    Example: 'branch1_finished/branch2_retry' → [(1,'finished'), (2,'retry')]

    Args:
        path: Branch path string

    Returns:
        List of (depth, state) tuples
    """
    if not path or path == ROOT:
        return []
    pairs = []
    if path.startswith(ROOT + SEP):
        path = path[len(ROOT + SEP) :]
    for seg in path.split(SEP):
        depth, state = seg.split("_", 1)
        pairs.append((int(depth.replace("branch", "")), state))
    return pairs


def elem_match(elem: Dict, pairs: List[Tuple[int, str]]) -> bool:
    """Check if element matches all branch state pairs.

    Args:
        elem: Element dictionary to check
        pairs: List of (depth, state) tuples

    Returns:
        True if element matches all pairs, False otherwise
    """
    return all(elem.get(f"branch{d}_state") == s for d, s in pairs)


def is_wrapped_list(lst: Any) -> bool:
    """Check if list is in wrapped format (contains {data, branch*_state} structure).

    Args:
        lst: List to check

    Returns:
        True if list is in wrapped format, False otherwise
    """
    if not isinstance(lst, list) or not lst:
        return False
    first = lst[0]
    if not isinstance(first, dict):
        return False
    # Check if there are any branch*_state keys
    return any(k.startswith("branch") and k.endswith("_state") for k in first.keys())


class UltraData:
    """Data manager for UltraRAG pipeline execution.

    Manages server configurations, global variables, I/O mappings,
    and memory snapshots for pipeline steps.
    """

    def __init__(
        self,
        pipeline_yaml_path: str,
        server_configs: Optional[Dict[str, Dict]] = None,
        parameter_file: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize UltraData with pipeline configuration.

        Args:
            pipeline_yaml_path: Path to pipeline YAML file
            server_configs: Optional pre-loaded server configurations
            parameter_file: Optional path to parameter file
        """
        self.pipeline_yaml_path = pipeline_yaml_path
        cfg = Configuration()
        pipeline = cfg.load_config(pipeline_yaml_path)
        servers = pipeline.get("servers", {})
        server_paths = servers

        if server_configs:
            self.servers = server_configs
        else:
            self.servers = {
                name: cfg.load_config(os.path.join(path, "server.yaml"))
                for name, path in server_paths.items()
            }

        self.local_vals = {
            name: cfg.load_parameter_config(os.path.join(path, "parameter.yaml"))
            for name, path in server_paths.items()
        }
        cfg_path = Path(pipeline_yaml_path)
        if parameter_file is not None:
            param_file = Path(parameter_file)
        else:
            param_file = (
                cfg_path.parent / "parameter" / f"{cfg_path.stem}_parameter.yaml"
            )
        all_local_vals = cfg.load_parameter_config(param_file)
        self.local_vals.update(all_local_vals)
        self.io = {}
        self.global_vars = {}
        self._extract_io(pipeline.get("pipeline", []))
        # Store history of memory states after each step
        self.snapshots: List[Dict[str, Any]] = []

    def _canonical_mem(self, name: str) -> str:
        """Convert memory variable name to canonical format.

        Args:
            name: Variable name (may start with "mem_" or "memory_")

        Returns:
            Canonical memory variable name (starts with "memory_")
        """
        if name.startswith("mem_"):
            return "memory_" + name[4:]
        return name

    def _get_branch_skeleton(self, depth: int) -> Optional[List[Dict[str, Any]]]:
        """Get branch skeleton structure for a given depth.

        Args:
            depth: Branch depth

        Returns:
            Skeleton list if found, None otherwise
        """
        key = f"branch{depth}_state"
        for v in self.global_vars.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and key in v[0]:
                return v
        return None

    def _pad_to_skeleton(
        self,
        skeleton: List[Dict[str, Any]],
        parent_pairs: List[Tuple[int, str]],
        sub_list: List[Any],
    ) -> List[Dict[str, Any]]:
        """Pad sub-list to match skeleton structure.

        Args:
            skeleton: Skeleton list structure
            parent_pairs: Parent branch state pairs
            sub_list: Sub-list to pad

        Returns:
            Padded list matching skeleton structure

        Raises:
            ValueError: If sub_list length doesn't match expected matches
        """
        new_full = []
        for elem in skeleton:
            new_elem = {k: v for k, v in elem.items() if k != "data"}
            new_elem["data"] = UNSET  # Use sentinel value to mark "not set"
            new_full.append(new_elem)

        it = iter(sub_list)
        for i, elem in enumerate(skeleton):
            if elem_match(elem, parent_pairs):
                try:
                    new_val = next(it)
                except StopIteration:
                    raise ValueError(
                        "[UltraRAG Error] Router sub_list length < expected matches when padding to skeleton"
                    )
                new_full[i]["data"] = new_val

        if any(True for _ in it):
            raise ValueError(
                "[UltraRAG Error] Router sub_list length > expected matches when padding to skeleton"
            )

        return new_full

    def _update_memory(self, var_name: str, value: Any):
        def unwrap(v):
            if isinstance(v, list) and v and isinstance(v[0], dict) and "data" in v[0]:
                return [item["data"] for item in v]
            return v

        mem_key = self._canonical_mem(
            var_name
            if var_name.startswith(("mem_", "memory_"))
            else f"memory_{var_name}"
        )

        if mem_key not in self.global_vars:
            self.global_vars[mem_key] = []
        self.global_vars[mem_key].append(copy.deepcopy(unwrap(value)))

        logger.debug("Updated memory %s -> %s", mem_key, self.global_vars[mem_key][-1])

    def _extract_io(self, pipeline: List[Any]) -> None:
        """Extract input/output mappings from pipeline configuration.

        Args:
            pipeline: List of pipeline steps

        Raises:
            ValueError: If required variables are not found in configuration
        """
        for pipe in pipeline:
            if isinstance(pipe, str):
                srv_name, tool_name = pipe.split(".")
                if len(self.servers) > 1:
                    tool_name_concated = f"{srv_name}_{tool_name}"
                else:
                    tool_name_concated = f"{tool_name}"

                if tool_name_concated not in self.io:
                    self.io[tool_name_concated] = {
                        "input": {},
                        "output": set(),
                    }

                if not srv_name == "prompt":
                    tool_input = self.servers[srv_name]["tools"][tool_name][
                        "input"
                    ].copy()
                else:
                    tool_input = self.servers[srv_name]["prompts"][tool_name][
                        "input"
                    ].copy()
                self.io[tool_name_concated]["input"].update(tool_input)

                for _, input_val in tool_input.items():
                    if input_val.startswith("$"):
                        stripped = input_val[1:]
                        if stripped not in self.local_vals[srv_name]:
                            raise ValueError(
                                f"[UltraRAG Error] Variable {stripped} not found in {srv_name} parameter.yaml"
                            )
                        if f"memory_{stripped}" not in self.global_vars:
                            self.global_vars[f"memory_{stripped}"] = []
                    else:
                        mem_name = self._canonical_mem(input_val)
                        if mem_name.startswith("memory_"):
                            if mem_name not in self.global_vars:
                                self.global_vars[mem_name] = []
                        else:
                            if input_val not in self.global_vars.keys():
                                raise ValueError(
                                    f"[UltraRAG Error] Variable {input_val} cannot be found from pipeline before {srv_name}.{tool_name} step"
                                )
                            if f"memory_{input_val}" not in self.global_vars:
                                self.global_vars[f"memory_{input_val}"] = []

                if not srv_name == "prompt":
                    tool_output = self.servers[srv_name]["tools"][tool_name].get(
                        "output", []
                    )
                else:
                    tool_output = self.servers[srv_name]["prompts"][tool_name].get(
                        "output", []
                    )

                self.io[tool_name_concated]["output"].update(tool_output)

                for output_val in tool_output:
                    if output_val.startswith("$"):
                        output_val = output_val[1:]
                        if not output_val in self.local_vals[srv_name]:
                            raise ValueError(
                                f"[UltraRAG Error] Variable {output_val} not found in {srv_name} parameter.yaml"
                            )
                    self.global_vars[output_val] = None
                    self.global_vars[f"memory_{output_val}"] = []
            elif isinstance(pipe, dict) and "loop" in pipe:
                self._extract_io(pipe["loop"].get("steps", []))
            elif isinstance(pipe, dict) and "branch" in pipe:
                self._extract_io(pipe["branch"].get("router", []))
                for _, branch_steps in pipe["branch"]["branches"].items():
                    self._extract_io(branch_steps)
            elif isinstance(pipe, dict) and "." in list(pipe.keys())[0]:
                srv_name, tool_name = list(pipe.keys())[0].split(".")
                tool_value = pipe[list(pipe.keys())[0]]
                if len(self.servers) > 1:
                    tool_name_concated = f"{srv_name}_{tool_name}"
                else:
                    tool_name_concated = f"{tool_name}"

                if tool_name_concated not in self.io:
                    self.io[tool_name_concated] = {
                        "input": {},
                        "output": set(),
                    }

                if not srv_name == "prompt":
                    tool_input = self.servers[srv_name]["tools"][tool_name][
                        "input"
                    ].copy()
                else:
                    tool_input = self.servers[srv_name]["prompts"][tool_name][
                        "input"
                    ].copy()
                self.io[tool_name_concated]["input"].update(tool_input)
                tool_input.update(tool_value.get("input", {}))

                for _, input_val in tool_input.items():
                    if input_val.startswith("$"):
                        stripped = input_val[1:]
                        if stripped not in self.local_vals[srv_name]:
                            raise ValueError(
                                f"[UltraRAG Error] Variable {stripped} not found in {srv_name} parameter.yaml"
                            )
                        if f"memory_{stripped}" not in self.global_vars:
                            self.global_vars[f"memory_{stripped}"] = []
                    else:
                        mem_name = self._canonical_mem(input_val)
                        if mem_name.startswith("memory_"):
                            if mem_name not in self.global_vars:
                                self.global_vars[mem_name] = []
                        else:
                            if input_val not in self.global_vars.keys():
                                raise ValueError(
                                    f"[UltraRAG Error] Variable {input_val} cannot be found from pipeline before {srv_name}.{tool_name} step"
                                )
                            if f"memory_{input_val}" not in self.global_vars:
                                self.global_vars[f"memory_{input_val}"] = []

                if not srv_name == "prompt":
                    tool_output = self.servers[srv_name]["tools"][tool_name].get(
                        "output", []
                    )
                else:
                    tool_output = self.servers[srv_name]["prompts"][tool_name].get(
                        "output", []
                    )
                self.io[tool_name_concated]["output"].update(tool_output)
                output_index = tool_value.get("output", {})
                tool_output = [output_index.get(key, key) for key in tool_output]

                for output_val in tool_output:
                    if output_val.startswith("$"):
                        output_val = output_val[1:]
                        if not output_val in self.local_vals[srv_name]:
                            raise ValueError(
                                f"[UltraRAG Error] Variable {output_val} not found in {srv_name} parameter.yaml"
                            )
                    self.global_vars[output_val] = None
                    # initialise corresponding memory list
                    self.global_vars[f"memory_{output_val}"] = []
            else:
                raise ValueError(f"[UltraRAG Error] Unrecognized pipeline step: {pipe}")

    def get_data(
        self,
        server_name: str,
        tool_name: str,
        branch_state: str,
        input_dict: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, Dict[str, Any], bool]:
        """Get input data for a tool execution step.

        Args:
            server_name: Name of the server
            tool_name: Name of the tool
            branch_state: Current branch state path
            input_dict: Optional additional input dictionary

        Returns:
            Tuple of (concatenated_name, args_input, signal)

        Raises:
            ValueError: If required variables are not found
        """
        concated = f"{server_name}_{tool_name}" if len(self.servers) > 1 else tool_name
        path_pairs = parse_path(branch_state)
        args_input = {}
        signal = None
        input_items = self.io[concated]["input"]
        input_items.update(input_dict or {})
        for k, v in input_items.items():
            if isinstance(v, str):
                if v.startswith("$"):
                    v = v[1:]

                    if v in self.local_vals[server_name]:
                        args_input[k] = self.local_vals[server_name][v]
                    else:
                        raise ValueError(
                            f"Variable {v} not found for step {server_name}.{tool_name}"
                        )

                else:
                    v = self._canonical_mem(v)
                    if v in self.global_vars:
                        val = self.global_vars[v]
                        # Only unwrap if list is in wrapped format
                        if (
                            isinstance(val, list)
                            and val
                            and isinstance(val[0], dict)
                            and is_wrapped_list(val)
                        ):
                            signal = signal & True if signal is not None else True
                            # val = [e["data"] for e in val if elem_match(e, path_pairs)]
                            sub = [
                                e["data"]
                                for e in val
                                if elem_match(e, path_pairs) and e["data"] is not UNSET
                            ]
                            val = sub
                            if signal is None:
                                signal = not bool(val)
                            logger.debug(f"val after filtering: {val}")
                            if len(val) != 0:
                                signal = False
                        args_input[k] = val
                    else:
                        raise ValueError(
                            f"Variable {v} not found in var pool for step {server_name}.{tool_name}"
                        )
        logger.debug(
            f"Executing step {server_name}.{tool_name} with args: {args_input}"
        )
        return concated, args_input, signal or False

    def save_data(
        self,
        server_name: str,
        tool_name: str,
        data: Any,
        state: str,
        output_dict: Optional[Dict[str, str]] = None,
    ) -> None:
        """Save output data from a tool execution step.

        Args:
            server_name: Name of the server
            tool_name: Name of the tool
            data: Output data to save
            state: Current branch state
            output_dict: Optional output mapping dictionary

        Raises:
            ValueError: If data length doesn't match expected structure
        """
        if output_dict is None:
            output_dict = {}
        concated = f"{server_name}_{tool_name}" if len(self.servers) > 1 else tool_name
        # Track which memory keys are updated for this step
        updated_mem_keys = []
        if server_name == "prompt":
            output_key = list(self.io[concated]["output"])[0]
            var_name = output_dict.get(output_key, output_key)

            # Handle branch case: if in branch, need to pad result to full length
            parent_pairs = parse_path(state)
            depth = parent_pairs[-1][0] if parent_pairs else 0

            if depth > 0:
                # In branch, need to pad short list to full length
                full_list = self.global_vars.get(var_name)
                sub_list = data.messages

                if is_wrapped_list(full_list):
                    # Already in wrapped format, need to fill by position
                    it = iter(sub_list)
                    for i, elem in enumerate(full_list):
                        if elem_match(elem, parent_pairs):
                            try:
                                new_elem = next(it)
                            except StopIteration:
                                raise ValueError(
                                    f"[UltraRAG Error] Prompt {var_name} length < "
                                    f"global_vars in step {server_name}.{tool_name}"
                                )
                            full_list[i]["data"] = new_elem
                    if any(True for _ in it):
                        raise ValueError(
                            f"[UltraRAG Error] Prompt {var_name} length > "
                            f"global_vars in step {server_name}.{tool_name}"
                        )
                    self.global_vars[var_name] = full_list
                else:
                    # Try to get skeleton structure from other variables
                    skeleton = self._get_branch_skeleton(depth)
                    if skeleton:
                        padded = self._pad_to_skeleton(skeleton, parent_pairs, sub_list)
                        self.global_vars[var_name] = padded
                    else:
                        # If no skeleton, save directly (this case should not occur)
                        self.global_vars[var_name] = sub_list
            else:
                # Not in branch, save directly
                self.global_vars[var_name] = data.messages

            self._update_memory(var_name, self.global_vars[var_name])
            mem_key_updated = self._canonical_mem(
                var_name
                if var_name.startswith(("mem_", "memory_"))
                else f"memory_{var_name}"
            )
            updated_mem_keys.append(mem_key_updated)
        else:
            output_keys = self.io[concated]["output"]
            iter_keys = list(output_dict.keys()) if output_dict else list(output_keys)

            if len(output_keys) > 0:
                data = json.loads(data.content[0].text)
                for key in iter_keys:
                    if not key.replace("$", "") in data:
                        raise ValueError(
                            f"[UltraRAG Error] Output key {key} not found in data for step {server_name}.{tool_name}"
                        )
                    if key.startswith("$"):
                        if key[1:] in self.local_vals[server_name]:
                            key = key[1:]
                            self.local_vals[server_name][output_dict.get(key, key)] = (
                                data[key]
                            )
                        else:
                            raise ValueError(
                                f"[UltraRAG Error] Variable {key[1:]} not found in {server_name} parameter.yaml"
                            )
                    elif output_dict.get(key, key) in self.global_vars:
                        if state.split(SEP)[-1] == "router":
                            is_router = True
                            parent_pairs = parse_path(state.rsplit(SEP, 1)[0])
                            depth = parent_pairs[-1][0] + 1 if parent_pairs else 1
                            state_key = f"branch{depth}_state"
                        else:
                            is_router = False
                            parent_pairs = parse_path(state)
                            depth = parent_pairs[-1][0] if parent_pairs else 0

                        if depth > 0:
                            full_list = self.global_vars[output_dict.get(key, key)]
                            sub_list = data[key]
                            it = iter(sub_list)
                            # Use is_wrapped_list check instead of only checking first element's data
                            # Fix: Previous condition failed when first element's data is None (e.g., in other branches)
                            if not is_router and is_wrapped_list(full_list):
                                for i, elem in enumerate(full_list):
                                    if elem_match(elem, parent_pairs):
                                        try:
                                            new_elem = next(it)
                                        except StopIteration:
                                            raise ValueError(
                                                f"[UltraRAG Error] Router {key} length < global_vars in step {server_name}.{tool_name}"
                                            )
                                        full_list[i]["data"] = new_elem
                                if any(True for _ in it):
                                    raise ValueError(
                                        f"[UltraRAG Error] Router {key} length > global_vars in step {server_name}.{tool_name}"
                                    )
                                self.global_vars[output_dict.get(key, key)] = full_list

                            elif is_router:
                                # Use is_wrapped_list to check if list is already in wrapped format
                                # Fix: Previous condition was inaccurate when first element's data is None
                                if (
                                    depth == 1
                                    and isinstance(full_list, list)
                                    and not is_wrapped_list(full_list)
                                ):
                                    full_list = [
                                        {
                                            "data": new_elem["data"],
                                            state_key: new_elem["state"],
                                        }
                                        for new_elem in sub_list
                                    ]
                                elif depth == 1 and not full_list:
                                    full_list = [
                                        {"data": item["data"], state_key: item["state"]}
                                        for item in sub_list
                                    ]
                                else:
                                    for i, elem in enumerate(full_list):
                                        if elem_match(elem, parent_pairs):
                                            try:
                                                new_elem = next(it)
                                            except StopIteration:
                                                raise ValueError(
                                                    f"[UltraRAG Error] Router {key} length < global_vars"
                                                )

                                            full_list[i]["data"] = new_elem["data"]
                                            full_list[i][state_key] = new_elem["state"]
                                    if any(True for _ in it):
                                        raise ValueError(
                                            f"[UltraRAG Error] Router {key} length > global_vars in step {server_name}.{tool_name}"
                                        )
                                self.global_vars[output_dict.get(key, key)] = full_list
                                for other_key, other_val in self.global_vars.items():
                                    if other_key == output_dict.get(key, key):
                                        continue
                                    if isinstance(other_val, list) and len(
                                        other_val
                                    ) == len(full_list):
                                        if other_val and isinstance(other_val[0], dict):
                                            for i in range(len(other_val)):
                                                if state_key in other_val[i]:
                                                    other_val[i][state_key] = full_list[
                                                        i
                                                    ][state_key]
                                self.remain_branch = set(
                                    [new_elem["state"] for new_elem in sub_list]
                                )
                            else:
                                skeleton = self._get_branch_skeleton(depth)
                                if skeleton:
                                    padded = self._pad_to_skeleton(
                                        skeleton, parent_pairs, sub_list
                                    )
                                    self.global_vars[output_dict.get(key, key)] = padded
                                else:
                                    self.global_vars[output_dict.get(key, key)] = data[
                                        key
                                    ]
                        else:
                            self.global_vars[output_dict.get(key, key)] = data[key]
                    else:
                        raise ValueError(
                            f"[UltraRAG Error] Output key {key} not found in data for step {server_name}.{tool_name}"
                        )
            # -------- update memory pools --------
            for key in iter_keys:
                var_name = output_dict.get(key, key)
                if var_name in self.global_vars:
                    self._update_memory(var_name, self.global_vars[var_name])
                    mem_key_updated = self._canonical_mem(
                        var_name
                        if var_name.startswith(("mem_", "memory_"))
                        else f"memory_{var_name}"
                    )
                    updated_mem_keys.append(mem_key_updated)

        # -------- record snapshot --------
        def _serialise(obj):
            """Recursively convert FastMCP Message / TextContent objects to plain text for JSON."""
            if isinstance(obj, list):
                return [_serialise(e) for e in obj]
            # FastMCP Message → .content.text
            if hasattr(obj, "content"):
                content = getattr(obj, "content")
                if hasattr(content, "text"):
                    return content.text
            # TextContent or similar → .text
            if hasattr(obj, "text"):
                return obj.text
            return obj  # fall back (will be handled by json default=str later)

        # Only record the memory entries updated by this step; store the latest value only
        mem_for_step = {}
        for mk in updated_mem_keys:
            if mk in self.global_vars:
                v = self.global_vars[mk]
                latest = v[-1] if isinstance(v, list) and v else v
                mem_for_step[mk] = _serialise(copy.deepcopy(latest))

        snapshot = {
            "step": f"{server_name}.{tool_name}",
            "memory": mem_for_step,
        }
        self.snapshots.append(snapshot)
        logger.debug(
            f"Saved data for {server_name}.{tool_name} to global_vars: {self.global_vars}"
        )
        return data

    def write_memory_output(self, pipeline_name: str, timestamp: str) -> None:
        """Write memory snapshots to JSON file.

        Args:
            pipeline_name: Name of the pipeline
            timestamp: Timestamp string for filename
        """
        benchmark_cfg = self.local_vals.get("benchmark", {})
        if isinstance(benchmark_cfg, dict):
            if "benchmark" in benchmark_cfg and "name" in benchmark_cfg["benchmark"]:
                benchmark_name = benchmark_cfg["benchmark"]["name"]
            else:
                benchmark_name = ""

        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = (
            output_dir / f"memory_{benchmark_name}_{pipeline_name}_{timestamp}.json"
        )

        with open(file_path, "w", encoding="utf-8") as fp:
            json.dump(self.snapshots, fp, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Memory output saved to {file_path}")

    def get_branch(self) -> set[str]:
        """Get remaining branch states.

        Returns:
            Set of remaining branch state names
        """
        logger.debug(f"remain_branch: {self.remain_branch}")
        return self.remain_branch


async def build(config_path: str) -> None:
    """Build server and parameter configuration files from pipeline.

    Args:
        config_path: Path to pipeline configuration file

    Raises:
        FileNotFoundError: If server file doesn't exist
        NodeNotInstalledError: If Node.js is required but not installed
        NodeVersionTooLowError: If Node.js version is too low
    """
    global node_status
    logger.info(f"Building configuration {config_path}")
    cfg_path = Path(config_path)
    pipline_name = cfg_path.stem
    loader = Configuration()
    init_cfg = loader.load_config(config_path)
    servers = init_cfg.get("servers", {})
    server_paths = servers

    parameter_path = {
        name: os.path.join(path, "parameter.yaml")
        for name, path in server_paths.items()
    }

    server_cfgs = {
        name: loader.load_parameter_config(os.path.join(path, "parameter.yaml"))
        for name, path in server_paths.items()
    }

    for name, path in server_paths.items():
        if not server_cfgs[name]:
            logger.warning(f"No parameter.yaml found for {name}, skipping")
            server_cfgs[name] = {}

        actual_server_path = path
        base_dir_name = os.path.basename(os.path.normpath(actual_server_path))
        server_cfgs[name]["path"] = server_cfgs[name].get(
            "path", str(Path(actual_server_path) / "src" / f"{base_dir_name}.py")
        )

    logger.debug("Server configurations loaded: %s", server_cfgs)

    mcp_servers: Dict[str, Any] = {}
    for name, conf in server_cfgs.items():
        path = conf.get("path", "")
        if path.endswith(".py"):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"[UltraRAG Error] Cannot find the server file of {name}: {path}"
                )
            mcp_servers[name] = {
                "command": "python",
                "args": [path],
                "env": os.environ.copy(),
            }
        elif path.startswith(("http://", "https://")):
            if not node_status:
                try:
                    check_node_version(20)
                    node_status = True
                except NodeNotInstalledError as e:
                    logger.error(
                        "[UltraRAG Error] Node.js is not installed or not found in PATH. Please install Node.js >= 20."
                    )
                    logger.error(str(e))
                    sys.exit(1)
                except NodeVersionTooLowError as e:
                    logger.error(
                        "[UltraRAG Error] Node.js version is too low. Please upgrade to Node.js >= 20."
                    )
                    logger.error(str(e))
                    sys.exit(1)
            mcp_servers[name] = (
                {
                    "command": "npx",
                    "args": [
                        "-y",
                        "mcp-remote",
                        path,
                    ],
                    "env": os.environ.copy(),
                },
            )
        else:
            raise ValueError(
                f"[UltraRAG Error] Unsupported server type for {name}: {path}"
            )

    mcp_cfg = {"mcpServers": mcp_servers}
    logger.debug("Initializing MCP client with config: %s", mcp_cfg)

    client = Client(mcp_cfg)
    # logging.getLogger("FastMCP").setLevel(logging.WARNING)
    logger.info("Building server configs")
    already_built = []
    parameter_all = {}
    server_all = {}

    async def build_steps(steps: List[PipelineStep]):
        nonlocal already_built, parameter_all, server_all
        for step in steps:
            if isinstance(step, str):
                srv_name, tool_name = step.split(".")
                full_tool = (
                    f"{srv_name}_build" if len(server_cfgs.keys()) > 1 else "build"
                )
                if srv_name not in server_all:
                    server_all[srv_name] = {
                        "prompts" if srv_name == "prompt" else "tools": {}
                    }
                    await client.call_tool(
                        full_tool, {"parameter_file": parameter_path[srv_name]}
                    )
                    logger.info(f"server.yaml for {srv_name} has been built already")
                param = loader.load_parameter_config(parameter_path[srv_name])
                serv = loader.load_parameter_config(
                    parameter_path[srv_name].replace("parameter.yaml", "server.yaml")
                )
                if os.path.exists(parameter_path[srv_name]):
                    server_all[srv_name]["parameter"] = parameter_path[srv_name]
                server_all[srv_name]["path"] = serv["path"]
                if param != {}:
                    if srv_name not in parameter_all:
                        parameter_all[srv_name] = {}
                    if srv_name == "prompt":
                        input_values: List[str] = serv["prompts"][tool_name][
                            "input"
                        ].values()
                    else:
                        input_values: List[str] = serv["tools"][tool_name][
                            "input"
                        ].values()
                    for k in input_values:
                        if k.startswith("$"):
                            parameter_all[srv_name][k[1:]] = param[k[1:]]
                if serv != {}:
                    if srv_name == "prompt":
                        server_all[srv_name]["prompts"][tool_name] = serv["prompts"][
                            tool_name
                        ]
                    else:
                        server_all[srv_name]["tools"][tool_name] = serv["tools"][
                            tool_name
                        ]

            elif isinstance(step, dict):
                if "loop" in step:
                    loop_steps = step["loop"].get("steps", [])
                    await build_steps(loop_steps)
                elif "branch" in step:
                    await build_steps(step["branch"].get("router", []))
                    for _, branch_steps in step["branch"]["branches"].items():
                        await build_steps(branch_steps)
                elif "." in list(step.keys())[0]:
                    srv_name, tool_name = list(step.keys())[0].split(".")
                    full_tool = (
                        f"{srv_name}_build" if len(server_cfgs.keys()) > 1 else "build"
                    )
                    if not srv_name in server_all:

                        server_all[srv_name] = {
                            "prompts" if srv_name == "prompt" else "tools": {}
                        }
                        await client.call_tool(
                            full_tool, {"parameter_file": parameter_path[srv_name]}
                        )
                        logger.info(
                            f"server.yaml for {srv_name} has been built already"
                        )
                    param = loader.load_parameter_config(parameter_path[srv_name])
                    serv = loader.load_parameter_config(
                        parameter_path[srv_name].replace(
                            "parameter.yaml", "server.yaml"
                        )
                    )
                    if os.path.exists(parameter_path[srv_name]):
                        server_all[srv_name]["parameter"] = parameter_path[srv_name]
                        server_all[srv_name]["path"] = serv["path"]
                    if param != {}:
                        if srv_name not in parameter_all:
                            parameter_all[srv_name] = {}
                        if srv_name == "prompt":
                            input_values: List[str] = serv["prompts"][tool_name][
                                "input"
                            ].values()
                        else:
                            input_values: List[str] = serv["tools"][tool_name][
                                "input"
                            ].values()
                        for k in input_values:
                            if k.startswith("$"):
                                # logger.info(parameter_all)
                                # logger.info(already_built)
                                parameter_all[srv_name][k[1:]] = param[k[1:]]

                    if serv != {}:
                        if srv_name == "prompt":
                            server_all[srv_name]["prompts"][tool_name] = serv[
                                "prompts"
                            ][tool_name]
                        else:
                            server_all[srv_name]["tools"][tool_name] = serv["tools"][
                                tool_name
                            ]
                else:
                    raise ValueError(
                        f"[UltraRAG Error] Unrecognized step in branch: {step}"
                    )
            else:
                raise ValueError(f"[UltraRAG Error] Unrecognized pipeline step: {step}")

    async with client:
        await build_steps(init_cfg.get("pipeline", []))

    param_save_path = cfg_path.parent / "parameter" / f"{pipline_name}_parameter.yaml"
    server_save_path = cfg_path.parent / "server" / f"{pipline_name}_server.yaml"
    param_save_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving all parameters to {param_save_path}")
    server_save_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving all server configs to {server_save_path}")

    with open(param_save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(parameter_all, f, allow_unicode=True, sort_keys=False)
    logger.info(f"All parameters have been saved in {param_save_path}")

    for srv_name in server_all:
        if "path" not in server_all[srv_name]:
            server_all[srv_name]["path"] = server_cfgs[srv_name]["path"]

    with open(server_save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(server_all, f, allow_unicode=True, sort_keys=False)
    logger.info(f"All server configurations have been saved in {server_save_path}")


def load_pipeline_context(
    config_path: str, param_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Load pipeline context from configuration files.

    Args:
        config_path: Path to pipeline configuration file
        param_path: Optional path to parameter file

    Returns:
        Dictionary containing pipeline context

    Raises:
        FileNotFoundError: If parameter file doesn't exist
    """
    global node_status
    cfg_path = Path(config_path)
    logger.info(f"Executing pipeline with configuration {config_path}")
    cfg = Configuration()
    init_cfg = cfg.load_config(config_path)
    servers = init_cfg.get("servers", {})
    pipeline_cfg = init_cfg.get("pipeline", [])
    server_paths = servers

    cfg_name = cfg_path.stem
    root_path = cfg_path.parent

    server_config_path = root_path / "server" / f"{cfg_name}_server.yaml"
    all_server_configs = cfg.load_config(server_config_path)
    server_cfg = {
        name: all_server_configs[name]
        for name in server_paths
        if name in all_server_configs
    }

    if param_path is not None:
        provided_path = Path(param_path).expanduser()
        candidate_paths = []
        if provided_path.is_absolute():
            candidate_paths.append(provided_path)
        else:
            candidate_paths.append(Path.cwd() / provided_path)
            candidate_paths.append(root_path / provided_path)

        param_config_path = next((p for p in candidate_paths if p.exists()), None)
        if param_config_path is None:
            raise FileNotFoundError(
                f"[UltraRAG Error] Parameter file '{provided_path}' does not exist"
            )
        param_config_path = param_config_path.resolve()
    else:
        param_config_path = root_path / "parameter" / f"{cfg_name}_parameter.yaml"

    param_cfg = cfg.load_parameter_config(param_config_path)
    for srv_name in server_cfg.keys():
        server_cfg[srv_name]["parameter"] = param_cfg.get(srv_name, {})

    mcp_cfg = {"mcpServers": {}}
    for name, sc in server_cfg.items():
        path = sc.get("path", "")
        if path.endswith(".py"):
            mcp_cfg["mcpServers"][name] = {
                "command": "python",
                "args": [path],
                "env": os.environ.copy(),
            }
        elif path.startswith(("http://", "https://")):
            if not node_status:
                try:
                    check_node_version(20)
                    node_status = True
                except NodeNotInstalledError as e:
                    logger.error(
                        "[UltraRAG Error] Node.js is not installed or not found in PATH. Please install Node.js >= 20."
                    )
                    logger.error(str(e))
                    sys.exit(1)
                except NodeVersionTooLowError as e:
                    logger.error(
                        "[UltraRAG Error] Node.js version is too low. Please upgrade to Node.js >= 20."
                    )
                    logger.error(str(e))
                    sys.exit(1)
            mcp_cfg["mcpServers"][name] = (
                {
                    "command": "npx",
                    "args": [
                        "-y",
                        "mcp-remote",
                        path,
                    ],
                    "env": os.environ.copy(),
                },
            )
        else:
            raise ValueError(f"Unsupported server type for {name}: {path}")

    return {
        "config_path": config_path,
        "param_config_path": param_config_path,
        "cfg_name": cfg_name,
        "mcp_cfg": mcp_cfg,
        "server_cfg": server_cfg,
        "pipeline_cfg": pipeline_cfg,
        "init_cfg": init_cfg,
    }


def create_mcp_client(mcp_cfg: Dict[str, Any]) -> Client:
    """Create and initialize MCP client.

    Args:
        mcp_cfg: MCP server configuration dictionary

    Returns:
        Initialized MCP Client instance
    """
    logger.info("Initializing MCP Client...")
    return Client(mcp_cfg)


def _summarize_step_result(step_name: str, result: Any) -> str:
    """Summarize step execution result for logging/display.

    Args:
        step_name: Name of the step
        result: Step execution result

    Returns:
        Human-readable summary string
    """
    try:
        content = ""
        if hasattr(result, "content") and result.content:
            content = result.content[0].text
        elif isinstance(result, str):
            content = result
        elif hasattr(result, "data"):  # MockResult
            content = result.data
        else:
            try:
                content = json.dumps(result)
            except Exception:
                return "Step completed."

        try:
            data = json.loads(content)
        except Exception:
            return f"Output: {str(content)[:100]}..."

        docs = data.get("ret_psg")

        if docs and isinstance(docs, list):
            if len(docs) > 0 and isinstance(docs[0], list):
                docs = docs[0]

            summary = f"Retrieved {len(docs)} documents:\n"

            for i, doc in enumerate(docs):
                if isinstance(doc, str):
                    # Relax truncation limit to 300 characters to show more document content
                    doc_preview = doc[:300] + "..." if len(doc) > 300 else doc
                    summary += f"{i+1}. {doc_preview}\n"

            return summary.strip()

        if "generate" in step_name.lower():
            if "ans_ls" in data and data["ans_ls"]:
                full_text = data["ans_ls"][0]
                if len(full_text) > 500:
                    return f"Generated:\n{full_text[:500]}...\n"
                return f"Generated:\n{full_text}"

        keys = list(data.keys())
        return f"Output Keys: {keys}"

    except Exception as e:
        return f"Output: {str(content)[:100]}..."


async def execute_pipeline(
    client: Client,
    context: Dict[str, Any],
    is_demo: bool = False,
    return_all: bool = False,
    stream_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    override_params: Optional[Dict[str, Any]] = None,
) -> Any:
    """Execute UltraRAG pipeline with given context.

    Args:
        client: MCP client instance
        context: Pipeline context dictionary
        is_demo: Whether running in demo mode
        return_all: Whether to return all intermediate results
        stream_callback: Optional callback for streaming results
        override_params: Optional parameters to override

    Returns:
        Pipeline execution results
    """
    # Create independent loop termination flag list for current coroutine
    # Use ContextVar to ensure no interference between concurrent users
    _loop_terminal_var.set([])

    config_path = context["config_path"]
    server_cfg = context["server_cfg"]
    param_config_path = context["param_config_path"]
    pipeline_cfg = context["pipeline_cfg"]
    cfg_name = context["cfg_name"]

    Data: UltraData = UltraData(
        config_path, server_configs=server_cfg, parameter_file=param_config_path
    )

    if override_params:
        for srv_name, params in override_params.items():
            if srv_name in Data.local_vals:
                Data.local_vals[srv_name].update(params)
                logger.info(f"Dynamic Override applied for '{srv_name}': {params}")

    generation_services_map = {}
    retriever_aliases = set()

    if is_demo:
        for srv_name, srv_conf in server_cfg.items():
            srv_path = str(srv_conf.get("path", "")).replace("\\", "/")

            if "servers/generation" in srv_path:
                sys.path.append(os.getcwd())
                try:
                    from servers.generation.src.local_generation import (
                        LocalGenerationService,
                    )
                except ImportError:
                    LocalGenerationService = None

                if LocalGenerationService:
                    gen_params = Data.local_vals.get(srv_name, {})
                    try:
                        service_instance = LocalGenerationService(
                            backend_configs=gen_params.get("backend_configs", {}),
                            sampling_params=gen_params.get("sampling_params", {}),
                            extra_params=gen_params.get("extra_params", {}),
                            backend="openai",
                        )
                        generation_services_map[srv_name] = service_instance
                    except Exception as e:
                        logger.warning(
                            f"Failed to init LocalGenerationService for '{srv_name}': {e}"
                        )

            elif "servers/retriever" in srv_path:
                retriever_aliases.add(srv_name)

    doc_id_counter = 0
    doc_content_to_id = {}

    async def _execute_steps(
        steps: List[PipelineStep],
        depth: int = 0,
        state: str = ROOT,
    ):
        nonlocal doc_id_counter
        indent = "  " * depth
        result = None
        for idx, step in enumerate(steps):
            logger.info(f"{indent}Executing step: {step}")

            if isinstance(step, str):
                current_step_name = step
            elif isinstance(step, dict) and len(step) > 0:
                current_step_name = list(step.keys())[0]
            else:
                current_step_name = "Unknown"

            is_final_step = depth == 0 and idx == len(steps) - 1

            if stream_callback:
                await stream_callback(
                    {"type": "step_start", "name": current_step_name, "depth": depth}
                )

            if isinstance(step, dict) and "loop" in step:
                loop_terminal = _loop_terminal_var.get()
                loop_terminal.append(True)
                loop_cfg = step["loop"]
                times = loop_cfg.get("times")
                inner_steps = loop_cfg.get("steps", [])
                if times is None or not isinstance(inner_steps, list):
                    raise ValueError(f"Invalid loop config: {loop_cfg}")
                for st in range(times):
                    loop_terminal[-1] = True
                    loop_res = await _execute_steps(inner_steps, depth + 1, state)
                    if loop_res is not None:
                        result = loop_res
                    logger.debug(
                        f"{indent}Loop iteration {st + 1}/{times} completed {loop_terminal}"
                    )
                    if loop_terminal[-1]:
                        loop_terminal.pop()
                        logger.debug(
                            f"{indent}Loop terminal in iteration {st + 1}/{times}"
                        )
                        break
            elif isinstance(step, dict) and any(k.startswith("branch") for k in step):
                branch_step = step["branch"]
                router = branch_step.get("router", None)
                if not router:
                    raise ValueError(
                        f"Router not found in branch config: {branch_step}"
                    )
                await _execute_steps(
                    router[:-1],
                    depth,
                    state,
                )
                if isinstance(router[-1], str):
                    server_name, tool_name = router[-1].split(".")
                    concated, args_input, _ = Data.get_data(
                        server_name, tool_name, state
                    )
                    router_res = await client.call_tool(concated, args_input)
                    output_text = Data.save_data(
                        server_name, tool_name, router_res, f"{state}{SEP}router"
                    )
                else:
                    server_name, tool_name = list(router[-1].keys())[0].split(".")
                    tool_value = router[-1][list(router[-1].keys())[0]]
                    concated, args_input, _ = Data.get_data(
                        server_name, tool_name, state, tool_value.get("input", {})
                    )
                    router_res = await client.call_tool(concated, args_input)
                    output_text = Data.save_data(
                        server_name,
                        tool_name,
                        router_res,
                        f"{state}{SEP}router",
                        tool_value.get("output", {}),
                    )

                logger.debug(f"{indent}Result: {output_text}")

                branch_depth = parse_path(state)[-1][0] + 1 if parse_path(state) else 1
                branches = Data.get_branch()
                # Reset result to None; if branch produces nothing (e.g., empty branch),
                # this step returns None so upper-level calls (e.g., loop) can retain
                # previous meaningful results
                result = None
                for branch_name in branches:
                    # for branch_name, branch_steps in branch_step["branches"].items():

                    logger.debug(f"{indent}Processing branch: {branch_name}")
                    # branch_steps = branch_step["branches"][branch_name]``
                    branch_res = await _execute_steps(
                        branch_step["branches"][branch_name],
                        depth,
                        f"{state}{SEP}branch{branch_depth}_{branch_name}",
                    )
                    if branch_res is not None:
                        result = branch_res
            elif isinstance(step, dict) and "." in list(step.keys())[0]:
                server_name, tool_name = list(step.keys())[0].split(".")
                tool_value = step[list(step.keys())[0]]
                concated, args_input, signal = Data.get_data(
                    server_name, tool_name, state, tool_value.get("input", {})
                )
                if is_demo:
                    demo_target_retriever_tools = {
                        "retriever_init",
                    }
                    if (
                        server_name in retriever_aliases
                        and tool_name in demo_target_retriever_tools
                    ):
                        args_input["is_demo"] = True

                    demo_target_gen_tools = ["generate", "multimodal_generate"]

                    if (
                        server_name in generation_services_map
                        and tool_name in demo_target_gen_tools
                        and not signal
                    ):
                        local_service = generation_services_map[server_name]

                        full_content = ""
                        try:
                            step_identifier = f"{server_name}.{tool_name}"
                            async for token in local_service.generate_stream(
                                **args_input
                            ):
                                full_content += token

                                if stream_callback:
                                    await stream_callback(
                                        {
                                            "type": "token",
                                            "content": token,
                                            "step": step_identifier,
                                            "is_final": is_final_step,
                                        }
                                    )

                        except Exception as e:
                            logger.error(f"Stream Error: {e}")
                        print("\n")

                        mock_json = json.dumps({"ans_ls": [full_content]})
                        mock_result_obj = MockResult(mock_json)
                        result = mock_result_obj

                        if stream_callback:
                            summary = _summarize_step_result(tool_name, mock_result_obj)
                            await stream_callback(
                                {
                                    "type": "step_end",
                                    "name": current_step_name,
                                    "output": summary,
                                }
                            )

                        Data.save_data(
                            server_name,
                            tool_name,
                            mock_result_obj,
                            state,
                            tool_value.get("output", {}),
                        )
                        if depth > 0:
                            _loop_terminal_var.get()[depth - 1] &= signal
                        continue

                if depth > 0:
                    _loop_terminal_var.get()[depth - 1] &= signal
                if not signal:
                    if server_name == "prompt":
                        result = await client.get_prompt(concated, args_input)
                    else:
                        result = await client.call_tool(concated, args_input)

                    # Check for sources in retriever or citation tools
                    should_extract_sources = (
                        server_name in retriever_aliases
                        or "citation" in tool_name.lower()
                    )
                    if stream_callback and should_extract_sources:
                        try:
                            content_str = ""
                            if hasattr(result, "content") and result.content:
                                content_str = result.content[0].text
                            elif isinstance(result, str):
                                content_str = result

                            data = json.loads(content_str)

                            raw_docs = data.get("ret_psg")

                            if raw_docs and isinstance(raw_docs, list):
                                if len(raw_docs) > 0 and isinstance(raw_docs[0], list):
                                    raw_docs = raw_docs[0]

                                sources = []
                                for i, doc in enumerate(raw_docs):
                                    text = str(doc)

                                    # Check if doc already has [id] prefix
                                    import re

                                    id_match = re.match(r"^\[(\d+)\]\s*", text)
                                    if id_match:
                                        # Extract existing ID and remove prefix from content
                                        current_id = int(id_match.group(1))
                                        text_without_prefix = text[id_match.end() :]
                                        lines = text_without_prefix.strip().split("\n")
                                        title = (
                                            lines[0][:30] + "..."
                                            if lines
                                            else f"Doc {current_id}"
                                        )
                                        content = text_without_prefix
                                    else:
                                        # No prefix, assign new ID
                                        lines = text.strip().split("\n")
                                        title = (
                                            lines[0][:30] + "..."
                                            if lines
                                            else f"Doc {i+1}"
                                        )
                                        content = text

                                        doc_hash = text.strip()
                                        if doc_hash in doc_content_to_id:
                                            current_id = doc_content_to_id[doc_hash]
                                        else:
                                            doc_id_counter += 1
                                            current_id = doc_id_counter
                                            doc_content_to_id[doc_hash] = current_id

                                    sources.append(
                                        {
                                            "id": current_id,
                                            "title": title,
                                            "content": content,
                                        }
                                    )

                                await stream_callback(
                                    {"type": "sources", "data": sources}
                                )
                        except Exception as e:
                            logger.warning(f"Failed to extract sources: {e}")

                    if stream_callback:
                        summary = _summarize_step_result(current_step_name, result)
                        await stream_callback(
                            {
                                "type": "step_end",
                                "name": current_step_name,
                                "output": summary,
                            }
                        )

                    output_text = Data.save_data(
                        server_name,
                        tool_name,
                        result,
                        state,
                        tool_value.get("output", {}),
                    )
                    logger.debug(f"{indent}Result: {output_text}")

                    logger.debug(f"{indent}Updated var pool")
            elif isinstance(step, str):
                server_name, tool_name = step.split(".")

                concated, args_input, signal = Data.get_data(
                    server_name, tool_name, state
                )
                if is_demo:
                    demo_target_retriever_tools = {
                        "retriever_init",
                    }
                    if (
                        server_name in retriever_aliases
                        and tool_name in demo_target_retriever_tools
                    ):
                        args_input["is_demo"] = True

                    demo_target_gen_tools = ["generate", "multimodal_generate"]

                    if (
                        server_name in generation_services_map
                        and tool_name in demo_target_gen_tools
                        and not signal
                    ):
                        local_service = generation_services_map[server_name]

                        full_content = ""
                        try:
                            step_identifier = f"{server_name}.{tool_name}"
                            async for token in local_service.generate_stream(
                                **args_input
                            ):
                                full_content += token

                                if stream_callback:
                                    await stream_callback(
                                        {
                                            "type": "token",
                                            "content": token,
                                            "step": step_identifier,
                                            "is_final": is_final_step,
                                        }
                                    )

                        except Exception as e:
                            logger.error(f"Stream Error: {e}")
                        print("\n")

                        mock_json = json.dumps({"ans_ls": [full_content]})

                        mock_result_obj = MockResult(mock_json)
                        result = mock_result_obj

                        if stream_callback:
                            summary = _summarize_step_result(tool_name, mock_result_obj)
                            await stream_callback(
                                {
                                    "type": "step_end",
                                    "name": current_step_name,
                                    "output": summary,
                                }
                            )

                        Data.save_data(server_name, tool_name, mock_result_obj, state)

                        if depth > 0:
                            _loop_terminal_var.get()[depth - 1] = signal
                        continue

                if depth > 0:
                    _loop_terminal_var.get()[depth - 1] = signal
                if not signal:
                    if server_name == "prompt":
                        result = await client.get_prompt(concated, args_input)
                    else:
                        result = await client.call_tool(concated, args_input)

                    # Check for sources in retriever or citation tools
                    should_extract_sources = (
                        server_name in retriever_aliases
                        or "citation" in tool_name.lower()
                    )
                    if stream_callback and should_extract_sources:
                        try:
                            content_str = ""
                            if hasattr(result, "content") and result.content:
                                content_str = result.content[0].text
                            elif isinstance(result, str):
                                content_str = result
                            data = json.loads(content_str)
                            raw_docs = data.get("ret_psg")

                            if raw_docs and isinstance(raw_docs, list):
                                if len(raw_docs) > 0 and isinstance(raw_docs[0], list):
                                    raw_docs = raw_docs[0]
                                sources = []
                                for i, doc in enumerate(raw_docs):
                                    text = str(doc)

                                    # Check if doc already has [id] prefix
                                    import re

                                    id_match = re.match(r"^\[(\d+)\]\s*", text)
                                    if id_match:
                                        # Extract existing ID and remove prefix from content
                                        current_id = int(id_match.group(1))
                                        text_without_prefix = text[id_match.end() :]
                                        lines = text_without_prefix.strip().split("\n")
                                        title = (
                                            lines[0][:30] + "..."
                                            if lines
                                            else f"Doc {current_id}"
                                        )
                                        content = text_without_prefix
                                    else:
                                        # No prefix, assign new ID
                                        lines = text.strip().split("\n")
                                        title = (
                                            lines[0][:30] + "..."
                                            if lines
                                            else f"Doc {i+1}"
                                        )
                                        content = text

                                        doc_hash = text.strip()
                                        if doc_hash in doc_content_to_id:
                                            current_id = doc_content_to_id[doc_hash]
                                        else:
                                            doc_id_counter += 1
                                            current_id = doc_id_counter
                                            doc_content_to_id[doc_hash] = current_id

                                    sources.append(
                                        {
                                            "id": current_id,
                                            "title": title,
                                            "content": content,
                                        }
                                    )
                                await stream_callback(
                                    {"type": "sources", "data": sources}
                                )
                        except Exception as e:
                            logger.warning(f"Failed to extract sources: {e}")

                    if stream_callback:
                        summary = _summarize_step_result(current_step_name, result)
                        await stream_callback(
                            {
                                "type": "step_end",
                                "name": current_step_name,
                                "output": summary,
                            }
                        )

                    output_text = Data.save_data(server_name, tool_name, result, state)
                    logger.debug(f"{indent}Result: {output_text}")
                    logger.debug(f"{indent}Updated var pool")
            else:
                raise ValueError(f"Unrecognized pipeline step: {step}")

        return result

    tools = await client.list_tools()
    tool_name_lst = [
        tool.name
        for tool in tools
        if not tool.name.endswith("_build" if "_" in tool.name else "build")
    ]
    logger.info(f"Available tools: {tool_name_lst}")

    cleanup_tools = [tool.name for tool in tools if tool.name.endswith("vllm_shutdown")]

    result = None
    try:
        result = await _execute_steps(pipeline_cfg)
        logger.info("Pipeline execution completed.")
    finally:
        for tool_name in cleanup_tools:
            try:
                logger.info(f"Invoking cleanup tool: {tool_name}")
                await client.call_tool(tool_name, {})
            except Exception as exc:
                logger.warning(
                    f"Cleanup tool {tool_name} raised {exc.__class__.__name__}: {exc}"
                )

    # save memory snapshots
    Data.write_memory_output(cfg_name, datetime.now().strftime("%Y%m%d_%H%M%S"))

    if return_all:
        if result is None:
            final = None
        else:
            if hasattr(result, "data"):
                final = result.data
            elif hasattr(result, "content") and result.content:
                final = result.content[0].text
            else:
                final = str(result)
        return {
            "final_result": final,
            "all_results": Data.snapshots,
        }

    if result is None:
        return None

    if hasattr(result, "data"):
        return result.data
    elif hasattr(result, "content") and result.content:
        return result.content[0].text
    return str(result)


async def run(
    config_path: str,
    param_path: Optional[Union[str, Path]] = None,
    return_all: bool = False,
    is_demo: bool = False,
) -> Any:
    """Run UltraRAG pipeline with given configuration.

    Args:
        config_path: Path to pipeline configuration file
        param_path: Optional path to parameter file
        return_all: Whether to return all intermediate results
        is_demo: Whether to run in demo mode

    Returns:
        Pipeline execution results
    """

    log_server_banner(Path(config_path).stem)

    context = load_pipeline_context(config_path, param_path)

    client = create_mcp_client(context["mcp_cfg"])

    async with client:
        result = await execute_pipeline(client, context, is_demo, return_all)

    return result


logging.getLogger("mcp").setLevel(logging.WARNING)


def main() -> None:
    """Main entry point for UltraRAG CLI."""
    parser = argparse.ArgumentParser(prog="ultrarag", description="UltraRAG CLI")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    p_val = subparsers.add_parser("build", help="Build the configuration")
    p_val.add_argument("config")

    p_run = subparsers.add_parser(
        "run", help="Run the pipeline with the given configuration"
    )
    p_run.add_argument("config")
    p_run.add_argument(
        "--param",
        type=str,
        help="Custom parameter file path",
    )

    p_run.add_argument(
        "--log_level",
        type=str,
        default="info",
        help="Set the logging level (debug, info, warn, error)",
    )
    p_run.add_argument(
        "--is_demo",
        action="store_true",
        help="Enable Demo Mode",
    )
    p_val.add_argument(
        "--log_level",
        type=str,
        default="info",
        help="Set the logging level (debug, info, warn, error)",
    )

    p_show = subparsers.add_parser("show", help="Show UI interface")
    show_sub = p_show.add_subparsers(dest="show_target", required=True)
    p_show_ui = show_sub.add_parser("ui", help="Launch the UltraRAG web UI")
    p_show_ui.add_argument("--host", default="127.0.0.1")
    p_show_ui.add_argument("--port", type=int, default=5050)
    p_show_ui.add_argument(
        "--admin",
        action="store_true",
        help="Launch full admin UI with pipeline builder (default: chat-only mode)",
    )
    p_show.add_argument(
        "--log_level",
        type=str,
        default="info",
        help="Set the logging level (debug, info, warn, error)",
    )

    global log_level, logger
    args = parser.parse_args()
    log_level = args.log_level.lower()
    os.environ["log_level"] = log_level
    logger = get_logger("Client", log_level)

    if args.cmd == "build":
        log_server_banner("Building")
        asyncio.run(build(args.config))
    elif args.cmd == "run":
        asyncio.run(run(args.config, args.param, is_demo=args.is_demo))
    elif args.cmd == "show":
        if args.show_target == "ui":
            launch_ui(host=args.host, port=args.port, admin_mode=args.admin)
        else:
            parser.print_help()
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
