import json
import random
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from fastmcp.exceptions import NotFoundError, ToolError
from ultrarag.server import UltraRAG_MCP_Server


app = UltraRAG_MCP_Server("benchmark")


def _load_data_from_file(
    path: str | Path,
    limit: int,
) -> List[Dict[str, Any]]:
    """Load data from file in various formats (jsonl, json, parquet).

    Args:
        path: Path to the data file
        limit: Maximum number of records to load. -1 means no limit, 0 is invalid.

    Returns:
        List of dictionaries containing the loaded data

    Raises:
        ToolError: If file format is not supported
    """
    # Convert Path object to string for string operations
    path_str = str(path)
    data = []
    if path_str.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit and limit > 0:
                    break
                data.append(json.loads(line))
    elif path_str.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if limit > 0:
                data = data[:limit]
    elif path_str.endswith(".parquet"):
        df = pd.read_parquet(path)
        data = df.to_dict(orient="records")
        if limit > 0:
            data = data[:limit]
    else:
        app.logger.error(
            f"Unsupported file format: ({path_str}). Supported: .jsonl, .json, .parquet"
        )
        raise ToolError(
            f"Unsupported file format: ({path_str}). Supported: .jsonl, .json, .parquet"
        )

    app.logger.info(f"Loaded from {path_str}")
    app.logger.debug(f"_load_data_from_file data: {data}")
    return data


def _load_from_local(
    path: str,
    key_map: Dict[str, str],
    limit: int,
    is_shuffle: bool = False,
    seed: int = 42,
) -> Dict[str, List[Any]]:
    """Load data from local file and map keys according to key_map.

    Args:
        path: Path to the data file
        key_map: Dictionary mapping alias keys to original keys in the data
        limit: Maximum number of records to load. -1 means no limit.
        is_shuffle: Whether to shuffle the data before limiting
        seed: Random seed for shuffling

    Returns:
        Dictionary with mapped keys containing lists of values
    """
    # Load all data if shuffling, otherwise load with limit
    data = _load_data_from_file(path, -1 if is_shuffle else limit)
    ret: Dict[str, List[Any]] = {}
    for alias, original_key in key_map.items():
        ret[alias] = [item[original_key] for item in data if original_key in item]

    if is_shuffle:
        # Check if ret is empty before accessing values
        if not ret:
            app.logger.warning("No data found after key mapping")
            return ret

        length = len(next(iter(ret.values())))
        idx = list(range(length))
        random.seed(seed)
        random.shuffle(idx)
        idx = idx if limit == -1 else idx[:limit]
        for k in ret:
            ret[k] = [ret[k][i] for i in idx]
    else:
        if limit != -1:
            for k in ret:
                ret[k] = ret[k][:limit]

    app.logger.debug(ret)
    return ret


@app.tool(output="benchmark->q_ls,gt_ls")
def get_data(
    benchmark: Dict[str, Any],
) -> Dict[str, List[Any]]:
    """Load benchmark data from file with key mapping and optional shuffling.

    Args:
        benchmark: Dictionary containing:
            - path: Path to the data file (required)
            - key_map: Dictionary mapping alias keys to original keys (required)
            - shuffle: Whether to shuffle the data (default: False)
            - seed: Random seed for shuffling (default: 42)
            - limit: Maximum number of records to load, -1 for no limit (default: -1)
            - name: Name of the benchmark (optional)

    Returns:
        Dictionary with mapped keys containing lists of values

    Raises:
        NotFoundError: If path is missing or invalid
        ToolError: If key_map is invalid or limit is invalid
    """
    app.logger.info(f"Loading data: {benchmark.get('path')}")

    path = benchmark.get("path")
    key_map = benchmark.get("key_map", {})
    is_shuffle = benchmark.get("shuffle", False)
    seed = benchmark.get("seed", 42)
    limit = benchmark.get("limit", -1)

    if not path:
        err_msg = f"Benchmark path: {path} is required"
        app.logger.error(err_msg)
        raise NotFoundError(err_msg)

    if not isinstance(key_map, dict):
        err_msg = f"Benchmark parameter key_map: {key_map} must be a dictionary"
        app.logger.error(err_msg)
        raise ToolError(err_msg)

    if not key_map:
        err_msg = (
            f"Benchmark parameter key_map: {key_map} must contain at least one key"
        )
        app.logger.error(err_msg)
        raise ToolError(err_msg)

    if not isinstance(limit, int) or limit < -1:
        err_msg = (
            f"Benchmark parameter limit: {limit} must be a non-negative integer or -1"
        )
        app.logger.error(err_msg)
        raise ToolError(err_msg)

    if limit == 0:
        err_msg = f"Benchmark parameter limit: {limit} cannot be 0"
        app.logger.error(err_msg)
        raise ToolError(err_msg)

    data = _load_from_local(path, key_map, limit, is_shuffle, seed)

    app.logger.info(
        f"Loaded benchmark: name={benchmark.get('name')}, path={benchmark.get('path')}"
    )
    app.logger.debug(f"Benchmark: {data}")
    return data


if __name__ == "__main__":
    app.run(transport="stdio")
