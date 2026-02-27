from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Any, Dict, List, Union

from fastmcp.exceptions import ToolError
from ultrarag.server import UltraRAG_MCP_Server

app = UltraRAG_MCP_Server("memory")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
USER_MEMORY_ROOT = PROJECT_ROOT / "data" / "user_memory"
MEMORY_TEMPLATE = "# MEMORY\ni am jack. i like LLMs.\n"
USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _normalize_user_id(user_id: str | None) -> str:
    normalized = str(user_id or "default").strip() or "default"
    if not USER_ID_PATTERN.fullmatch(normalized):
        raise ToolError("Invalid user_id format. Only letters, numbers, '_' and '-' are allowed.")
    return normalized


def _ensure_user_memory_paths(user_id: str) -> tuple[Path, Path]:
    user_dir = USER_MEMORY_ROOT / user_id
    project_dir = user_dir / "project"
    memory_file = user_dir / "MEMORY.md"

    project_dir.mkdir(parents=True, exist_ok=True)
    if not memory_file.exists():
        memory_file.write_text(MEMORY_TEMPLATE, encoding="utf-8")

    return memory_file, project_dir


@app.tool(output="user_id->global_memory_content,current_user_id")
def get_global_memory(user_id: str = "default") -> Dict[str, str]:
    """Read global memory from MEMORY.md.

    Args:
        user_id: User identifier. Defaults to "default".

    Returns:
        Dictionary containing global memory content.
    """
    normalized_user_id = _normalize_user_id(user_id)
    memory_file, _ = _ensure_user_memory_paths(normalized_user_id)
    profile_content = memory_file.read_text(encoding="utf-8")

    return {
        "global_memory_content": profile_content,
        "current_user_id": normalized_user_id
    }


@app.tool(output="user_id,q_ls,ans_ls->None")
def save_memory(
    user_id: str,
    q_ls: List[str],
    ans_ls: List[str],
):
    """Save one round of user-assistant dialogue into daily project memory.

    Args:
        user_id: User identifier. Defaults to "default".
        q_ls: Current round question.
        ans_ls: Current round answer.

    Returns:
        None.
    """
    normalized_user_id = _normalize_user_id(user_id)

    user_text = str(q_ls[0] or "").strip()
    assistant_text = str(ans_ls[0] or "").strip()

    if not user_text:
        raise ToolError("user_message cannot be empty.")
    if not assistant_text:
        raise ToolError("assistant_message cannot be empty.")

    _, project_dir = _ensure_user_memory_paths(normalized_user_id)

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    daily_file = project_dir / f"{date_str}.md"

    entry = (
        f"\n## {date_str} {time_str}\n"
        f"- user: {user_text}\n"
        f"- assistant: {assistant_text}\n"
    )
    if not daily_file.exists():
        daily_file.write_text(f"# Project Memory {date_str}\n{entry}", encoding="utf-8")
    else:
        with daily_file.open("a", encoding="utf-8") as f:
            f.write(entry)




if __name__ == "__main__":
    app.run(transport="stdio")
