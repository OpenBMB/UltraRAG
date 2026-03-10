from __future__ import annotations

import os
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

UI_STORAGE_ENV_VAR: Final[str] = "ULTRARAG_UI_STORAGE_ROOT"
DEFAULT_UI_STORAGE_ROOT: Final[Path] = PROJECT_ROOT / "ui" / "storage"


def _resolve_ui_storage_root() -> Path:
    raw_value = str(os.getenv(UI_STORAGE_ENV_VAR, "")).strip()
    if not raw_value:
        return DEFAULT_UI_STORAGE_ROOT

    configured = Path(raw_value).expanduser()
    if not configured.is_absolute():
        configured = PROJECT_ROOT / configured
    return configured.resolve()


UI_STORAGE_ROOT: Final[Path] = _resolve_ui_storage_root()

UI_DB_DIR: Final[Path] = UI_STORAGE_ROOT / "db"
UI_USERS_DB_PATH: Final[Path] = UI_DB_DIR / "users.sqlite3"

UI_CHAT_SESSIONS_DIR: Final[Path] = UI_STORAGE_ROOT / "chat_sessions"

UI_KB_ROOT_DIR: Final[Path] = UI_STORAGE_ROOT / "knowledge_base"
UI_KB_RAW_DIR: Final[Path] = UI_KB_ROOT_DIR / "raw"
UI_KB_CORPUS_DIR: Final[Path] = UI_KB_ROOT_DIR / "corpus"
UI_KB_CHUNKS_DIR: Final[Path] = UI_KB_ROOT_DIR / "chunks"
UI_KB_INDEX_DIR: Final[Path] = UI_KB_ROOT_DIR / "index"
UI_KB_CONFIG_PATH: Final[Path] = UI_KB_ROOT_DIR / "kb_config.json"
UI_MEMORY_SYNC_WORKDIR: Final[Path] = UI_KB_ROOT_DIR / "_memory_sync"

UI_MEMORY_ROOT_DIR: Final[Path] = UI_STORAGE_ROOT / "memory"
UI_EXT_DIR: Final[Path] = UI_STORAGE_ROOT / "ext"

UI_STORAGE_DIRS: Final[tuple[Path, ...]] = (
    UI_STORAGE_ROOT,
    UI_DB_DIR,
    UI_CHAT_SESSIONS_DIR,
    UI_KB_RAW_DIR,
    UI_KB_CORPUS_DIR,
    UI_KB_CHUNKS_DIR,
    UI_KB_INDEX_DIR,
    UI_MEMORY_SYNC_WORKDIR,
    UI_MEMORY_ROOT_DIR,
    UI_EXT_DIR,
)


def ensure_ui_storage_dirs() -> None:
    for path in UI_STORAGE_DIRS:
        path.mkdir(parents=True, exist_ok=True)
