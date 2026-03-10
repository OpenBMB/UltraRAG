from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .storage_paths import (
    LEGACY_CHAT_SESSIONS_DIR,
    LEGACY_DATA_ROOT,
    LEGACY_KB_ROOT_DIR,
    LEGACY_MEMORY_ROOT_DIR,
    LEGACY_USERS_DB_PATH,
    UI_CHAT_SESSIONS_DIR,
    UI_KB_ROOT_DIR,
    UI_MEMORY_ROOT_DIR,
    UI_STORAGE_MIGRATION_MARKER,
    UI_STORAGE_ROOT,
    UI_USERS_DB_PATH,
    ensure_ui_storage_dirs,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _MigrationItem:
    name: str
    source: Path
    target: Path


_MIGRATION_ITEMS: tuple[_MigrationItem, ...] = (
    _MigrationItem("users_db", LEGACY_USERS_DB_PATH, UI_USERS_DB_PATH),
    _MigrationItem("chat_sessions", LEGACY_CHAT_SESSIONS_DIR, UI_CHAT_SESSIONS_DIR),
    _MigrationItem("knowledge_base", LEGACY_KB_ROOT_DIR, UI_KB_ROOT_DIR),
    _MigrationItem("memory", LEGACY_MEMORY_ROOT_DIR, UI_MEMORY_ROOT_DIR),
)


def _path_has_content(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return True
    return any(path.iterdir())


def _dir_stats(path: Path) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    for item in path.rglob("*"):
        if item.is_file():
            file_count += 1
            total_bytes += item.stat().st_size
    return file_count, total_bytes


def _validate_copy(source: Path, target: Path) -> None:
    if source.is_file():
        src_size = source.stat().st_size
        tgt_size = target.stat().st_size
        if src_size != tgt_size:
            raise RuntimeError(
                f"File size mismatch after copy: {source} ({src_size}) != {target} ({tgt_size})"
            )
        return

    src_stats = _dir_stats(source)
    tgt_stats = _dir_stats(target)
    if src_stats != tgt_stats:
        raise RuntimeError(
            f"Directory stats mismatch after copy: {source} {src_stats} != {target} {tgt_stats}"
        )


def _copy_item(item: _MigrationItem) -> Dict[str, Any]:
    source = item.source
    target = item.target

    record: Dict[str, Any] = {
        "name": item.name,
        "source": str(source),
        "target": str(target),
        "copy_status": "missing",
        "backup_status": "not_applicable",
    }

    if not source.exists():
        return record

    if target.exists() and _path_has_content(target):
        record["copy_status"] = "skipped_existing_target"
        return record

    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        shutil.copy2(source, target)
    else:
        if target.exists():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copytree(source, target)

    _validate_copy(source, target)
    record["copy_status"] = "copied"
    return record


def _ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    index = 1
    while True:
        candidate = path.with_name(f"{path.name}_{index}")
        if not candidate.exists():
            return candidate
        index += 1


def _backup_legacy_sources(
    records: List[Dict[str, Any]],
    logger: logging.Logger,
) -> str:
    sources_to_move = [
        Path(record["source"])
        for record in records
        if Path(record["source"]).exists()
    ]
    if not sources_to_move:
        return ""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = _ensure_unique_path(LEGACY_DATA_ROOT / f"_ui_backup_{timestamp}")
    backup_root.mkdir(parents=True, exist_ok=True)

    for record in records:
        source = Path(record["source"])
        if not source.exists():
            continue

        destination = _ensure_unique_path(backup_root / source.name)
        shutil.move(str(source), str(destination))
        record["backup_status"] = "moved"
        record["backup_target"] = str(destination)
        logger.info("Moved legacy UI data: %s -> %s", source, destination)

    return str(backup_root)


def _write_marker(payload: Dict[str, Any]) -> None:
    UI_STORAGE_MIGRATION_MARKER.parent.mkdir(parents=True, exist_ok=True)
    UI_STORAGE_MIGRATION_MARKER.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def migrate_ui_storage_with_backup(
    logger: logging.Logger | None = None,
) -> Dict[str, Any]:
    """Migrate legacy UI runtime data from data/* to ui/storage/* once."""
    active_logger = logger or LOGGER
    ensure_ui_storage_dirs()

    if UI_STORAGE_MIGRATION_MARKER.exists():
        existing_payload: Dict[str, Any] = {}
        try:
            existing_payload = json.loads(
                UI_STORAGE_MIGRATION_MARKER.read_text(encoding="utf-8")
            )
        except Exception:
            existing_payload = {}
        return {
            "status": "already_migrated",
            "marker_path": str(UI_STORAGE_MIGRATION_MARKER),
            "marker": existing_payload,
        }

    records = [_copy_item(item) for item in _MIGRATION_ITEMS]
    backup_root = _backup_legacy_sources(records, active_logger)

    summary = {
        "status": "migrated" if backup_root else "no_legacy_data",
        "migrated_at": datetime.now(timezone.utc).isoformat(),
        "ui_storage_root": str(UI_STORAGE_ROOT),
        "backup_root": backup_root,
        "records": records,
    }
    _write_marker(summary)
    return summary
