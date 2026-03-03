from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


TABLE_NAME = "kb_collection_visibility"


class KbVisibilityValidationError(ValueError):
    """Raised when visibility payload is invalid."""


class SQLiteKbVisibilityStore:
    """SQLite-backed knowledge-base visibility mapping store."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def db_path(self) -> Path:
        return self._db_path

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                    collection_name TEXT PRIMARY KEY,
                    owner_user_id TEXT NOT NULL,
                    is_public INTEGER NOT NULL DEFAULT 0,
                    visible_users_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_owner
                ON {TABLE_NAME}(owner_user_id)
                """
            )
            conn.commit()

    def normalize_collection_name(self, raw_name: Any) -> str:
        name = str(raw_name or "").strip()
        if not name:
            raise KbVisibilityValidationError("collection_name is required")
        if "\x00" in name:
            raise KbVisibilityValidationError("collection_name contains invalid character")
        if len(name) > 255:
            raise KbVisibilityValidationError("collection_name is too long")
        return name

    def normalize_user_id(self, raw_user_id: Any) -> str:
        user_id = str(raw_user_id or "").strip()
        if not user_id:
            raise KbVisibilityValidationError("user_id is required")
        if len(user_id) > 128:
            raise KbVisibilityValidationError("user_id is too long")
        return user_id

    def list_shareable_users(self) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT username
                FROM users
                ORDER BY username COLLATE NOCASE ASC
                """
            ).fetchall()
        return [str(row["username"]) for row in rows if row and row["username"]]

    def get_visibility(self, collection_name: Any) -> Optional[Dict[str, Any]]:
        normalized_collection = self.normalize_collection_name(collection_name)
        with self._connect() as conn:
            row = self._fetch_row(conn, normalized_collection)
        if not row:
            return None
        return self._row_to_dict(row)

    def upsert_default_private(
        self, collection_name: Any, owner_user_id: Any
    ) -> Dict[str, Any]:
        normalized_collection = self.normalize_collection_name(collection_name)
        normalized_owner = self.normalize_user_id(owner_user_id)
        now = _utc_now_iso()

        with self._connect() as conn:
            existing = self._fetch_row(conn, normalized_collection)
            if existing:
                # Always enforce default-private ownership for newly indexed collections.
                # This avoids legacy bootstrap rows keeping admin/public ownership.
                conn.execute(
                    f"""
                    UPDATE {TABLE_NAME}
                    SET owner_user_id = ?, is_public = 0, visible_users_json = '[]', updated_at = ?
                    WHERE collection_name = ?
                    """,
                    (normalized_owner, now, normalized_collection),
                )
                conn.commit()
                existing = self._fetch_row(conn, normalized_collection)
            else:
                conn.execute(
                    f"""
                    INSERT INTO {TABLE_NAME}
                    (collection_name, owner_user_id, is_public, visible_users_json, created_at, updated_at)
                    VALUES (?, ?, 0, '[]', ?, ?)
                    """,
                    (normalized_collection, normalized_owner, now, now),
                )
                conn.commit()
                existing = self._fetch_row(conn, normalized_collection)
        if not existing:
            raise RuntimeError("failed to create collection visibility mapping")
        return self._row_to_dict(existing)

    def ensure_legacy_public(
        self, collection_name: Any, owner_user_id: Any
    ) -> Dict[str, Any]:
        normalized_collection = self.normalize_collection_name(collection_name)
        normalized_owner = self.normalize_user_id(owner_user_id)
        now = _utc_now_iso()

        with self._connect() as conn:
            existing = self._fetch_row(conn, normalized_collection)
            if not existing:
                conn.execute(
                    f"""
                    INSERT INTO {TABLE_NAME}
                    (collection_name, owner_user_id, is_public, visible_users_json, created_at, updated_at)
                    VALUES (?, ?, 1, '[]', ?, ?)
                    """,
                    (normalized_collection, normalized_owner, now, now),
                )
                conn.commit()
                existing = self._fetch_row(conn, normalized_collection)
        if not existing:
            raise RuntimeError("failed to ensure legacy public visibility mapping")
        return self._row_to_dict(existing)

    def bootstrap_legacy_public(
        self, collection_names: Iterable[Any], owner_user_id: Any
    ) -> int:
        normalized_owner = self.normalize_user_id(owner_user_id)
        candidates: List[str] = []
        seen = set()
        for raw_name in collection_names:
            try:
                normalized = self.normalize_collection_name(raw_name)
            except KbVisibilityValidationError:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(normalized)

        if not candidates:
            return 0

        now = _utc_now_iso()
        inserted = 0
        with self._connect() as conn:
            existing_names = set(self._fetch_many_rows(conn, candidates).keys())
            missing = [name for name in candidates if name not in existing_names]
            if not missing:
                return 0
            conn.executemany(
                f"""
                INSERT INTO {TABLE_NAME}
                (collection_name, owner_user_id, is_public, visible_users_json, created_at, updated_at)
                VALUES (?, ?, 1, '[]', ?, ?)
                """,
                [(name, normalized_owner, now, now) for name in missing],
            )
            conn.commit()
            inserted = len(missing)
        return inserted

    def set_visibility(
        self,
        collection_name: Any,
        owner_user_id: Any,
        visibility: Any,
        visible_users: Any,
    ) -> Dict[str, Any]:
        normalized_collection = self.normalize_collection_name(collection_name)
        normalized_owner = self.normalize_user_id(owner_user_id)
        normalized_visibility = str(visibility or "private").strip().lower()
        if normalized_visibility not in {"private", "public", "shared"}:
            raise KbVisibilityValidationError(
                "visibility must be one of private/public/shared"
            )

        shared_users = self._normalize_visible_users(visible_users)
        shared_users = [u for u in shared_users if u != normalized_owner]
        if normalized_visibility != "shared":
            shared_users = []
        elif not shared_users:
            # Shared without selected users behaves like private.
            normalized_visibility = "private"

        now = _utc_now_iso()
        is_public = 1 if normalized_visibility == "public" else 0
        visible_users_json = json.dumps(shared_users, ensure_ascii=False)

        with self._connect() as conn:
            existing = self._fetch_row(conn, normalized_collection)
            if existing:
                current_owner = str(existing["owner_user_id"] or "").strip()
                effective_owner = current_owner or normalized_owner
                conn.execute(
                    f"""
                    UPDATE {TABLE_NAME}
                    SET owner_user_id = ?, is_public = ?, visible_users_json = ?, updated_at = ?
                    WHERE collection_name = ?
                    """,
                    (
                        effective_owner,
                        int(is_public),
                        visible_users_json,
                        now,
                        normalized_collection,
                    ),
                )
            else:
                conn.execute(
                    f"""
                    INSERT INTO {TABLE_NAME}
                    (collection_name, owner_user_id, is_public, visible_users_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        normalized_collection,
                        normalized_owner,
                        int(is_public),
                        visible_users_json,
                        now,
                        now,
                    ),
                )
            conn.commit()
            refreshed = self._fetch_row(conn, normalized_collection)

        if not refreshed:
            raise RuntimeError("failed to save collection visibility")
        return self._row_to_dict(refreshed)

    def can_view(self, collection_name: Any, current_user_id: Any) -> bool:
        visibility = self.get_visibility(collection_name)
        if not visibility:
            return False

        current_user = str(current_user_id or "default").strip() or "default"
        if visibility.get("is_public"):
            return True
        if visibility.get("owner_user_id") == current_user:
            return True
        return current_user in set(visibility.get("visible_users") or [])

    def can_manage(self, collection_name: Any, current_user_id: Any) -> bool:
        visibility = self.get_visibility(collection_name)
        if not visibility:
            return False
        current_user = str(current_user_id or "default").strip() or "default"
        owner_user_id = str(visibility.get("owner_user_id") or "").strip()
        return bool(owner_user_id) and owner_user_id == current_user

    def filter_viewable_collections(
        self, collections: List[Dict[str, Any]], current_user_id: Any
    ) -> List[Dict[str, Any]]:
        if not collections:
            return []
        current_user = str(current_user_id or "default").strip() or "default"
        names = [str(item.get("name") or "").strip() for item in collections]

        with self._connect() as conn:
            row_map = self._fetch_many_rows(conn, names)

        visible: List[Dict[str, Any]] = []
        for item in collections:
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            row = row_map.get(name)
            if not row:
                continue
            parsed = self._row_to_dict(row)
            if parsed["is_public"] or parsed["owner_user_id"] == current_user:
                visible_item = dict(item)
                visible_item["visibility"] = parsed["visibility"]
                visible_item["owner_user_id"] = parsed["owner_user_id"]
                visible_item["can_manage"] = parsed["owner_user_id"] == current_user
                visible.append(visible_item)
                continue
            if current_user in set(parsed["visible_users"]):
                visible_item = dict(item)
                visible_item["visibility"] = parsed["visibility"]
                visible_item["owner_user_id"] = parsed["owner_user_id"]
                visible_item["can_manage"] = False
                visible.append(visible_item)
        return visible

    def delete_mapping(self, collection_name: Any) -> bool:
        normalized_collection = self.normalize_collection_name(collection_name)
        with self._connect() as conn:
            cursor = conn.execute(
                f"DELETE FROM {TABLE_NAME} WHERE collection_name = ?",
                (normalized_collection,),
            )
            conn.commit()
            return int(cursor.rowcount or 0) > 0

    def _normalize_visible_users(self, raw_users: Any) -> List[str]:
        if raw_users is None:
            return []
        if not isinstance(raw_users, list):
            raise KbVisibilityValidationError("visible_users must be a list")

        deduplicated: List[str] = []
        seen = set()
        for raw in raw_users:
            user_id = str(raw or "").strip()
            if not user_id:
                continue
            if len(user_id) > 128:
                raise KbVisibilityValidationError("visible user id is too long")
            if user_id in seen:
                continue
            seen.add(user_id)
            deduplicated.append(user_id)
        return deduplicated

    def _fetch_row(
        self, conn: sqlite3.Connection, collection_name: str
    ) -> Optional[sqlite3.Row]:
        return conn.execute(
            f"""
            SELECT
                collection_name,
                owner_user_id,
                is_public,
                visible_users_json,
                created_at,
                updated_at
            FROM {TABLE_NAME}
            WHERE collection_name = ?
            """,
            (collection_name,),
        ).fetchone()

    def _fetch_many_rows(
        self, conn: sqlite3.Connection, collection_names: List[str]
    ) -> Dict[str, sqlite3.Row]:
        if not collection_names:
            return {}

        out: Dict[str, sqlite3.Row] = {}
        chunk_size = 400
        for idx in range(0, len(collection_names), chunk_size):
            chunk = collection_names[idx : idx + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"""
                SELECT
                    collection_name,
                    owner_user_id,
                    is_public,
                    visible_users_json,
                    created_at,
                    updated_at
                FROM {TABLE_NAME}
                WHERE collection_name IN ({placeholders})
                """,
                tuple(chunk),
            ).fetchall()
            for row in rows:
                out[str(row["collection_name"])] = row
        return out

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        visible_users = _parse_visible_users(row["visible_users_json"])
        is_public = bool(int(row["is_public"] or 0))
        visibility = "public" if is_public else ("shared" if visible_users else "private")
        return {
            "collection_name": str(row["collection_name"]),
            "owner_user_id": str(row["owner_user_id"]),
            "is_public": is_public,
            "visible_users": visible_users,
            "visibility": visibility,
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn


def _parse_visible_users(raw_json: Any) -> List[str]:
    if raw_json is None:
        return []
    try:
        data = json.loads(str(raw_json))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out: List[str] = []
    seen = set()
    for item in data:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

