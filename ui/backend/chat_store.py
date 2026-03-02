from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_CHAT_TITLE = "New Chat"
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class ChatStoreValidationError(ValueError):
    """Raised when payload validation fails."""


class ChatStorePermissionError(PermissionError):
    """Raised when user tries to access another user's chat session."""


class SQLiteChatStore:
    """SQLite-backed chat session/message store."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def db_path(self) -> Path:
        return self._db_path

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT 'New Chat',
                    pipeline_name TEXT,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    meta_json TEXT,
                    message_ts TEXT,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated
                ON chat_sessions(user_id, updated_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session_position
                ON chat_messages(session_id, position ASC)
                """
            )
            conn.commit()

    def list_sessions(self, user_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        normalized_user = self._normalize_user_id(user_id)
        capped_limit = max(1, min(int(limit), 500))
        with self._connect() as conn:
            session_rows = conn.execute(
                """
                SELECT id, user_id, title, pipeline_name, created_at, updated_at
                FROM chat_sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (normalized_user, capped_limit),
            ).fetchall()
            if not session_rows:
                return []

            session_ids = [str(row["id"]) for row in session_rows]
            placeholders = ",".join("?" for _ in session_ids)
            message_rows = conn.execute(
                f"""
                SELECT session_id, position, role, content, meta_json, message_ts
                FROM chat_messages
                WHERE session_id IN ({placeholders})
                ORDER BY session_id ASC, position ASC
                """,
                tuple(session_ids),
            ).fetchall()

        grouped_messages: Dict[str, List[Dict[str, Any]]] = {
            session_id: [] for session_id in session_ids
        }
        for row in message_rows:
            sid = str(row["session_id"])
            grouped_messages.setdefault(sid, []).append(self._row_to_message(row))

        sessions: List[Dict[str, Any]] = []
        for row in session_rows:
            sid = str(row["id"])
            session_dict = self._row_to_session(row)
            session_dict["messages"] = grouped_messages.get(sid, [])
            sessions.append(session_dict)
        return sessions

    def get_session(self, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        normalized_user = self._normalize_user_id(user_id)
        normalized_session = self._normalize_session_id(session_id)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, user_id, title, pipeline_name, created_at, updated_at
                FROM chat_sessions
                WHERE id = ?
                """,
                (normalized_session,),
            ).fetchone()
            if not row:
                return None
            self._ensure_owner(row, normalized_user)

            message_rows = conn.execute(
                """
                SELECT session_id, position, role, content, meta_json, message_ts
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY position ASC
                """,
                (normalized_session,),
            ).fetchall()

        session_dict = self._row_to_session(row)
        session_dict["messages"] = [self._row_to_message(r) for r in message_rows]
        return session_dict

    def upsert_session(self, user_id: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        normalized_user = self._normalize_user_id(user_id)
        session_id = self._normalize_session_id(
            session_data.get("id") or session_data.get("session_id")
        )
        title = self._normalize_title(session_data.get("title"))
        pipeline_name = self._normalize_pipeline(session_data.get("pipeline"))
        updated_at = self._normalize_timestamp_ms(session_data.get("timestamp")) or self._now_ms()
        messages = self._normalize_messages(session_data.get("messages", []))

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT user_id, created_at FROM chat_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if existing:
                self._ensure_owner(existing, normalized_user)
                created_at = int(existing["created_at"])
                conn.execute(
                    """
                    UPDATE chat_sessions
                    SET title = ?, pipeline_name = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (title, pipeline_name, updated_at, session_id),
                )
            else:
                created_at = updated_at
                conn.execute(
                    """
                    INSERT INTO chat_sessions (id, user_id, title, pipeline_name, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, normalized_user, title, pipeline_name, created_at, updated_at),
                )

            conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            if messages:
                now_ms = self._now_ms()
                insert_rows = []
                for idx, message in enumerate(messages):
                    insert_rows.append(
                        (
                            session_id,
                            idx,
                            message["role"],
                            message["text"],
                            json.dumps(message.get("meta", {}), ensure_ascii=False),
                            message.get("timestamp"),
                            now_ms,
                        )
                    )
                conn.executemany(
                    """
                    INSERT INTO chat_messages
                    (session_id, position, role, content, meta_json, message_ts, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    insert_rows,
                )
            conn.commit()

        return {
            "id": session_id,
            "title": title,
            "pipeline": pipeline_name,
            "timestamp": updated_at,
            "messages": messages,
            "created_at": created_at,
            "updated_at": updated_at,
        }

    def rename_session(self, user_id: str, session_id: str, title: str) -> Dict[str, Any]:
        normalized_user = self._normalize_user_id(user_id)
        normalized_session = self._normalize_session_id(session_id)
        normalized_title = self._normalize_title(title)
        updated_at = self._now_ms()

        with self._connect() as conn:
            row = conn.execute(
                "SELECT user_id FROM chat_sessions WHERE id = ?",
                (normalized_session,),
            ).fetchone()
            if not row:
                raise KeyError("session not found")
            self._ensure_owner(row, normalized_user)

            conn.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
                (normalized_title, updated_at, normalized_session),
            )
            conn.commit()

        session = self.get_session(normalized_user, normalized_session)
        if session is None:
            raise KeyError("session not found")
        return session

    def delete_session(self, user_id: str, session_id: str) -> bool:
        normalized_user = self._normalize_user_id(user_id)
        normalized_session = self._normalize_session_id(session_id)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT user_id FROM chat_sessions WHERE id = ?",
                (normalized_session,),
            ).fetchone()
            if not row:
                return False
            self._ensure_owner(row, normalized_user)
            conn.execute("DELETE FROM chat_sessions WHERE id = ?", (normalized_session,))
            conn.commit()
            return True

    def clear_sessions(self, user_id: str) -> int:
        normalized_user = self._normalize_user_id(user_id)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM chat_sessions WHERE user_id = ?",
                (normalized_user,),
            ).fetchone()
            count = int(row["cnt"] if row else 0)
            conn.execute("DELETE FROM chat_sessions WHERE user_id = ?", (normalized_user,))
            conn.commit()
            return count

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    def _normalize_user_id(raw_user_id: Any) -> str:
        user_id = str(raw_user_id or "").strip()
        if not user_id:
            raise ChatStoreValidationError("user_id is required")
        return user_id

    @staticmethod
    def _normalize_session_id(raw_session_id: Any) -> str:
        session_id = str(raw_session_id or "").strip()
        if not SESSION_ID_PATTERN.fullmatch(session_id):
            raise ChatStoreValidationError("invalid session_id")
        return session_id

    @staticmethod
    def _normalize_title(raw_title: Any) -> str:
        title = str(raw_title or "").strip()
        return title or DEFAULT_CHAT_TITLE

    @staticmethod
    def _normalize_pipeline(raw_pipeline: Any) -> Optional[str]:
        if raw_pipeline is None:
            return None
        text = str(raw_pipeline).strip()
        return text or None

    def _normalize_messages(self, raw_messages: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_messages, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").strip() or "assistant"
            text = msg.get("text")
            if text is None:
                text = msg.get("content", "")
            text = str(text or "")
            meta = msg.get("meta", {})
            if not isinstance(meta, dict):
                meta = {}
            timestamp = msg.get("timestamp")
            if timestamp is None:
                timestamp = self._now_iso()
            else:
                timestamp = str(timestamp)
            normalized.append(
                {
                    "role": role,
                    "text": text,
                    "meta": meta,
                    "timestamp": timestamp,
                }
            )
        return normalized

    @staticmethod
    def _normalize_timestamp_ms(raw_timestamp: Any) -> Optional[int]:
        if raw_timestamp is None:
            return None
        if isinstance(raw_timestamp, bool):
            return None
        if isinstance(raw_timestamp, (int, float)):
            return int(raw_timestamp)
        text = str(raw_timestamp).strip()
        if not text:
            return None
        if text.isdigit():
            return int(text)
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000)
        except Exception:
            return None

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": str(row["id"]),
            "title": str(row["title"] or DEFAULT_CHAT_TITLE),
            "pipeline": str(row["pipeline_name"]) if row["pipeline_name"] else None,
            "timestamp": int(row["updated_at"]),
            "created_at": int(row["created_at"]),
            "updated_at": int(row["updated_at"]),
        }

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> Dict[str, Any]:
        meta_raw = row["meta_json"]
        try:
            meta = json.loads(meta_raw) if meta_raw else {}
        except Exception:
            meta = {}
        return {
            "role": str(row["role"] or "assistant"),
            "text": str(row["content"] or ""),
            "meta": meta if isinstance(meta, dict) else {},
            "timestamp": str(row["message_ts"] or ""),
        }

    @staticmethod
    def _ensure_owner(row: sqlite3.Row, user_id: str) -> None:
        row_user = str(row["user_id"] or "").strip()
        if row_user != user_id:
            raise ChatStorePermissionError("forbidden session access")

    @staticmethod
    def _now_ms() -> int:
        return int(datetime.now(timezone.utc).timestamp() * 1000)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
