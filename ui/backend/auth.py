from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from werkzeug.security import check_password_hash, generate_password_hash

USERNAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{2,31}$")
MIN_PASSWORD_LENGTH = 6
RESERVED_USERNAMES = {"default"}
ADMIN_USERNAME = "admin"
ADMIN_DEFAULT_PASSWORD = "12345678"
MODEL_SETTING_COLUMNS = (
    "retriever_api_key",
    "retriever_base_url",
    "retriever_model_name",
    "generation_api_key",
    "generation_base_url",
    "generation_model_name",
)


class AuthValidationError(ValueError):
    """Raised when provided auth payload is invalid."""


class UserAlreadyExistsError(ValueError):
    """Raised when trying to create a duplicated username."""


class InvalidCredentialsError(ValueError):
    """Raised when credentials are invalid for a protected auth operation."""


class SQLiteUserStore:
    """Lightweight SQLite-backed user store for local auth."""

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
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    nickname TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._ensure_users_table_columns(conn)
            self._ensure_default_admin_user(conn)
            conn.commit()

    def normalize_username(self, raw_username: Any) -> str:
        username = str(raw_username or "").strip()
        if not USERNAME_PATTERN.fullmatch(username):
            raise AuthValidationError(
                "username must match ^[A-Za-z][A-Za-z0-9_]{2,31}$"
            )
        if username.lower() in RESERVED_USERNAMES:
            raise AuthValidationError("username is reserved")
        return username

    def validate_password(self, raw_password: Any) -> str:
        password = raw_password if isinstance(raw_password, str) else ""
        if len(password) < MIN_PASSWORD_LENGTH:
            raise AuthValidationError(
                f"password must be at least {MIN_PASSWORD_LENGTH} characters"
            )
        return password

    def _normalize_optional_text(self, raw_value: Any) -> Optional[str]:
        text = str(raw_value or "").strip()
        return text or None

    def normalize_nickname(self, raw_nickname: Any) -> Optional[str]:
        return self._normalize_optional_text(raw_nickname)

    def normalize_model_settings(
        self, raw_payload: Any
    ) -> Dict[str, Optional[str]]:
        payload = raw_payload if isinstance(raw_payload, dict) else {}
        retriever_payload = payload.get("retriever")
        generation_payload = payload.get("generation")

        # Backward-compatible flat payload support.
        if retriever_payload is None and generation_payload is None:
            retriever_payload = {
                "api_key": payload.get("retriever_api_key"),
                "base_url": payload.get("retriever_base_url"),
                "model_name": payload.get("retriever_model_name"),
            }
            generation_payload = {
                "api_key": payload.get("generation_api_key"),
                "base_url": payload.get("generation_base_url"),
                "model_name": payload.get("generation_model_name"),
            }

        if retriever_payload is None:
            retriever_payload = {}
        if generation_payload is None:
            generation_payload = {}

        if not isinstance(retriever_payload, dict):
            raise AuthValidationError("retriever model settings must be an object")
        if not isinstance(generation_payload, dict):
            raise AuthValidationError("generation model settings must be an object")

        return {
            "retriever_api_key": self._normalize_optional_text(
                retriever_payload.get("api_key")
            ),
            "retriever_base_url": self._normalize_optional_text(
                retriever_payload.get("base_url")
            ),
            "retriever_model_name": self._normalize_optional_text(
                retriever_payload.get("model_name")
            ),
            "generation_api_key": self._normalize_optional_text(
                generation_payload.get("api_key")
            ),
            "generation_base_url": self._normalize_optional_text(
                generation_payload.get("base_url")
            ),
            "generation_model_name": self._normalize_optional_text(
                generation_payload.get("model_name")
            ),
        }

    def _fetch_user_row(
        self, conn: sqlite3.Connection, username: str
    ) -> Optional[sqlite3.Row]:
        return conn.execute(
            """
            SELECT
                id,
                username,
                password_hash,
                nickname,
                retriever_api_key,
                retriever_base_url,
                retriever_model_name,
                generation_api_key,
                generation_base_url,
                generation_model_name,
                created_at
            FROM users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()

    def create_user(self, raw_username: Any, raw_password: Any) -> Dict[str, Any]:
        username = self.normalize_username(raw_username)
        password = self.validate_password(raw_password)
        password_hash = generate_password_hash(password)
        created_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO users (username, password_hash, nickname, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (username, password_hash, None, created_at),
                )
            except sqlite3.IntegrityError as exc:
                raise UserAlreadyExistsError("username already exists") from exc
            conn.commit()
            user_id = int(cursor.lastrowid or 0)
            refreshed = self._fetch_user_row(conn, username)

        if refreshed:
            return self._row_to_dict(refreshed)
        return {"id": user_id, "username": username, "created_at": created_at}

    def verify_credentials(
        self, raw_username: Any, raw_password: Any
    ) -> Optional[Dict[str, Any]]:
        username = self.normalize_username(raw_username)
        password = raw_password if isinstance(raw_password, str) else ""
        if not password:
            raise AuthValidationError("password is required")

        with self._connect() as conn:
            row = self._fetch_user_row(conn, username)

        if not row:
            return None
        if not check_password_hash(str(row["password_hash"]), password):
            return None
        return self._row_to_dict(row)

    def get_user(self, raw_username: Any) -> Optional[Dict[str, Any]]:
        username = self.normalize_username(raw_username)
        with self._connect() as conn:
            row = self._fetch_user_row(conn, username)
        if not row:
            return None
        return self._row_to_dict(row)

    def is_admin_username(self, raw_username: Any) -> bool:
        return str(raw_username or "").strip() == ADMIN_USERNAME

    def update_password(
        self, raw_username: Any, raw_current_password: Any, raw_new_password: Any
    ) -> Dict[str, Any]:
        username = self.normalize_username(raw_username)
        current_password = (
            raw_current_password if isinstance(raw_current_password, str) else ""
        )
        if not current_password:
            raise AuthValidationError("current password is required")
        new_password = self.validate_password(raw_new_password)
        if current_password == new_password:
            raise AuthValidationError(
                "new password must be different from current password"
            )

        with self._connect() as conn:
            row = self._fetch_user_row(conn, username)
            if not row or not check_password_hash(
                str(row["password_hash"]), current_password
            ):
                raise InvalidCredentialsError("invalid current password")

            conn.execute(
                """
                UPDATE users
                SET password_hash = ?
                WHERE username = ?
                """,
                (generate_password_hash(new_password), username),
            )
            conn.commit()

            refreshed = self._fetch_user_row(conn, username)

        if not refreshed:
            raise InvalidCredentialsError("invalid current password")
        return self._row_to_dict(refreshed)

    def update_nickname(self, raw_username: Any, raw_nickname: Any) -> Dict[str, Any]:
        username = self.normalize_username(raw_username)
        nickname = self.normalize_nickname(raw_nickname)

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE users
                SET nickname = ?
                WHERE username = ?
                """,
                (nickname, username),
            )
            conn.commit()
            refreshed = self._fetch_user_row(conn, username)

        if not refreshed:
            raise AuthValidationError("user not found")
        return self._row_to_dict(refreshed)

    def update_model_settings(
        self, raw_username: Any, raw_payload: Any
    ) -> Dict[str, Any]:
        username = self.normalize_username(raw_username)
        settings = self.normalize_model_settings(raw_payload)

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE users
                SET
                    retriever_api_key = ?,
                    retriever_base_url = ?,
                    retriever_model_name = ?,
                    generation_api_key = ?,
                    generation_base_url = ?,
                    generation_model_name = ?
                WHERE username = ?
                """,
                (
                    settings["retriever_api_key"],
                    settings["retriever_base_url"],
                    settings["retriever_model_name"],
                    settings["generation_api_key"],
                    settings["generation_base_url"],
                    settings["generation_model_name"],
                    username,
                ),
            )
            conn.commit()
            refreshed = self._fetch_user_row(conn, username)

        if not refreshed:
            raise AuthValidationError("user not found")
        return self._row_to_dict(refreshed)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_users_table_columns(self, conn: sqlite3.Connection) -> None:
        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(users)").fetchall()
        }
        if "nickname" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN nickname TEXT")
        for column in MODEL_SETTING_COLUMNS:
            if column not in columns:
                conn.execute(f"ALTER TABLE users ADD COLUMN {column} TEXT")

    def _ensure_default_admin_user(self, conn: sqlite3.Connection) -> None:
        existing = conn.execute(
            """
            SELECT 1
            FROM users
            WHERE username = ?
            LIMIT 1
            """,
            (ADMIN_USERNAME,),
        ).fetchone()
        if existing:
            return

        conn.execute(
            """
            INSERT INTO users (username, password_hash, nickname, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                ADMIN_USERNAME,
                generate_password_hash(ADMIN_DEFAULT_PASSWORD),
                None,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        def _clean_optional(row_key: str) -> Optional[str]:
            value = row[row_key] if row_key in row.keys() else None
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        nickname_text = _clean_optional("nickname")
        return {
            "id": int(row["id"]),
            "username": str(row["username"]),
            "password_hash": str(row["password_hash"]),
            "nickname": nickname_text or None,
            "model_settings": {
                "retriever": {
                    "api_key": _clean_optional("retriever_api_key"),
                    "base_url": _clean_optional("retriever_base_url"),
                    "model_name": _clean_optional("retriever_model_name"),
                },
                "generation": {
                    "api_key": _clean_optional("generation_api_key"),
                    "base_url": _clean_optional("generation_base_url"),
                    "model_name": _clean_optional("generation_model_name"),
                },
            },
            "created_at": str(row["created_at"]),
        }
