from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
import time
import queue
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from types import ModuleType
import yaml
import ast
import os
import uuid
import shutil
import re
import unicodedata
import hashlib
from datetime import datetime

try:
    from pymilvus import MilvusClient
except ImportError:
    MilvusClient = None

LOGGER = logging.getLogger(__name__)

# Suppress noisy "Event loop is closed" errors triggered by fastmcp transport
try:
    from fastmcp.client.transports import StdioTransport  # type: ignore

    _orig_stdio_del = getattr(StdioTransport, "__del__", None)

    def _safe_stdio_del(self):
        try:
            if _orig_stdio_del:
                _orig_stdio_del(self)
        except RuntimeError as exc:  # pragma: no cover - best effort guard
            if "Event loop is closed" in str(exc):
                LOGGER.debug("Suppressed closed-loop warning in StdioTransport.__del__")
                return
            raise
        except Exception as exc:  # pragma: no cover - best effort guard
            LOGGER.debug("Suppressed StdioTransport.__del__ error: %s", exc)

    if _orig_stdio_del:
        StdioTransport.__del__ = _safe_stdio_del  # type: ignore
except Exception as exc:  # pragma: no cover - defensive
    LOGGER.warning("Failed to patch StdioTransport destructor: %s", exc)

# Suppress noisy "Event loop is closed" stack traces from fakeredis callbacks
try:
    import fakeredis._helpers as _fr_helpers

    def _safe_notify_watch(self, key):  # type: ignore[no-untyped-def]
        for sock in list(self._watches.get(key, set())):
            try:
                sock.notify_watch()
            except RuntimeError as exc:
                if "Event loop is closed" in str(exc):
                    LOGGER.debug("Suppressing fakeredis watch callback on closed loop")
                    continue
                raise
        self.condition.notify_all()
        for callback in list(self._change_callbacks):
            try:
                callback()
            except RuntimeError as exc:
                if "Event loop is closed" in str(exc):
                    LOGGER.debug("Suppressing fakeredis change callback on closed loop")
                    continue
                raise
            except Exception as exc:
                LOGGER.debug("Suppressing fakeredis change callback error: %s", exc)

    _fr_helpers.Database.notify_watch = _safe_notify_watch  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover - defensive
    LOGGER.debug("Skipping fakeredis notify_watch patch: %s", exc)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

BASE_DIR = Path(__file__).resolve().parent.parent
SERVERS_DIR = PROJECT_ROOT / "servers"
PIPELINES_DIR = PROJECT_ROOT / "examples"
LEGACY_PIPELINES_DIR = BASE_DIR / "pipelines"
CHAT_DATASET_DIR = PROJECT_ROOT / "data" / "chat_sessions"
OUTPUT_DIR = PROJECT_ROOT / "output"

KB_ROOT = PROJECT_ROOT / "data" / "knowledge_base"
KB_RAW_DIR = KB_ROOT / "raw"
KB_CORPUS_DIR = KB_ROOT / "corpus"
KB_CHUNKS_DIR = KB_ROOT / "chunks"
KB_INDEX_DIR = KB_ROOT / "index"
KB_CONFIG_PATH = KB_ROOT / "kb_config.json"
MAX_COLLECTION_NAME_LEN = 255

# Default session keep-alive duration, can be overridden via environment variable;
# Set to <=0 to disable automatic expiration
SESSION_TIMEOUT_SECONDS = int(os.getenv("ULTRARAG_SESSION_TIMEOUT", "86400"))
BACKGROUND_SESSION_TIMEOUT_SECONDS = int(
    os.getenv("ULTRARAG_BG_SESSION_TIMEOUT", "172800")
)


def _secure_filename_unicode(filename: str) -> str:
    """Sanitize filename, preserving Unicode characters (e.g., Chinese) but removing dangerous characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    filename = unicodedata.normalize("NFC", filename)
    # Remove path separators and dangerous characters
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", filename)
    # Remove leading/trailing whitespace and dots
    filename = filename.strip().strip(".")
    return filename


def _normalize_collection_name(raw_name: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(raw_name or "")).strip()
    normalized = re.sub(r"[^\w]+", "_", normalized, flags=re.UNICODE)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized or not re.search(r"[A-Za-z0-9]", normalized):
        return ""
    return normalized[:MAX_COLLECTION_NAME_LEN]


def _normalize_display_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(name or "")).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _make_safe_collection_name(display_name: str) -> tuple[str, str]:
    base = _normalize_collection_name(display_name)

    if not base:
        digest = hashlib.sha1(display_name.encode("utf-8")).hexdigest()[:8]
        base = f"kb_{digest}"

    safe_name = base

    return safe_name, display_name


def _transliterate_name(name: str) -> str:
    """Convert arbitrary input to ASCII slug usable for collection_name.

    - Prefer pypinyin for transliteration; if unavailable, use Unicode decomposition
      and remove non-ASCII characters.

    Args:
        name: Input name string

    Returns:
        ASCII slug string
    """
    raw = str(name or "").strip()
    if not raw:
        return ""
    candidate = ""
    try:
        from pypinyin import lazy_pinyin  # type: ignore

        candidate = "_".join(lazy_pinyin(raw))
    except Exception:
        normalized = unicodedata.normalize("NFKD", raw)
        candidate = (
            normalized.encode("ascii", "ignore").decode("ascii") if normalized else ""
        )

    candidate = re.sub(r"[^\w]+", "_", candidate, flags=re.UNICODE)
    candidate = re.sub(r"_+", "_", candidate).strip("_")
    if candidate and not re.match(r"[A-Za-z_]", candidate[0]):
        candidate = f"kb_{candidate}"
    if not candidate:
        candidate = "kb"
    return candidate[:MAX_COLLECTION_NAME_LEN]


def _make_unique_name(base: str, taken: set[str]) -> str:
    if base not in taken:
        return base
    i = 1
    while True:
        candidate = f"{base}_{i}"
        if candidate not in taken:
            return candidate[:MAX_COLLECTION_NAME_LEN]
        i += 1


def _make_unique_display(name: str, taken: set[str]) -> str:
    if name not in taken:
        return name
    i = 1
    while True:
        candidate = f"{name} ({i})"
        if candidate not in taken:
            return candidate
        i += 1


def _extract_display_name_from_desc(desc: str, fallback: str) -> str:
    """Extract display name from Milvus collection description.

    Expected format: 'UltraRAG KB | display_name=<original_name>'

    Args:
        desc: Collection description string
        fallback: Fallback name if extraction fails

    Returns:
        Extracted display name or fallback
    """
    if not desc:
        return fallback
    marker = "display_name="
    if marker in desc:
        try:
            part = desc.split(marker, 1)[1]
            # Remove possible surrounding separators
            part = part.split("|", 1)[0].strip()
            if part:
                return part
        except Exception:
            pass
    return fallback


for d in [
    LEGACY_PIPELINES_DIR,
    CHAT_DATASET_DIR,
    OUTPUT_DIR,
    KB_RAW_DIR,
    KB_CORPUS_DIR,
    KB_CHUNKS_DIR,
    KB_INDEX_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)


class PipelineManagerError(RuntimeError):
    pass


STUBBED_MODULES: set[str] = set()

# Client Function Loader (Lazy Import)
_client_funcs = {}


def _ensure_client_funcs():
    if _client_funcs:
        return _client_funcs

    try:
        from ultrarag.client import (
            build,
            load_pipeline_context,
            create_mcp_client,
            execute_pipeline,
        )

        _client_funcs["build"] = build
        _client_funcs["load_ctx"] = load_pipeline_context
        _client_funcs["create_client"] = create_mcp_client
        _client_funcs["exec_pipe"] = execute_pipeline
        return _client_funcs
    except ModuleNotFoundError as exc:
        msg = f"Missing dependency: {exc.name}"
        LOGGER.error(msg)
        raise PipelineManagerError(msg) from exc


# Session Management (Multi-Client Support)
class DemoSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._bg_loop, daemon=True)
        self._client = None
        self._context = None
        self._active = False
        self.last_accessed = time.time()
        self._thread.start()
        self._current_future = None

        # Multi-turn conversation support
        self._conversation_history: List[Dict[str, str]] = (
            []
        )  # Conversation history (empty means first question)
        self._pipeline_name: Optional[str] = None  # Current pipeline name

        # Multi-turn conversation dedicated MCP client and context
        self._multiturn_client = None
        self._multiturn_context = None
        self._multiturn_active = False

    def _bg_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def touch(self):
        self.last_accessed = time.time()

    def start(self, name: str):
        self.touch()
        if self._active:
            return

        funcs = _ensure_client_funcs()
        config_file = _find_pipeline_file(name)
        if not config_file:
            raise PipelineManagerError(f"Pipeline {name} not found")

        param_path = _resolve_parameter_path(name, for_write=False)
        self._context = funcs["load_ctx"](str(config_file), str(param_path))
        self._client = funcs["create_client"](self._context["mcp_cfg"])

        future = asyncio.run_coroutine_threadsafe(self._client.__aenter__(), self._loop)
        try:
            future.result(timeout=15)
            self._active = True
            self._pipeline_name = name  # Record pipeline name
            LOGGER.info(f"Session {self.session_id} connected for pipeline {name}")
        except Exception as e:
            LOGGER.error(f"Session start failed: {e}")
            raise PipelineManagerError(f"Connection failed: {e}")

    def stop(self):
        self.touch()
        if self._active and self._client:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._client.__aexit__(None, None, None), self._loop
                ).result(timeout=5)
            except Exception as e:
                LOGGER.warning(f"Error during disconnect: {e}")

        # Clean up multi-turn conversation client
        if self._multiturn_active and self._multiturn_client:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._multiturn_client.__aexit__(None, None, None), self._loop
                ).result(timeout=5)
            except Exception as e:
                LOGGER.warning(f"Error during multiturn disconnect: {e}")

        self._active = False
        self._multiturn_active = False
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=1)

        # Reset conversation state
        self._conversation_history = []

        LOGGER.info(f"Session {self.session_id} stopped")

    def add_to_history(self, role: str, content: str) -> None:
        """Add message to conversation history.

        Args:
            role: Message role ("user" or "assistant")
            content: Message content
        """
        if role in ("user", "assistant") and content:
            self._conversation_history.append({"role": role, "content": str(content)})
            LOGGER.debug(
                f"Session {self.session_id} history updated: "
                f"{len(self._conversation_history)} messages"
            )

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history.

        Returns:
            List of conversation messages
        """
        return self._conversation_history.copy()

    def clear_history(self) -> None:
        """Clear conversation history, making next question run full pipeline again."""
        self._conversation_history = []
        LOGGER.info(f"Session {self.session_id} conversation history cleared")

    def is_first_turn(self) -> bool:
        """Check if this is the first turn - based on whether conversation history is empty.

        Returns:
            True if first turn, False otherwise
        """
        return len(self._conversation_history) == 0

    def mark_first_turn_done(self) -> None:
        """Mark first turn as done (deprecated, kept for compatibility).

        No longer uses _is_first_turn flag, now determined by conversation history.
        """
        pass

    def init_multiturn_client(self) -> None:
        """Initialize multi-turn conversation MCP client."""
        if self._multiturn_active:
            return

        funcs = _ensure_client_funcs()
        multiturn_config_file = _find_pipeline_file("multiturn_chat")
        if not multiturn_config_file:
            LOGGER.warning(
                "multiturn_chat pipeline not found, will fall back to full pipeline"
            )
            return

        try:
            multiturn_param_path = _resolve_parameter_path(
                "multiturn_chat", for_write=False
            )
            self._multiturn_context = funcs["load_ctx"](
                str(multiturn_config_file), str(multiturn_param_path)
            )
            self._multiturn_client = funcs["create_client"](
                self._multiturn_context["mcp_cfg"]
            )

            future = asyncio.run_coroutine_threadsafe(
                self._multiturn_client.__aenter__(), self._loop
            )
            future.result(timeout=15)
            self._multiturn_active = True
            LOGGER.info(f"Session {self.session_id} multiturn client initialized")
        except Exception as e:
            LOGGER.warning(
                f"Failed to init multiturn client: {e}, will fall back to full pipeline"
            )
            self._multiturn_active = False

    def run_multiturn_chat(
        self, callback: Any, dynamic_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute multi-turn conversation generation.

        Args:
            callback: Stream callback function
            dynamic_params: Optional dynamic parameters

        Returns:
            Pipeline execution result

        Raises:
            PipelineManagerError: If multiturn client is not active
        """
        self.touch()
        if not self._multiturn_active:
            raise PipelineManagerError("Multiturn client not active")

        funcs = _ensure_client_funcs()

        # Build override_params, passing conversation history
        override_params = dynamic_params.copy() if dynamic_params else {}

        # Inject messages into generation parameters (as global variable)
        # We need to set messages before execute_pipeline
        if "generation" not in override_params:
            override_params["generation"] = {}

        self._current_future = asyncio.run_coroutine_threadsafe(
            funcs["exec_pipe"](
                self._multiturn_client,
                self._multiturn_context,
                is_demo=True,
                return_all=True,
                stream_callback=callback,
                override_params=override_params,
            ),
            self._loop,
        )

        try:
            return self._current_future.result()
        except asyncio.CancelledError:
            LOGGER.info(f"Session {self.session_id} multiturn task cancelled by user.")
            raise
        finally:
            self._current_future = None

    def run_chat(
        self, callback: Any, dynamic_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Run chat pipeline execution.

        Args:
            callback: Stream callback function
            dynamic_params: Optional dynamic parameters

        Returns:
            Pipeline execution result

        Raises:
            PipelineManagerError: If session is not active
        """
        self.touch()
        if not self._active:
            raise PipelineManagerError("Session not active")

        funcs = _ensure_client_funcs()
        self._current_future = asyncio.run_coroutine_threadsafe(
            funcs["exec_pipe"](
                self._client,
                self._context,
                is_demo=True,
                return_all=True,
                stream_callback=callback,
                override_params=dynamic_params,
            ),
            self._loop,
        )

        try:
            return self._current_future.result()
        except asyncio.CancelledError:
            LOGGER.info(f"Session {self.session_id} task cancelled by user.")
            raise
        finally:
            self._current_future = None

    def interrupt_task(self):
        self.touch()
        if self._current_future and not self._current_future.done():
            self._current_future.cancel()
            LOGGER.info(f"Interrupt signal sent to session {self.session_id}")
            return True
        return False


class SessionManager:
    def __init__(self, timeout_seconds: int | None = 1800):
        self._sessions: Dict[str, DemoSession] = {}
        self._lock = threading.Lock()

        self.timeout = (
            timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
        )
        self._cleaner_thread = None
        if self.timeout:
            self._cleaner_thread = threading.Thread(
                target=self._cleanup_loop, daemon=True
            )
            self._cleaner_thread.start()

    def _cleanup_loop(self):
        while self.timeout:
            time.sleep(60)
            try:
                self._check_timeouts()
            except Exception as e:
                LOGGER.error(f"Error in session cleanup loop: {e}")

    def _check_timeouts(self):
        if self.timeout is None:
            return
        now = time.time()
        ids_to_remove = []

        with self._lock:
            for sid, session in self._sessions.items():
                if now - session.last_accessed > self.timeout:
                    ids_to_remove.append(sid)

        if ids_to_remove:
            LOGGER.info(f"Cleanup: Found {len(ids_to_remove)} expired sessions.")
            for sid in ids_to_remove:
                LOGGER.info(f"Auto-closing expired session: {sid}")
                self.remove(sid)

    def get_or_create(self, session_id: str) -> DemoSession:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = DemoSession(session_id)
            else:
                self._sessions[session_id].touch()
            return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[DemoSession]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.touch()
            return session

    def remove(self, session_id: str):
        session = None
        with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                del self._sessions[session_id]

        if session:
            session.stop()


SESSION_MANAGER = SessionManager(timeout_seconds=SESSION_TIMEOUT_SECONDS)

# ===== Background Chat Task Manager =====
# Manager for background-running chat tasks


@dataclass
class BackgroundChatTask:
    """Background chat task data structure."""

    task_id: str
    pipeline_name: str
    question: str
    session_id: str
    status: str  # 'running', 'completed', 'failed'
    created_at: float
    user_id: str = ""  # User ID for isolating tasks from different users
    completed_at: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None
    sources: Optional[List[Dict]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "pipeline_name": self.pipeline_name,
            "question": (
                self.question[:50] + "..." if len(self.question) > 50 else self.question
            ),
            "full_question": self.question,
            "status": self.status,
            "created_at": self.created_at,
            "user_id": self.user_id,
            "completed_at": self.completed_at,
            "result": self.result,
            "result_preview": (
                (self.result[:200] + "...")
                if self.result and len(self.result) > 200
                else self.result
            ),
            "error": self.error,
            "sources": self.sources,
        }


class BackgroundTaskManager:
    """Background task manager."""

    def __init__(self, max_tasks: int = 50) -> None:
        """Initialize background task manager.

        Args:
            max_tasks: Maximum number of tasks to keep
        """
        self._tasks: Dict[str, BackgroundChatTask] = {}
        self._lock = threading.Lock()
        self._max_tasks = max_tasks

    def create_task(
        self, pipeline_name: str, question: str, session_id: str, user_id: str = ""
    ) -> str:
        """Create a new background task.

        Args:
            pipeline_name: Name of the pipeline
            question: User question
            session_id: Session identifier
            user_id: User identifier

        Returns:
            Task ID
        """
        task_id = f"bg_{int(time.time()*1000)}_{str(uuid.uuid4())[:8]}"

        with self._lock:
            # Clean up old tasks to keep list under maximum size
            if len(self._tasks) >= self._max_tasks:
                self._cleanup_old_tasks()

            task = BackgroundChatTask(
                task_id=task_id,
                pipeline_name=pipeline_name,
                question=question,
                session_id=session_id,
                status="running",
                created_at=time.time(),
                user_id=user_id,
            )
            self._tasks[task_id] = task

        return task_id

    def update_task(
        self,
        task_id: str,
        status: Optional[str] = None,
        result: Optional[str] = None,
        error: Optional[str] = None,
        sources: Optional[List[Dict]] = None,
    ) -> None:
        """Update task status.

        Args:
            task_id: Task identifier
            status: New status
            result: Task result
            error: Error message
            sources: Source documents
        """
        with self._lock:
            if task_id not in self._tasks:
                return

            task = self._tasks[task_id]
            if status:
                task.status = status
            if result is not None:
                task.result = result
            if error is not None:
                task.error = error
            if sources is not None:
                task.sources = sources
            if status in ("completed", "failed"):
                task.completed_at = time.time()

    def get_task(self, task_id: str, user_id: str = "") -> Optional[Dict]:
        """Get task information (only returns tasks belonging to the user).

        Args:
            task_id: Task identifier
            user_id: User identifier

        Returns:
            Task dictionary or None
        """
        with self._lock:
            task = self._tasks.get(task_id)
            # Must provide user_id and match to return task
            if task and user_id and task.user_id == user_id:
                return task.to_dict()
            return None

    def list_tasks(self, limit: int = 20, user_id: str = "") -> List[Dict]:
        """List user's tasks (sorted by time, descending).

        Args:
            limit: Maximum number of tasks to return
            user_id: User identifier

        Returns:
            List of task dictionaries
        """
        with self._lock:
            # Debug log: print all task user_ids
            all_user_ids = [(t.task_id, t.user_id) for t in self._tasks.values()]
            LOGGER.debug(
                f"All tasks user_ids: {all_user_ids}, filtering for user_id: '{user_id}'"
            )

            # Filter tasks belonging to this user
            # (If no user_id provided, return no tasks for security)
            if not user_id:
                LOGGER.warning("No user_id provided, returning empty list for security")
                return []

            user_tasks = [t for t in self._tasks.values() if t.user_id == user_id]
            tasks = sorted(user_tasks, key=lambda t: t.created_at, reverse=True)[:limit]
            return [t.to_dict() for t in tasks]

    def delete_task(self, task_id: str, user_id: str = "") -> bool:
        """Delete a task (only allows deletion of tasks belonging to the user).

        Args:
            task_id: Task identifier
            user_id: User identifier

        Returns:
            True if deleted, False otherwise
        """
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                # Must provide user_id and match to delete
                if user_id and task.user_id == user_id:
                    del self._tasks[task_id]
                    return True
            return False

    def clear_completed(self, user_id: str = "") -> int:
        """Clear user's completed tasks.

        Args:
            user_id: User identifier

        Returns:
            Number of tasks cleared
        """
        with self._lock:
            # Must provide user_id to clear
            if not user_id:
                return 0

            to_delete = [
                tid
                for tid, task in self._tasks.items()
                if task.status in ("completed", "failed") and task.user_id == user_id
            ]
            for tid in to_delete:
                del self._tasks[tid]
            return len(to_delete)

    def _cleanup_old_tasks(self) -> None:
        """Clean up oldest completed tasks."""
        completed = [
            (tid, task)
            for tid, task in self._tasks.items()
            if task.status in ("completed", "failed")
        ]
        completed.sort(key=lambda x: x[1].created_at)

        # Delete half of completed tasks
        for tid, _ in completed[: len(completed) // 2]:
            del self._tasks[tid]


BACKGROUND_TASK_MANAGER = BackgroundTaskManager()

# API Logic (Wrappers)


def start_demo_session(name: str, session_id: str):
    session = SESSION_MANAGER.get_or_create(session_id)
    try:
        session.start(name)
        return {"status": "started", "session_id": session_id}
    except Exception as e:
        SESSION_MANAGER.remove(session_id)
        raise e


def stop_demo_session(session_id: str):
    SESSION_MANAGER.remove(session_id)
    return {"status": "stopped", "session_id": session_id}


# Chat Logic (Streaming + Context)


def _prepare_chat_context(name: str, question: str):
    # This prepares the temporary JSONL file for Benchmark/Retriever
    param_path = _resolve_parameter_path(name, for_write=False)
    original_text = param_path.read_text(encoding="utf-8")
    params = load_parameters(name)

    dataset_path = None
    if "benchmark" in params and "benchmark" in params["benchmark"]:
        d_dir = CHAT_DATASET_DIR
        d_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = d_dir / f"{name}_{int(time.time()*1000)}.jsonl"
        record = {"id": 0, "question": question, "golden_answers": [], "meta_data": {}}
        dataset_path.write_text(
            json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        params["benchmark"]["benchmark"]["path"] = _as_project_relative(dataset_path)
        save_parameters(name, params)

    return param_path, original_text, dataset_path


def chat_demo_stream(
    name: str, question: str, session_id: str, dynamic_params: Dict[str, Any] = None
):
    """Generator function for SSE streaming"""
    session = SESSION_MANAGER.get(session_id)
    if not session:
        yield f"data: {json.dumps({'type': 'error', 'message': 'Session expired'})}\n\n"
        return

    # 1. Prepare (Write JSONL)
    param_path, original_text, dataset_path = _prepare_chat_context(name, question)
    before_memory = {
        str(path) for path in sorted(OUTPUT_DIR.glob(f"memory_*_{name}_*.json"))
    }

    # 2. Queue for async -> sync bridge
    token_queue = queue.Queue()

    async def token_callback(event_data: Any) -> None:
        """Callback for streaming events."""
        if isinstance(event_data, dict):
            # Structured events from client.py (step_start, sources, step_end, token dict)
            token_queue.put(event_data)
        else:
            # Compatible with old logic: if pure string, treat as regular token
            token_queue.put({"type": "token", "content": str(event_data)})

    # 3. Background Task
    def run_bg():
        try:
            # Execute
            res = session.run_chat(token_callback, dynamic_params)

            # Cleanup & Resolve
            param_path.write_text(original_text, encoding="utf-8")

            # Resolve Answer
            final_ans = _extract_result(res)
            mem_ans, mem_file = _find_memory_answer(name, before_memory)

            answer = final_ans or mem_ans or "No answer generated"

            # Update conversation history (first turn)
            session.add_to_history("user", question)
            session.add_to_history("assistant", answer)
            session.mark_first_turn_done()

            # Initialize multi-turn conversation client (for subsequent turns)
            try:
                session.init_multiturn_client()
            except Exception as e:
                LOGGER.warning(f"Failed to init multiturn client: {e}")

            final_data = {
                "status": "succeeded",
                "answer": answer,
                "dataset_path": (
                    _as_project_relative(dataset_path) if dataset_path else None
                ),
                "memory_path": _as_project_relative(mem_file) if mem_file else None,
                "is_first_turn": True,
            }
            token_queue.put({"type": "final", "data": final_data})

        except asyncio.CancelledError:
            # [New] Catch CancelledError, treat as user-initiated stop
            # Don't send error message, or send a specific info
            # Actually since frontend has already Aborted, whatever we send here
            # won't be received by frontend, mainly for clean backend logs
            LOGGER.info("Chat background task cancelled.")
            token_queue.put(None)  # End queue
            return
        except Exception as e:
            LOGGER.error(f"Chat error: {e}")
            try:
                LOGGER.error(f"Faulty item: {item}")
            except Exception:
                pass
            token_queue.put({"type": "error", "message": str(e)})
        finally:
            token_queue.put(None)  # Sentinel

    threading.Thread(target=run_bg, daemon=True).start()

    # 4. Yield Stream
    while True:
        item = token_queue.get()
        if item is None:
            break
        try:
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        except TypeError as e:
            LOGGER.error(f"JSON serialization failed for item type {type(item)}: {e}")
            import traceback

            LOGGER.error(traceback.format_exc())
            # Try to identify the bad field
            if isinstance(item, dict):
                for k, v in item.items():
                    try:
                        json.dumps(v, ensure_ascii=False)
                    except TypeError:
                        LOGGER.error(
                            f"Field '{k}' is not serializable: type={type(v)}, value={v}"
                        )

            yield f"data: {json.dumps({'type': 'error', 'message': 'Internal Serialization Error'})}\n\n"


def chat_multiturn_stream(
    session_id: str,
    question: str,
    dynamic_params: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
):
    """Multi-turn conversation streaming - directly uses LocalGenerationService for generation.

    Does not go through full pipeline, only pure multi-turn conversation.

    Args:
        session_id: Session identifier
        question: Current user question
        dynamic_params: Dynamic parameters
        conversation_history: Frontend-provided conversation history (preferred)

    Yields:
        Server-sent events stream
    """
    session = SESSION_MANAGER.get(session_id)
    if not session:
        yield f"data: {json.dumps({'type': 'error', 'message': 'Session expired'})}\n\n"
        return

    # Use frontend-provided conversation history (frontend is "single source of truth")
    # If frontend didn't provide, use backend-maintained history (compatible with old logic)
    if conversation_history is not None:
        history = list(conversation_history)  # Copy to avoid modifying original list
    else:
        history = session.get_conversation_history()

    # Add current question
    history.append({"role": "user", "content": question})

    token_queue = queue.Queue()

    async def token_callback(event_data):
        if isinstance(event_data, dict):
            token_queue.put(event_data)
        else:
            token_queue.put({"type": "token", "content": str(event_data)})

    def run_bg() -> None:
        """Background task for multi-turn chat generation."""
        try:
            # Get generation configuration
            # First try to get from current pipeline parameters
            pipeline_name = session._pipeline_name
            if pipeline_name:
                try:
                    params = load_parameters(pipeline_name)
                    gen_params = params.get("generation", {})
                except Exception:
                    gen_params = {}
            else:
                gen_params = {}

            # If dynamic_params has generation config, override
            if dynamic_params and "generation" in dynamic_params:
                gen_params.update(dynamic_params["generation"])

            # Get system_prompt
            system_prompt = gen_params.get("system_prompt", "")

            # Import LocalGenerationService
            import sys
            import os

            sys.path.append(os.getcwd())
            try:
                from servers.generation.src.local_generation import (
                    LocalGenerationService,
                )
            except ImportError:
                token_queue.put(
                    {"type": "error", "message": "LocalGenerationService not available"}
                )
                token_queue.put(None)
                return

            # Create service instance
            try:
                service = LocalGenerationService(
                    backend_configs=gen_params.get("backend_configs", {}),
                    sampling_params=gen_params.get("sampling_params", {}),
                    extra_params=gen_params.get("extra_params", {}),
                    backend="openai",
                )
            except Exception as e:
                token_queue.put(
                    {
                        "type": "error",
                        "message": f"Failed to init generation service: {e}",
                    }
                )
                token_queue.put(None)
                return

            # Send step_start event
            token_queue.put(
                {"type": "step_start", "name": "multiturn_generate", "depth": 0}
            )

            # Execute streaming generation
            full_content = ""

            async def generate():
                nonlocal full_content
                async for token in service.multiturn_generate_stream(
                    messages=history, system_prompt=system_prompt
                ):
                    full_content += token
                    token_queue.put(
                        {
                            "type": "token",
                            "content": token,
                            "step": "generation.multiturn_generate",
                            "is_final": True,
                        }
                    )

            # Run in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(generate())
            finally:
                loop.close()

            # Send step_end event
            token_queue.put(
                {
                    "type": "step_end",
                    "name": "multiturn_generate",
                    "output": (
                        f"Generated:\n{full_content[:500]}..."
                        if len(full_content) > 500
                        else f"Generated:\n{full_content}"
                    ),
                }
            )

            # Update conversation history
            session.add_to_history("user", question)
            session.add_to_history("assistant", full_content)

            # Send final result
            final_data = {
                "status": "succeeded",
                "answer": full_content,
                "is_multiturn": True,
            }
            token_queue.put({"type": "final", "data": final_data})

        except asyncio.CancelledError:
            LOGGER.info("Multiturn chat cancelled.")
            token_queue.put(None)
            return
        except Exception as e:
            LOGGER.error(f"Multiturn chat error: {e}")
            import traceback

            traceback.print_exc()
            token_queue.put({"type": "error", "message": str(e)})
        finally:
            token_queue.put(None)

    threading.Thread(target=run_bg, daemon=True).start()

    # Output stream
    while True:
        item = token_queue.get()
        if item is None:
            break
        try:
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        except TypeError as e:
            LOGGER.error(f"JSON serialization failed: {e}")
            yield (
                f"data: {json.dumps({'type': 'error', 'message': 'Internal Serialization Error'})}\n\n"
            )


# Helpers (Config & Files)


def _extract_result(res: Any) -> Optional[str]:
    """Extract answer from pipeline result.

    Args:
        res: Pipeline execution result

    Returns:
        Extracted answer string or None
    """
    if not res:
        return None

    # 1. If res is wrapped dictionary returned by execute_pipeline (return_all=True)
    if isinstance(res, dict) and "final_result" in res:
        target = res["final_result"]
    else:
        target = res

    # [Safe Guard] Handle Pydantic objects (RootModel, etc.)
    try:
        if hasattr(target, "model_dump"):
            target = target.model_dump()
        elif hasattr(target, "dict"):
            target = target.dict()
    except Exception:
        pass

    # 2. Try to parse target data
    try:
        # Case A: target is JSON string
        if isinstance(target, str):
            try:
                parsed = json.loads(target)
                if isinstance(parsed, dict) and "ans_ls" in parsed:
                    val = parsed["ans_ls"][0]
                    return str(val) if not isinstance(val, str) else val
            except Exception:
                return target

        # Case B: target is already Dict
        if isinstance(target, dict):
            if "ans_ls" in target:
                val = target["ans_ls"][0]
                return str(val) if not isinstance(val, str) else val
            # Handle root (Pydantic RootModel dump result may be in root field)
            if "root" in target and isinstance(target["root"], str):
                return target["root"]

        # Case C: Still not found, try to find from content structure (fallback)
        if hasattr(target, "content") and isinstance(target.content, list):
            text = target.content[0].text
            try:
                parsed = json.loads(text)
                val = parsed.get("ans_ls", [""])[0]
                return str(val) if not isinstance(val, str) else val
            except Exception:
                return text

    except Exception as e:
        LOGGER.warning(f"Failed to extract result: {e}")

    # Final fallback: convert to string if not None
    return str(target) if target is not None else None


def _find_memory_answer(name: str, before: set[str]):
    files = sorted(OUTPUT_DIR.glob(f"memory_*_{name}_*.json"))
    candidates = [p for p in files if str(p) not in before]
    if not candidates:
        return None, None
    target = max(candidates, key=lambda p: p.stat().st_mtime)

    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        if data:
            last_mem = data[-1].get("memory", {})
            # Simple heuristic
            for k in ["ans_ls", "pred_ls", "result", "answer"]:
                if k in last_mem:
                    val = last_mem[k]
                    return (
                        str(val[0]) if isinstance(val, list) and val else str(val)
                    ), target
    except Exception:
        pass
    return None, target


def _as_project_relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def pipeline_path(name: str) -> Path:
    return PIPELINES_DIR / f"{name.replace('..','_')}.yaml"


def _find_pipeline_file(name: str) -> Path | None:
    p = pipeline_path(name)
    if p.exists():
        return p
    l = LEGACY_PIPELINES_DIR / f"{name}.yaml"
    return l if l.exists() else None


def _parameter_candidates(config_file: Path) -> List[Path]:
    base = config_file.stem
    return [
        config_file.parent / "parameter" / f"{base}_parameter.yaml",
        PIPELINES_DIR / "parameter" / f"{base}_parameter.yaml",
    ]


def _resolve_parameter_path(name: str, *, for_write: bool = False) -> Path:
    cfg = _find_pipeline_file(name)
    if not cfg:
        raise PipelineManagerError("Pipeline not found")
    cands = _parameter_candidates(cfg)
    if not for_write:
        for c in cands:
            if c.exists():
                return c
    for c in cands:
        if c.parent.exists() or for_write:
            return c
    return cands[0]


# CRUD Operations


def list_servers() -> Dict:
    res = {}
    if SERVERS_DIR.exists():
        for d in SERVERS_DIR.iterdir():
            if d.is_dir():
                res[d.name] = _ensure_server_yaml(d)
    return res


def list_server_tools() -> List[ServerTool]:
    res = []
    for srv, cfg in list_servers().items():
        for t, m in (cfg.get("tools") or {}).items():
            res.append(
                ServerTool(srv, t, "tool", m.get("input", {}), m.get("output", []))
            )
        for t, m in (cfg.get("prompts") or {}).items():
            res.append(
                ServerTool(srv, t, "prompt", m.get("input", {}), m.get("output", []))
            )
    return res


# Hidden list: these pipelines are for underlying KB processing, not shown in UI
HIDDEN_PIPELINES = {
    "build_text_corpus",
    "corpus_chunk",
    "milvus_index",
    "multiturn_chat",
}


def list_pipelines() -> List[Dict[str, Any]]:
    res = []
    for d in [PIPELINES_DIR, LEGACY_PIPELINES_DIR]:
        if d.exists():
            for f in d.glob("*.yaml"):
                # Skip pipelines in hidden list
                if f.stem in HIDDEN_PIPELINES:
                    continue

                if not any(r["name"] == f.stem for r in res):

                    param_path = d / "parameter" / f"{f.stem}_parameter.yaml"
                    if not param_path.exists():
                        param_path = f.parent / "parameter" / f"{f.stem}_parameter.yaml"

                    is_ready = param_path.exists()

                    res.append(
                        {
                            "name": f.stem,
                            "config": yaml.safe_load(f.read_text(encoding="utf-8"))
                            or {},
                            "is_ready": is_ready,
                        }
                    )
    return res


def load_pipeline(name: str) -> Dict:
    p = _find_pipeline_file(name)
    if not p:
        raise PipelineManagerError(f"Pipeline {name} not found")
    yaml_text = p.read_text(encoding="utf-8")
    data = yaml.safe_load(yaml_text) or {}

    # Also return original YAML text for frontend consistency
    if isinstance(data, dict):
        data["_raw_yaml"] = yaml_text
    else:
        data = {"pipeline": data, "_raw_yaml": yaml_text}

    return data


def parse_pipeline_yaml_content(yaml_content: str) -> Dict:
    """Parse arbitrary YAML text and return safe Python object.

    Args:
        yaml_content: YAML content string

    Returns:
        Parsed dictionary

    Raises:
        PipelineManagerError: If YAML content is empty or invalid
    """
    if yaml_content is None:
        raise PipelineManagerError("YAML content is empty")

    try:
        parsed = yaml.safe_load(yaml_content)
    except yaml.YAMLError as exc:
        raise PipelineManagerError(f"Invalid YAML syntax: {exc}") from exc

    return parsed or {}


def save_pipeline(payload: Dict) -> Dict:
    name = payload.get("name")
    if not name:
        raise PipelineManagerError("Name required")

    steps = payload.get("pipeline", [])
    servers = set()

    def _scan(s):
        if isinstance(s, str) and "." in s:
            servers.add(s.split(".")[0])
        elif isinstance(s, list):
            for item in s:
                _scan(item)
        elif isinstance(s, dict):
            for v in s.values():
                _scan(v)

    _scan(steps)

    yaml_data = {
        "servers": {s: f"servers/{s}" for s in sorted(servers)},
        "pipeline": steps,
    }

    with pipeline_path(name).open("w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False, allow_unicode=True)

    return {"name": name, **yaml_data}


def save_pipeline_yaml(name: str, yaml_content: str) -> Dict:
    """Save YAML text content directly to file.

    Args:
        name: Pipeline name
        yaml_content: YAML content string

    Returns:
        Dictionary with save status

    Raises:
        PipelineManagerError: If name is missing or YAML syntax is invalid
    """
    if not name:
        raise PipelineManagerError("Name required")

    # Validate YAML syntax
    try:
        parsed = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        raise PipelineManagerError(f"Invalid YAML syntax: {e}")

    # Write file
    p = pipeline_path(name)
    p.write_text(yaml_content, encoding="utf-8")

    return {"name": name, "status": "saved"}


def delete_pipeline(name: str):
    p = _find_pipeline_file(name)
    if p:
        p.unlink()


def rename_pipeline(old_name: str, new_name: str) -> Dict:
    """Rename a pipeline file.

    Args:
        old_name: Current pipeline name
        new_name: New pipeline name

    Returns:
        Dictionary with rename status

    Raises:
        PipelineManagerError: If pipeline not found or new name already exists
    """
    if not old_name or not new_name:
        raise PipelineManagerError("Both old_name and new_name are required")

    # Check if old file exists
    old_path = _find_pipeline_file(old_name)
    if not old_path:
        raise PipelineManagerError(f"Pipeline '{old_name}' not found")

    # Check if new name already exists
    new_path = pipeline_path(new_name)
    if new_path.exists():
        raise PipelineManagerError(f"Pipeline '{new_name}' already exists")

    # Execute rename
    old_path.rename(new_path)

    # Also rename corresponding parameter file (if exists)
    old_param_path = _resolve_parameter_path(old_name, for_write=False)
    if old_param_path.exists():
        new_param_path = _resolve_parameter_path(new_name, for_write=True)
        new_param_path.parent.mkdir(parents=True, exist_ok=True)
        old_param_path.rename(new_param_path)

    return {"status": "renamed", "old_name": old_name, "new_name": new_name}


def load_parameters(name: str) -> Dict:
    p = _resolve_parameter_path(name, for_write=False)
    if not p.exists():
        raise PipelineManagerError("Parameters not found. Build first.")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def save_parameters(name: str, params: Dict):
    p = _resolve_parameter_path(name, for_write=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(params, f, sort_keys=False, allow_unicode=True)


def build(name: str) -> Dict:
    funcs = _ensure_client_funcs()
    p = _find_pipeline_file(name)
    if not p:
        raise PipelineManagerError(f"Not found: {name}")

    try:
        # Build is fast, run synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(funcs["build"](str(p)))
        loop.close()
    except Exception as e:
        raise PipelineManagerError(f"Build failed: {e}")
    return {"status": "ok"}


# Internal Utils (Stubs & Parsing)
@dataclass
class ServerTool:
    server: str
    tool: str
    kind: str
    input_spec: Dict
    output_spec: List

    @property
    def identifier(self):
        return f"{self.server}.{self.tool}"


def _flatten_param_keys(data: Any, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else key
            keys.add(path)
            keys |= _flatten_param_keys(value, path)
    return keys


def _generate_server_stub(
    server_dir: Path, module_path: Path, parameter_path: Path
) -> Dict[str, Any]:
    """Generate server stub configuration from source code using static analysis.

    Strategy:
    1. Scan entire file, record all function names and parameters (definitions).
    2. Scan entire file, record all tool registration behaviors (registrations).
    3. Finally match uniformly, ignoring code definition order.

    Args:
        server_dir: Server directory path
        module_path: Path to server module file
        parameter_path: Path to parameter YAML file

    Returns:
        Dictionary with tools and prompts configuration
    """
    try:
        source = module_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}

    tree = ast.parse(source, filename=str(module_path))

    # === Data containers ===
    # Store all function definitions: { "func_name": ["arg1", "arg2"] }
    definitions: Dict[str, List[str]] = {}
    # Store all registration intents: [ {"name": "func_name", "output": "...", "type": "tool"} ]
    registrations: List[Dict[str, Any]] = []

    # === Helper functions ===
    def parse_output_spec(keywords: List[ast.keyword]) -> List[str]:
        """Parse output specification from decorator keywords."""
        for kw in keywords:
            if kw.arg == "output" and isinstance(kw.value, ast.Constant):
                val = kw.value.value
                if not val:
                    return []
                out_part = val.split("->", 1)[-1]
                return [
                    x.strip()
                    for x in out_part.split(",")
                    if x.strip().lower() != "none"
                ]
        return []

    # === AST visitor ===
    class Collector(ast.NodeVisitor):
        """AST visitor to collect function definitions and registrations."""

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._handle_func(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._handle_func(node)

        def _handle_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            # 1. [Collect definitions] Record function name and parameters (ignore self/cls)
            args = [a.arg for a in node.args.args if a.arg not in ("self", "cls")]
            definitions[node.name] = args

            # 2. [Collect decorators] @app.tool
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                    if dec.func.attr in {"tool", "prompt"}:
                        registrations.append(
                            {
                                "name": node.name,
                                "output": parse_output_spec(dec.keywords),
                                "type": dec.func.attr,  # 'tool' or 'prompt'
                            }
                        )

            # Continue traversing function body
            # (Generally functions aren't defined inside functions, but for safety)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            # 3. [Collect explicit calls] mcp.tool(self.func)
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "tool",
                "prompt",
            }:
                tool_name = None
                # Parse argument: mcp.tool(self.my_func) -> my_func
                if node.args:
                    first = node.args[0]
                    if isinstance(first, ast.Attribute):
                        tool_name = first.attr
                    elif isinstance(first, ast.Name):
                        tool_name = first.id

                if tool_name:
                    registrations.append(
                        {
                            "name": tool_name,
                            "output": parse_output_spec(node.keywords),
                            "type": node.func.attr,
                        }
                    )

            self.generic_visit(node)

    # Execute scan
    Collector().visit(tree)

    # === Match and build ===

    # Prepare parameter file keys for $ matching
    param_keys: set[str] = set()
    if parameter_path.exists():
        try:
            param_data = (
                yaml.safe_load(parameter_path.read_text(encoding="utf-8")) or {}
            )
            param_keys = {k.split(".")[0] for k in _flatten_param_keys(param_data)}
        except Exception:
            pass

    tools_map = {}
    prompts_map = {}

    for reg in registrations:
        name = reg["name"]
        kind = reg["type"]
        outputs = reg["output"]

        # Find parameters from definition pool
        # Note: Assumes no duplicate function names in file (or last definition overrides)
        # In MCP Server files, tool function names are usually unique
        func_args = definitions.get(name, [])

        # Build input mapping
        input_mapping = {}
        for arg in func_args:
            if arg in param_keys:
                input_mapping[arg] = f"${arg}"
            else:
                input_mapping[arg] = arg

        entry = {"input": input_mapping}
        if outputs:
            entry["output"] = outputs

        if kind == "prompt":
            prompts_map[name] = entry
        else:
            tools_map[name] = entry

    return {
        "path": str(module_path),
        "parameter": str(parameter_path),
        "tools": tools_map,
        "prompts": prompts_map,
    }


def _ensure_server_yaml(server_dir: Path) -> Dict[str, Any]:
    """Ensure server.yaml exists, generate stub from source if needed.

    Args:
        server_dir: Server directory path

    Returns:
        Server configuration dictionary
    """
    server_name = server_dir.name
    yaml_path = server_dir / "server.yaml"
    parameter_path = server_dir / "parameter.yaml"

    # 1. Prefer reading existing server.yaml (if built)
    if yaml_path.exists():
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            # Return if file format is correct, even if content is empty
            if isinstance(data, dict):
                return data
        except Exception as e:
            LOGGER.warning(
                f"Failed to load existing server.yaml for {server_name}: {e}"
            )

    # 2. If no yaml, try parsing from source code (Stub)
    # Strategy A: Standard path src/{server_name}.py
    module_path = server_dir / "src" / f"{server_name}.py"

    # Strategy B: If standard path doesn't exist, scan any .py files in src
    if not module_path.exists():
        src_dir = server_dir / "src"
        if src_dir.exists():
            # Exclude __init__.py, find other .py files
            candidates = [p for p in src_dir.glob("*.py") if p.name != "__init__.py"]
            if candidates:
                # Prefer files containing 'server' or 'app', otherwise take first
                module_path = next(
                    (p for p in candidates if "server" in p.name or "app" in p.name),
                    candidates[0],
                )
                LOGGER.info(
                    f"Inferred entry point for {server_name}: {module_path.name}"
                )

    # 3. Execute parsing
    if module_path.exists():
        try:
            # Dynamically generate configuration so frontend can see tool list
            stub = _generate_server_stub(server_dir, module_path, parameter_path)
            if not stub.get("tools") and not stub.get("prompts"):
                LOGGER.warning(
                    f"Server {server_name} parsed but no tools/prompts found "
                    f"in {module_path.name}"
                )
            return stub
        except Exception as e:
            LOGGER.warning(f"Failed to generate stub for {server_name}: {e}")

    LOGGER.warning(f"Skipping {server_name}: No valid entry point or config found.")
    return {}


# Stub dependencies for imports
def _ensure_stub_module(name):
    if name not in sys.modules:
        m = ModuleType(name)
        sys.modules[name] = m


def interrupt_chat(session_id: str):
    session = SESSION_MANAGER.get(session_id)
    if session:
        success = session.interrupt_task()
        return {"status": "interrupted", "active_task_cancelled": success}
    return {"status": "session_not_found"}


# ===== Background Chat Execution =====

# Session manager specifically for background tasks (independent from foreground)
BACKGROUND_SESSION_MANAGER = SessionManager(
    timeout_seconds=BACKGROUND_SESSION_TIMEOUT_SECONDS
)  # Background task sessions keep alive longer


def run_background_chat(
    name: str,
    question: str,
    session_id: str,
    dynamic_params: Optional[Dict[str, Any]] = None,
    user_id: str = "",
) -> str:
    """Start background chat task.

    Uses independent Session, unaffected by foreground operations.
    Returns task ID for subsequent status queries.

    Args:
        name: Pipeline name
        question: User question
        session_id: Session identifier
        dynamic_params: Optional dynamic parameters
        user_id: User identifier

    Returns:
        Task ID
    """
    # Create independent session ID for background task
    bg_session_id = f"bg_{name}_{int(time.time()*1000)}_{str(uuid.uuid4())[:8]}"
    task_id = BACKGROUND_TASK_MANAGER.create_task(
        name, question, bg_session_id, user_id=user_id
    )

    def _execute_background() -> None:
        """Execute background chat task."""
        bg_session = None
        try:
            # Create independent background Session
            bg_session = BACKGROUND_SESSION_MANAGER.get_or_create(bg_session_id)
            bg_session.start(name)

            LOGGER.info(
                f"Background task {task_id} started with independent session {bg_session_id}"
            )

        except Exception as e:
            LOGGER.error(f"Failed to create background session: {e}")
            BACKGROUND_TASK_MANAGER.update_task(
                task_id, status="failed", error=f"Failed to initialize: {str(e)}"
            )
            # Clean up failed session
            if bg_session_id:
                BACKGROUND_SESSION_MANAGER.remove(bg_session_id)
            return

        # Prepare context
        try:
            param_path, original_text, dataset_path = _prepare_chat_context(
                name, question
            )
        except Exception as e:
            BACKGROUND_TASK_MANAGER.update_task(
                task_id, status="failed", error=f"Context preparation failed: {str(e)}"
            )
            BACKGROUND_SESSION_MANAGER.remove(bg_session_id)
            return

        before_memory = {
            str(path) for path in sorted(OUTPUT_DIR.glob(f"memory_*_{name}_*.json"))
        }

        # Collect sources
        collected_sources = []

        async def token_callback(event_data: Any) -> None:
            """Callback for collecting sources from events."""
            if isinstance(event_data, dict):
                if event_data.get("type") == "sources":
                    docs = event_data.get("data", [])
                    collected_sources.extend(docs)

        try:
            # Execute pipeline
            res = bg_session.run_chat(token_callback, dynamic_params)

            # Restore original parameter file
            param_path.write_text(original_text, encoding="utf-8")

            # Extract result
            final_ans = _extract_result(res)
            mem_ans, mem_file = _find_memory_answer(name, before_memory)

            answer = final_ans or mem_ans or "No answer generated"

            # Update task to completed status
            BACKGROUND_TASK_MANAGER.update_task(
                task_id, status="completed", result=answer, sources=collected_sources
            )

            LOGGER.info(f"Background task {task_id} completed successfully")

        except asyncio.CancelledError:
            BACKGROUND_TASK_MANAGER.update_task(
                task_id, status="failed", error="Task was cancelled"
            )
            LOGGER.info(f"Background task {task_id} cancelled")

        except Exception as e:
            LOGGER.error(f"Background task {task_id} failed: {e}")
            BACKGROUND_TASK_MANAGER.update_task(task_id, status="failed", error=str(e))
        finally:
            # Clean up background Session after task completion
            BACKGROUND_SESSION_MANAGER.remove(bg_session_id)
            LOGGER.info(f"Background session {bg_session_id} cleaned up")

    # Execute in background thread
    threading.Thread(target=_execute_background, daemon=True).start()

    return task_id


def get_background_task(task_id: str, user_id: str = "") -> Optional[Dict]:
    """Get background task status.

    Args:
        task_id: Task identifier
        user_id: User identifier

    Returns:
        Task dictionary or None
    """
    return BACKGROUND_TASK_MANAGER.get_task(task_id, user_id)


def list_background_tasks(limit: int = 20, user_id: str = "") -> List[Dict]:
    """List user's background tasks.

    Args:
        limit: Maximum number of tasks to return
        user_id: User identifier

    Returns:
        List of task dictionaries
    """
    return BACKGROUND_TASK_MANAGER.list_tasks(limit, user_id)


def delete_background_task(task_id: str, user_id: str = "") -> bool:
    """Delete a background task.

    Args:
        task_id: Task identifier
        user_id: User identifier

    Returns:
        True if deleted, False otherwise
    """
    return BACKGROUND_TASK_MANAGER.delete_task(task_id, user_id)


def clear_completed_background_tasks(user_id: str = "") -> int:
    """Clear user's completed background tasks.

    Args:
        user_id: User identifier

    Returns:
        Number of tasks cleared
    """
    return BACKGROUND_TASK_MANAGER.clear_completed(user_id)


# Knowledge Base Management
def load_kb_config() -> Dict[str, Any]:
    default_config = {
        "milvus": {
            "uri": "tcp://127.0.0.1:19530",
            "token": "",
            "id_field_name": "id",
            "vector_field_name": "vector",
            "text_field_name": "contents",
            "id_max_length": 64,
            "text_max_length": 60000,
            "metric_type": "IP",
            "index_params": {"index_type": "AUTOINDEX", "metric_type": "IP"},
            "search_params": {"metric_type": "IP", "params": {}},
            "index_chunk_size": 1000,
        }
    }

    if not KB_CONFIG_PATH.exists():
        return default_config

    try:
        saved = json.loads(KB_CONFIG_PATH.read_text(encoding="utf-8"))
        if "milvus" not in saved:
            return default_config

        full_cfg = default_config["milvus"].copy()
        full_cfg.update(saved["milvus"])
        return {"milvus": full_cfg}

    except Exception:
        return default_config


def save_kb_config(config: Dict[str, str]):
    KB_CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _get_milvus_client() -> Any:
    """Get Milvus client instance.

    Returns:
        MilvusClient instance

    Raises:
        ValueError: If URI is empty
        Exception: If connection fails
    """
    cfg = load_kb_config()
    try:
        milvus_cfg = cfg.get("milvus", {})
        uri = milvus_cfg.get("uri", "")

        if not uri:
            raise ValueError("URI is empty")

        if not uri.startswith("http") and not Path(uri).is_absolute():
            pass

        return MilvusClient(uri=uri, token=milvus_cfg.get("token", ""))
    except Exception as e:
        LOGGER.error(f"Failed to connect to Milvus: {e}")
        raise e


def list_kb_files() -> Dict[str, List[Dict[str, Any]]]:
    """List knowledge base files across all categories.

    Returns:
        Dictionary with files organized by category (raw, corpus, chunks, index)
    """

    def _scan_files(d: Path, category: str) -> List[Dict[str, Any]]:
        """Scan files in a directory and return file information.

        Args:
            d: Directory path to scan
            category: Category name (raw, corpus, chunks)

        Returns:
            List of file information dictionaries
        """
        items = []
        if not d.exists():
            return items

        # Load directory-level display name mapping (for files)
        dir_display_map = _load_display_names_map(d)

        for f in sorted(d.glob("*")):
            if f.name.startswith("."):
                continue
            if f.name.startswith("_"):
                continue  # Skip metadata files (_meta.json, _display_names.json)

            is_dir = f.is_dir()

            size = 0
            file_count = 0
            if is_dir:
                # Calculate size excluding _meta.json
                for p in f.rglob("*"):
                    if p.is_file() and not p.name.startswith("_"):
                        size += p.stat().st_size
                        file_count += 1
            else:
                size = f.stat().st_size

            # [New] Read display_name
            display_name = f.name  # Default to filename
            if is_dir:
                # Folder: read from _meta.json
                meta = _read_folder_meta(f)
                if meta and meta.get("display_name"):
                    display_name = meta["display_name"]
            else:
                # File: read from directory-level mapping
                if f.name in dir_display_map:
                    display_name = dir_display_map[f.name]
                else:
                    # Default to filename without extension
                    display_name = f.stem

            item = {
                "name": f.name,
                "display_name": display_name,  # [New]
                "path": _as_project_relative(f),
                "size": size,
                "mtime": f.stat().st_mtime,
                "category": category,
                "type": "folder" if is_dir else "file",
            }

            if is_dir and file_count > 0:
                item["file_count"] = file_count

            items.append(item)
        return items

    collections = []
    db_status = "unknown"
    try:
        client = _get_milvus_client()
        names = client.list_collections()
        for name in names:
            res = client.get_collection_stats(name)
            count = res.get("row_count", 0)
            try:
                desc = client.describe_collection(name).get("description", "")
            except Exception:
                desc = ""
            collections.append(
                {
                    "name": name,
                    "display_name": _extract_display_name_from_desc(desc, name),
                    "count": count,
                    "category": "collection",
                }
            )
        client.close()
        db_status = "connected"
    except Exception as e:
        LOGGER.warning(f"Milvus connection failed: {e}")
        db_status = "error"

    return {
        "raw": _scan_files(KB_RAW_DIR, "raw"),
        "corpus": _scan_files(KB_CORPUS_DIR, "corpus"),
        "chunks": _scan_files(KB_CHUNKS_DIR, "chunks"),
        "index": collections,
        "db_status": db_status,
        "db_config": load_kb_config(),
    }


def _generate_display_name(original_names: List[str]) -> str:
    """Generate user-friendly display name.

    Args:
        original_names: List of original file names

    Returns:
        Display name string
    """
    if not original_names:
        return "Untitled"

    # Take first file's original name (without extension)
    first_name = Path(original_names[0]).stem

    if len(original_names) == 1:
        # Single file: use filename directly
        return first_name
    else:
        # Multiple files: first filename + count suffix
        return f"{first_name} (+{len(original_names) - 1} more)"


def _write_folder_meta(
    folder_path: Path, display_name: str, original_files: List[str]
) -> None:
    """Write folder metadata.

    Args:
        folder_path: Path to folder
        display_name: Display name for the folder
        original_files: List of original file names
    """
    meta_path = folder_path / "_meta.json"
    meta_data = {
        "display_name": display_name,
        "original_files": original_files,
        "created_at": datetime.now().isoformat(),
    }
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        LOGGER.warning(f"Failed to write meta.json: {e}")


def _read_folder_meta(folder_path: Path) -> Optional[Dict[str, Any]]:
    """Read folder metadata.

    Args:
        folder_path: Path to folder

    Returns:
        Metadata dictionary or None if not found
    """
    meta_path = folder_path / "_meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        LOGGER.warning(f"Failed to read meta.json: {e}")
        return None


def _load_display_names_map(directory: Path) -> Dict[str, str]:
    """Load directory-level display name mapping.

    Args:
        directory: Directory path

    Returns:
        Dictionary mapping filenames to display names
    """
    map_path = directory / "_display_names.json"
    if not map_path.exists():
        return {}
    try:
        with open(map_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        LOGGER.warning(f"Failed to load display names map: {e}")
        return {}


def _save_display_names_map(directory: Path, name_map: Dict[str, str]) -> None:
    """Save directory-level display name mapping.

    Args:
        directory: Directory path
        name_map: Dictionary mapping filenames to display names
    """
    map_path = directory / "_display_names.json"
    try:
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(name_map, f, ensure_ascii=False, indent=2)
    except Exception as e:
        LOGGER.warning(f"Failed to save display names map: {e}")


def _register_display_name(
    directory: Path, filename: str, display_name: str
) -> None:
    """Register file display name.

    Args:
        directory: Directory path
        filename: File name
        display_name: Display name to register
    """
    name_map = _load_display_names_map(directory)
    name_map[filename] = display_name
    _save_display_names_map(directory, name_map)


def upload_kb_files_batch(file_objs: List[Any]) -> Dict[str, Any]:
    """Upload multiple files to knowledge base.

    Args:
        file_objs: List of file objects to upload

    Returns:
        Dictionary with upload session information

    Raises:
        Exception: If upload fails
    """
    if not file_objs:
        return {"error": "No files provided"}

    session_id = (
        f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    )
    session_dir = KB_RAW_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    original_names = []  # [New] Record original file names
    total_size = 0

    try:
        for file_obj in file_objs:
            original_name = file_obj.filename
            original_names.append(original_name)  # [New] Save original name

            safe_name = _secure_filename_unicode(original_name)
            if not safe_name:
                safe_name = (
                    f"file_{str(uuid.uuid4())[:8]}{Path(original_name).suffix}"
                )

            save_path = session_dir / safe_name

            counter = 1
            while save_path.exists():
                stem = Path(safe_name).stem
                suffix = Path(safe_name).suffix
                save_path = session_dir / f"{stem}_{counter}{suffix}"
                counter += 1

            file_obj.save(save_path)

            saved_files.append(safe_name)
            total_size += save_path.stat().st_size

        # [New] Generate display name and write metadata
        display_name = _generate_display_name(original_names)
        _write_folder_meta(session_dir, display_name, original_names)

        LOGGER.info(
            f"Created upload session: {session_dir} with {len(saved_files)} files. "
            f"Display: {display_name}"
        )

        return {
            "name": session_id,
            "display_name": display_name,  # [New]
            "path": _as_project_relative(session_dir),
            "size": total_size,
            "type": "folder",
            "file_count": len(saved_files),
            "mtime": session_dir.stat().st_mtime,
        }

    except Exception as e:
        if session_dir.exists():
            shutil.rmtree(session_dir)
        raise e


def delete_kb_file(category: str, filename: str) -> Dict[str, str]:
    """Delete a knowledge base file or collection.

    Args:
        category: File category (raw, corpus, chunks, collection, index)
        filename: File or collection name

    Returns:
        Dictionary with deletion status

    Raises:
        Exception: If deletion fails
    """
    base_dir = None
    if category == "raw":
        base_dir = KB_RAW_DIR
    elif category == "corpus":
        base_dir = KB_CORPUS_DIR
    elif category == "chunks":
        base_dir = KB_CHUNKS_DIR
    elif category == "collection":
        return _delete_milvus_collection(filename)
    elif category == "index":
        return _delete_milvus_collection(filename)

    if not base_dir:
        raise ValueError("Invalid category")

    target_path = base_dir / filename

    if not str(target_path.resolve()).startswith(str(base_dir.resolve())):
        raise ValueError("Invalid filename path")

    if not target_path.exists():
        raise FileNotFoundError(f"File or directory not found: {filename}")

    try:
        if target_path.is_dir():
            shutil.rmtree(target_path)
            LOGGER.info(f"Deleted folder: {target_path}")
        else:
            target_path.unlink()
            LOGGER.info(f"Deleted file: {target_path}")

        return {"status": "deleted", "file": filename}

    except Exception as e:
        LOGGER.error(f"Failed to delete {filename}: {e}")
        raise e


def _delete_milvus_collection(name: str) -> Dict[str, str]:
    """Delete a Milvus collection.

    Args:
        name: Collection name

    Returns:
        Dictionary with deletion status

    Raises:
        Exception: If deletion fails
    """
    try:
        client = _get_milvus_client()
        if client.has_collection(name):
            client.drop_collection(name)
            LOGGER.info(f"Dropped collection: {name}")
        client.close()
        return {"status": "deleted", "collection": name}
    except Exception as e:
        LOGGER.error(f"Failed to drop collection {name}: {e}")
        raise e


def clear_staging_area() -> Dict[str, Any]:
    """Clear staging area: delete all files in raw, corpus, chunks directories.

    Returns:
        Dictionary with deletion counts and any errors
    """
    deleted_counts = {"raw": 0, "corpus": 0, "chunks": 0}
    errors = []

    for category, base_dir in [
        ("raw", KB_RAW_DIR),
        ("corpus", KB_CORPUS_DIR),
        ("chunks", KB_CHUNKS_DIR),
    ]:
        if not base_dir.exists():
            continue

        try:
            # Iterate through all files and folders in directory
            for item in base_dir.iterdir():
                if item.name.startswith("."):
                    continue

                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                        deleted_counts[category] += 1
                        LOGGER.info(f"Deleted folder: {item}")
                    else:
                        item.unlink()
                        deleted_counts[category] += 1
                        LOGGER.info(f"Deleted file: {item}")
                except Exception as e:
                    error_msg = f"Failed to delete {item} in {category}: {e}"
                    LOGGER.error(error_msg)
                    errors.append(error_msg)

        except Exception as e:
            error_msg = f"Error processing {category} directory: {e}"
            LOGGER.error(error_msg)
            errors.append(error_msg)

    total_deleted = sum(deleted_counts.values())
    result = {
        "status": "completed",
        "deleted_counts": deleted_counts,
        "total_deleted": total_deleted,
    }

    if errors:
        result["errors"] = errors
        result["status"] = "completed_with_errors"

    LOGGER.info(f"Staging area cleared: {total_deleted} items deleted")
    return result


def run_kb_pipeline_tool(
    pipeline_name: str,
    target_file_path: str,
    output_dir: str,
    collection_name: Optional[str] = None,
    index_mode: str = "append",  # 'append' (append), 'overwrite' (overwrite)
    chunk_params: Optional[Dict[str, Any]] = None,  # [New] Receive parameters
    embedding_params: Optional[
        Dict[str, Any]
    ] = None,  # [New] Embedding configuration (for milvus_index)
) -> Dict[str, Any]:
    """Run knowledge base pipeline tool.

    Args:
        pipeline_name: Name of the pipeline to run
        target_file_path: Path to target file or directory
        output_dir: Output directory path
        collection_name: Optional Milvus collection name
        index_mode: Index mode ("append" or "overwrite")
        chunk_params: Optional chunking parameters
        embedding_params: Optional embedding parameters

    Returns:
        Dictionary with execution result

    Raises:
        PipelineManagerError: If pipeline or parameters not found
    """

    pipeline_cfg = load_pipeline(pipeline_name)
    if not pipeline_cfg:
        raise PipelineManagerError(
            f"Pipeline '{pipeline_name}' not found. Please create it in Builder."
        )

    base_params = load_parameters(pipeline_name)
    if not base_params:
        raise PipelineManagerError(
            f"Pipeline '{pipeline_name}' parameters not found. Please Build & Save first."
        )

    target_file = Path(target_file_path)
    stem = target_file.stem
    override_params = {}

    # [New] Get source file's display_name to pass to output
    source_display_name = None
    if target_file.is_dir():
        # Source is folder (Raw stage)
        meta = _read_folder_meta(target_file)
        if meta and meta.get("display_name"):
            source_display_name = meta["display_name"]
    else:
        # Source is file (Corpus/Chunks stage), read from directory mapping
        parent_map = _load_display_names_map(target_file.parent)
        if target_file.name in parent_map:
            source_display_name = parent_map[target_file.name]
        else:
            source_display_name = target_file.stem

    if pipeline_name == "build_text_corpus":
        out_path = os.path.join(output_dir, f"{stem}.jsonl")
        override_params = {
            "corpus": {
                "parse_file_path": str(target_file),
                "text_corpus_save_path": out_path,
            }
        }
        # [New] Register output file's display_name
        if source_display_name:
            _register_display_name(
                Path(output_dir), f"{stem}.jsonl", source_display_name
            )

    elif pipeline_name == "corpus_chunk":
        out_path = os.path.join(output_dir, f"{stem}.jsonl")

        # [Modified] Base parameters
        corpus_override = {"raw_chunk_path": str(target_file), "chunk_path": out_path}

        if chunk_params:
            try:
                if "chunk_size" in chunk_params:
                    chunk_params["chunk_size"] = int(chunk_params["chunk_size"])
            except Exception:
                pass
            corpus_override.update(chunk_params)

        override_params = {"corpus": corpus_override}
        # [New] Register output file's display_name
        if source_display_name:
            _register_display_name(
                Path(output_dir), f"{stem}.jsonl", source_display_name
            )

    elif pipeline_name == "milvus_index":
        requested_name = collection_name if collection_name else stem

        # Get existing collection names and display names for deduplication
        existing_collections: set[str] = set()
        existing_display_names: dict[str, str] = {}
        display_name_lookup: dict[str, str] = {}
        client = None
        try:
            client = _get_milvus_client()
            existing_collections = set(client.list_collections())
            for _name in existing_collections:
                try:
                    desc = client.describe_collection(_name).get("description", "")
                except Exception:
                    desc = ""
                display_name = _extract_display_name_from_desc(desc, _name)
                existing_display_names[_name] = display_name
                display_name_lookup[_normalize_display_name(display_name)] = _name
        except Exception as exc:
            raise PipelineManagerError(f"Milvus connection failed: {exc}") from exc
        finally:
            try:
                if client:
                    client.close()
            except Exception:
                pass

        existing_display_name_values = set(existing_display_names.values())
        normalized_request = _normalize_display_name(requested_name)
        matched_collection = None
        if requested_name in existing_collections:
            matched_collection = requested_name
        elif normalized_request and normalized_request in display_name_lookup:
            matched_collection = display_name_lookup[normalized_request]

        if matched_collection and index_mode == "new":
            raise PipelineManagerError(
                "Collection name already exists. Choose append or overwrite."
            )

        if matched_collection and index_mode in {"append", "overwrite"}:
            safe_collection_name = matched_collection
            display_collection_name = existing_display_names.get(
                matched_collection, requested_name
            )
        else:
            # Convert to pinyin -> ASCII slug, then deduplicate
            slug_base = _transliterate_name(requested_name)
            safe_collection_name = _make_unique_name(slug_base, existing_collections)

            # Display name uses original input, add (1) increment if duplicate
            display_collection_name = _make_unique_display(
                requested_name, existing_display_name_values
            )

        is_overwrite = index_mode == "overwrite"

        full_kb_cfg = load_kb_config()
        milvus_config_dict = dict(full_kb_cfg["milvus"])
        milvus_config_dict["collection_display_name"] = display_collection_name

        # Ensure parent directory exists
        KB_INDEX_DIR.mkdir(parents=True, exist_ok=True)

        # [New] Build embedding backend configuration
        retriever_override = {
            "corpus_path": str(target_file),
            "collection_name": safe_collection_name,
            "overwrite": is_overwrite,
            "index_backend": "milvus",
            "index_backend_configs": {"milvus": milvus_config_dict},
        }

        # If user configured Embedding parameters, use OpenAI backend
        if embedding_params and embedding_params.get("api_key"):
            retriever_override["backend"] = "openai"
            retriever_override["backend_configs"] = {
                "openai": {
                    "api_key": embedding_params.get("api_key", ""),
                    "base_url": embedding_params.get(
                        "base_url", "https://api.openai.com/v1"
                    ),
                    "model_name": embedding_params.get(
                        "model_name", "text-embedding-3-small"
                    ),
                }
            }

        override_params = {"retriever": retriever_override}

    else:
        raise PipelineManagerError(f"Unsupported KB Pipeline: {pipeline_name}")

    funcs = _ensure_client_funcs()
    config_file = str(_find_pipeline_file(pipeline_name))
    param_file = str(_resolve_parameter_path(pipeline_name))

    async def _async_task():
        context = funcs["load_ctx"](config_file, param_file)

        client = funcs["create_client"](context["mcp_cfg"])

        async with client:
            return await funcs["exec_pipe"](
                client,
                context,
                is_demo=True,
                return_all=True,
                override_params=override_params,
            )

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            raw_result = loop.run_until_complete(_async_task())
        finally:
            loop.close()

        return {"status": "success", "result": _extract_result(raw_result)}

    except Exception as e:
        LOGGER.error(f"KB Task Failed: {e}")
        raise e
