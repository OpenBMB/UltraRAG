import logging
import os
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal

from rich.console import Console
from rich.logging import RichHandler

_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

_LOGGING_INITIALIZED = False
_LOGFILE_PATH: str | None = None


def redact_config(value: Any) -> Any:
    """Copy diagnostic payloads while hiding credential-bearing fields."""
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            name = str(key).lower()
            sensitive = name in {"token", "key"} or any(
                part in name
                for part in (
                    "api_key",
                    "password",
                    "secret",
                    "authorization",
                    "access_token",
                    "refresh_token",
                )
            )
            result[key] = "[REDACTED]" if sensitive and item else redact_config(item)
        return result
    if isinstance(value, (list, tuple)):
        return [redact_config(item) for item in value]
    return value


def normalize_log_level(level: str | int) -> int:
    """Convert log level string or int to logging level constant.

    Args:
        level: Log level as string or integer

    Returns:
        Logging level constant (defaults to INFO if invalid)
    """
    if isinstance(level, int):
        return level
    return _LOG_LEVELS.get(str(level).lower(), logging.INFO)


def get_logger(
    name: str,
    level: Literal["debug", "info", "warn", "error"] | str | int | None = None,
    enable_rich_tracebacks: bool = True,
    log_file: str | None = None,
) -> logging.Logger:
    """Get or create a logger with Rich formatting and file output.

    Initializes logging system on first call with both console (stderr) and
    file handlers. Calls without a level inherit the current configuration;
    an explicit level reconfigures the application-wide handlers.

    Args:
        name: Logger name (child logger of "UltraRAG" if not "UltraRAG")
        level: Application log level; omitted calls preserve the current level
        enable_rich_tracebacks: Whether to enable Rich traceback formatting
        log_file: Optional path to log file (default: timestamped file in logs/)

    Returns:
        Logger instance
    """
    global _LOGGING_INITIALIZED, _LOGFILE_PATH

    base = logging.getLogger("UltraRAG")
    lvl = (
        normalize_log_level(level)
        if level is not None
        else base.level
        if _LOGGING_INITIALIZED
        else normalize_log_level(os.environ.get("log_level", "info"))
    )

    if not _LOGGING_INITIALIZED:
        os.makedirs("logs", exist_ok=True)

        if log_file:
            _LOGFILE_PATH = log_file
        else:
            ts = os.environ.get(
                "ULTRARAG_LOG_TS"
            ) or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
            _LOGFILE_PATH = os.path.join("logs", f"{ts}.log")

        rich_handler = RichHandler(
            console=Console(stderr=True),
            rich_tracebacks=enable_rich_tracebacks,
            omit_repeated_times=False,
        )
        rich_handler.setLevel(lvl)
        rich_handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))

        file_handler = logging.FileHandler(_LOGFILE_PATH, mode="a", encoding="utf-8")
        file_handler.setLevel(lvl)
        file_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
                datefmt="%m/%d/%y %H:%M:%S",
            )
        )

        base.setLevel(lvl)
        base.addHandler(rich_handler)
        base.addHandler(file_handler)
        base.propagate = False

        _LOGGING_INITIALIZED = True

    if lvl != base.level:
        base.setLevel(lvl)
        for h in base.handlers:
            h.setLevel(lvl)

    return base if name == "UltraRAG" else base.getChild(name)
