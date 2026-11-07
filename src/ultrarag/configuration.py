"""Validated YAML loading shared by orchestration and server metadata builds."""

import os
import tempfile
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class Configuration:
    """Load environment defaults and require mapping-shaped YAML documents."""

    def __init__(self) -> None:
        self.load_env()

    @staticmethod
    def load_env() -> None:
        load_dotenv()

    @staticmethod
    def load_config(file_path: str | Path) -> dict[str, Any]:
        """Read a YAML mapping; empty documents represent an empty mapping."""
        path = Path(file_path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError(f"Configuration must be a YAML mapping: {path}")  # noqa: TRY004 - invalid document value
        return data

    @staticmethod
    def load_parameter_config(file_path: str | Path) -> dict[str, Any]:
        """Read optional parameters, preserving missing-file compatibility."""
        path = Path(file_path)
        return Configuration.load_config(path) if path.is_file() else {}

    @staticmethod
    def save_config(
        file_path: str | Path, data: dict[str, Any], *, overwrite: bool = True
    ) -> None:
        """Atomically replace a YAML document without leaving truncated files."""
        Configuration.save_text(
            file_path,
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            overwrite=overwrite,
        )

    @staticmethod
    def save_text(file_path: str | Path, text: str, *, overwrite: bool = True) -> None:
        """Publish complete UTF-8 text, optionally requiring a new destination."""
        path = Path(file_path)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(text)
            if overwrite:
                os.replace(temporary, path)
            else:
                os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
