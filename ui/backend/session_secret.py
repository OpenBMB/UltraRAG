"""Persist an installation-specific signing key without a shared fallback."""

import os
import secrets
import tempfile
from pathlib import Path


def load_session_secret(path: Path) -> str:
    """Honor explicit configuration, otherwise atomically publish a private key."""
    configured = os.environ.get("ULTRARAG_SESSION_SECRET")
    if configured and configured.strip():
        return configured
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="ascii", dir=path.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(secrets.token_hex(32))
                stream.flush()
                os.fsync(stream.fileno())
            try:
                # A hard link publishes complete content and never replaces a
                # key created concurrently by another application worker.
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    value = path.read_text(encoding="ascii").strip()
    if len(value) < 32:
        raise RuntimeError(f"Invalid session signing key file: {path}")
    return value
