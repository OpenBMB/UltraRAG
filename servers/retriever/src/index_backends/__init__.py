from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from .base import BaseIndexBackend  
from .faiss_backend import FaissIndexBackend

_INDEX_BACKENDS = {
    "faiss": FaissIndexBackend,
}


def create_index_backend(
    name: str,
    contents: Sequence[str],
    logger,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> BaseIndexBackend:
    backend_key = name.lower()
    if backend_key not in _INDEX_BACKENDS:
        raise ValueError(
            f"Unsupported index backend '{name}'. "
            f"Available options: {', '.join(sorted(_INDEX_BACKENDS))}."
        )
    backend_cls = _INDEX_BACKENDS[backend_key]
    return backend_cls(contents=contents, config=config or {}, logger=logger, **kwargs)


__all__ = [
    "BaseIndexBackend",
    "FaissIndexBackend",
    "create_index_backend",
]
