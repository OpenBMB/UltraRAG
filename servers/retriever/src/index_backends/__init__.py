from __future__ import annotations
from typing import Any, Dict, Optional, Sequence, Type
import importlib

from .base import BaseIndexBackend  


_INDEX_BACKENDS: Dict[str, str] = {
    "faiss": ".faiss_backend.FaissIndexBackend",
    "milvus": ".milvus_backend.MilvusIndexBackend",
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

    module_path, class_name = _INDEX_BACKENDS[backend_key].rsplit(".", 1)
    try:
        module = importlib.import_module(module_path, package=__package__)
        backend_cls: Type[BaseIndexBackend] = getattr(module, class_name)
    except ImportError as e:
        raise ImportError(
            f"Backend '{backend_key}' requires optional dependency not installed.\n"
            f"Original error: {e}"
        )
    except AttributeError:
        raise ImportError(f"Class '{class_name}' not found in module '{module_path}'")

    return backend_cls(contents=contents, config=config or {}, logger=logger, **kwargs)


__all__ = [
    "BaseIndexBackend",
    "create_index_backend",
]