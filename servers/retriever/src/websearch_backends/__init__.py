from __future__ import annotations

import importlib
from typing import Any, Dict, Optional, Type

from .base import BaseWebSearchBackend

_WEBSEARCH_BACKENDS: Dict[str, str] = {
    "exa": ".exa_backend.ExaWebSearchBackend",
    "tavily": ".tavily_backend.TavilyWebSearchBackend",
    "zhipuai": ".zhipuai_backend.ZhipuaiWebSearchBackend",
}


def create_websearch_backend(
    name: str,
    logger,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> BaseWebSearchBackend:
    backend_key = name.lower()
    if backend_key not in _WEBSEARCH_BACKENDS:
        raise ValueError(
            f"Unsupported websearch backend '{name}'. "
            f"Available options: {', '.join(sorted(_WEBSEARCH_BACKENDS))}."
        )

    module_path, class_name = _WEBSEARCH_BACKENDS[backend_key].rsplit(".", 1)
    try:
        module = importlib.import_module(module_path, package=__package__)
        backend_cls: Type[BaseWebSearchBackend] = getattr(module, class_name)
    except ImportError as e:
        raise ImportError(
            f"Backend '{backend_key}' requires optional dependency not installed.\n"
            f"Original error: {e}"
        )
    except AttributeError:
        raise ImportError(f"Class '{class_name}' not found in module '{module_path}'")

    return backend_cls(config=config or {}, logger=logger, **kwargs)


__all__ = [
    "BaseWebSearchBackend",
    "create_websearch_backend",
]
