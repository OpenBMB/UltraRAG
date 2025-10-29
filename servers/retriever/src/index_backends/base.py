from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class BaseIndexBackend(abc.ABC):
    def __init__(
        self,
        contents: Sequence[str],
        config: Optional[Dict[str, Any]],
        logger,
        **_: Any,
    ) -> None:
        self.contents: List[str] = list(contents)
        self.config: Dict[str, Any] = dict(config or {})
        self.logger = logger


    @abc.abstractmethod
    def load_index(self, *, index_path: Optional[str] = None) -> None:
        ...

    @abc.abstractmethod
    def build_index(
        self,
        *,
        embeddings: np.ndarray,
        ids: np.ndarray,
        index_path: Optional[str] = None,
        overwrite: bool = False,
        index_chunk_size: int = 50000,
    ) -> None:
        ...

    @abc.abstractmethod
    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int,
    ) -> List[List[str]]:
        ...

    def close(self) -> None:
        """Optional hook for releasing resources."""
        return None
