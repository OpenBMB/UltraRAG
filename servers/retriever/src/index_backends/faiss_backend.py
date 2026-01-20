from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from .base import BaseIndexBackend
from fastmcp.exceptions import ValidationError

try:
    import faiss
except ImportError:
    faiss = None


class FaissIndexBackend(BaseIndexBackend):
    """FAISS-based index backend for vector similarity search."""

    def __init__(
        self,
        contents: Sequence[str],
        config: Optional[dict[str, Any]],
        logger,
        *,
        device_num: int = 1,
        **_: Any,
    ) -> None:
        """Initialize FAISS index backend.

        Args:
            contents: Sequence of document contents
            config: Configuration dictionary
            logger: Logger instance
            device_num: Number of GPU devices to use
        """
        if faiss is None:
            err_msg = (
                "faiss is not installed. "
                "Please install it with `pip install faiss-cpu` "
                "or `pip install faiss-gpu-cu12`."
            )
            logger.error(err_msg)
            raise ImportError(err_msg)

        super().__init__(contents=contents, config=config, logger=logger)
        self.use_gpu = self.config.get("index_use_gpu")
        self.device_num = max(1, int(device_num or 1))
        self.index_path = self._resolve_index_path(self.config.get("index_path"))
        self.index = None

    def _resolve_index_path(self, index_path: Optional[str]) -> str:
        if index_path:
            path = Path(index_path).expanduser().resolve()
            return os.fspath(path)

        project_root = Path(__file__).resolve().parents[2]
        path = project_root / "output" / "index" / "index.index"
        return os.fspath(path)

    def _maybe_to_gpu(self, cpu_index):
        if not self.use_gpu:
            return cpu_index
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        try:
            gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co)
            info_msg = (
                f"[faiss] Loaded index to GPU(s) with {self.device_num} device(s)."
            )
            self.logger.info(info_msg)
            return gpu_index
        except RuntimeError as e:
            warn_msg = f"[faiss] GPU index load failed: {e}. Falling back to CPU."
            self.logger.warning(warn_msg)
            self.use_gpu = False
            return cpu_index

    def load_index(self) -> None:
        """Load existing FAISS index from disk.

        If index file doesn't exist, initializes without index.
        """
        if not os.path.exists(self.index_path):
            info_msg = (
                f"[faiss] Index path '{self.index_path}' does not exist. "
                "Retriever initialized without index."
            )
            self.logger.info(info_msg)
            self.index = None
        else:
            cpu_index = faiss.read_index(self.index_path)
            self.index = self._maybe_to_gpu(cpu_index)
            if self.use_gpu:
                self.logger.info("[faiss] Index loaded on GPU(s).")
            else:
                self.logger.info("[faiss] Index loaded on CPU.")

    def build_index(
        self,
        *,
        embeddings: np.ndarray,
        ids: np.ndarray,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        """Build FAISS index from embeddings.

        Args:
            embeddings: 2D numpy array of embeddings
            ids: 1D numpy array of vector IDs
            overwrite: Whether to overwrite existing index
            **kwargs: Additional parameters (index_chunk_size from config)

        Raises:
            ValidationError: If index_path format is invalid
            ValueError: If embeddings or ids have invalid shapes
        """
        if not self.index_path.endswith(".index"):
            err_msg = (
                f"Parameter 'index_path' must end with '.index', "
                f"got '{self.index_path}'"
            )
            raise ValidationError(err_msg)

        if not overwrite and os.path.exists(self.index_path):
            info_msg = (
                f"Index file already exists: {self.index_path}. "
                "Set overwrite=True to overwrite."
            )
            self.logger.info(info_msg)
            return

        dir_path = os.path.dirname(self.index_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        embeddings = np.asarray(embeddings, dtype=np.float32, order="C")
        ids = np.asarray(ids, dtype=np.int64)
        if embeddings.ndim != 2:
            raise ValueError("[faiss] embeddings must be 2-D array.")
        if ids.ndim != 1 or ids.shape[0] != embeddings.shape[0]:
            raise ValueError("[faiss] ids must be 1-D array aligned with embeddings.")

        dim = embeddings.shape[1]
        cpu_flat = faiss.IndexFlatIP(dim)
        cpu_index = faiss.IndexIDMap2(cpu_flat)

        total = embeddings.shape[0]
        info_msg = f"Start building FAISS index, total vectors: {total}"
        self.logger.info(info_msg)

        index_chunk_size = int(
            self.config.get("index_chunk_size", kwargs.get("index_chunk_size", 50000))
        )
        with tqdm(
            total=total,
            desc="[faiss] Indexing: ",
            unit="vec",
        ) as pbar:
            for start in range(0, total, index_chunk_size):
                end = min(start + index_chunk_size, total)
                cpu_index.add_with_ids(embeddings[start:end], ids[start:end])
                pbar.update(end - start)

        faiss.write_index(cpu_index, self.index_path)
        self.logger.info("[faiss] Index written to '%s'.", self.index_path)

        self.index = self._maybe_to_gpu(cpu_index)

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int,
        **kwargs: Any,
    ) -> List[List[str]]:
        """Search for similar passages using FAISS index.

        Args:
            query_embeddings: 2D numpy array of query embeddings
            top_k: Number of top results to return per query
            **kwargs: Additional parameters (unused)

        Returns:
            List of lists, where each inner list contains top_k passage strings

        Raises:
            RuntimeError: If index is not loaded
            ValueError: If query_embeddings has invalid shape
        """
        if self.index is None:
            raise RuntimeError(
                "[faiss] Index is not loaded. "
                "Build the index or provide a valid index_path."
            )

        query_embeddings = np.asarray(query_embeddings, dtype=np.float32, order="C")
        if query_embeddings.ndim != 2:
            raise ValueError("[faiss] query embeddings must be a 2-D array.")

        _, indices = self.index.search(query_embeddings, top_k)
        results = []
        for doc_ids in indices:
            cur_ret = []
            for doc_id in doc_ids:
                cur_ret.append(self.contents[doc_id])
            results.append(cur_ret)
        return results
