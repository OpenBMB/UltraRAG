from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
from urllib.parse import urlparse
from tqdm import tqdm

from .base import BaseIndexBackend

try:
    from pymilvus import MilvusClient
except ImportError: 
    MilvusClient = None  


class MilvusIndexBackend(BaseIndexBackend):
    def __init__(
        self,
        contents: Sequence[str],
        config: Optional[dict[str, Any]],
        logger,
        **_: Any,
    ) -> None:
        
        if MilvusClient is None:
            err_msg = (
                "pymilvus is not installed. Install it with `pip install pymilvus` "
                "or include it in the retriever extras."
            )
            logger.error(err_msg)
            raise ImportError(err_msg)

        super().__init__(contents=contents, config=config, logger=logger)


        self.uri = str(self._resolve_index_path(self.config.get("uri")))  
        self.token = self.config.get("token")
        self.collection_name = self.config.get("collection_name")

        self.id_field: str = str(self.config.get("id_field_name", "id"))
        self.vector_field: str = str(self.config.get("vector_field_name", "vector"))
        
        self.metric_type: str = str(self.config.get("metric_type", "IP"))
        self.index_params: dict[str, Any] = dict(self.config.get("index_params"))
        self.search_params = dict(self.config.get("search_params"))

        self.client = None
        self.collection_loaded = False
        
    def _resolve_index_path(self, index_path) -> str:
        if not index_path:
            raise ValueError("[milvus] 'uri' (index_path) is required in config.")

        parsed = urlparse(str(index_path))
        if parsed.scheme in {"http", "https", "tcp"}:
            return str(index_path)
        
        dir_path = os.path.dirname(index_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        path = Path(index_path).expanduser().resolve()
        return os.fspath(path)

 
    def _client_connect(self):
        if self.client is None:
            if self.token:
                self.client = MilvusClient(uri=self.uri, token=self.token)
            else:
                self.client = MilvusClient(self.uri)
        return self.client

    def _ensure_collection(self, dim: int, overwrite: bool) -> None:
        client = self._client_connect()

        if overwrite:
            try:
                client.drop_collection(self.collection_name)
                self.logger.info(f"[milvus] Dropped existing collection '{self.collection_name}'.")
            except Exception:
                pass

        client.create_collection(
            collection_name=self.collection_name,
            dimension=int(dim),
            primary_field_name=self.id_field,
            id_type="int",
            vector_field_name=self.vector_field,
            metric_type=self.metric_type,
            auto_id=False,
            index_params=self.index_params,
        )
        self.collection_loaded = True
        self.logger.info(f"[milvus] Created new collection '{self.collection_name}'.")




    def load_index(self) -> None: 
        self._client_connect()
        try:
            _ = self.client.describe_collection(self.collection_name)
            self.collection_loaded = True  
        except Exception:
            self.collection_loaded = False
            self.logger.info(
                "[milvus] Collection '%s' not found yet (this is OK before build).",
                self.collection_name,
            )

    def build_index(
        self,
        *,
        embeddings: np.ndarray,
        ids: np.ndarray,
        overwrite: bool = False,
    ) -> None:
        
        if not overwrite and self.collection_loaded:
            self.logger.info(
                f"[milvus] Collection '{self.collection_name}' already loaded, skip creation."
            )
            return
        
        client = self._client_connect()
        
        embeddings = np.asarray(embeddings, dtype=np.float32, order="C")
        ids = np.asarray(ids, dtype=np.int64)
        if embeddings.ndim != 2:
            raise ValueError("[milvus] embeddings must be a 2-D array.")
        if ids.ndim != 1 or ids.shape[0] != embeddings.shape[0]:
            raise ValueError("[milvus] ids must align with embeddings.")

        dim = int(embeddings.shape[1])
        self._ensure_collection(dim=dim, overwrite=overwrite)

        total = embeddings.shape[0]
        info_msg = f"[milvus] Inserting {total} vectors into collection {self.collection_name}."
        self.logger.info(info_msg)
        
        index_chunk_size = int(self.config.get("index_chunk_size"))
        with tqdm(
            total=total,
            desc="[milvus] Indexing: ",
            unit="vec",
        ) as pbar:
            for start in range(0, total, index_chunk_size):
                end = min(start + index_chunk_size, total)
                rows = [
                    {self.id_field: int(i), self.vector_field: vec}
                    for i, vec in zip(ids[start:end].tolist(), embeddings[start:end].tolist())
                ]
                client.insert(collection_name=self.collection_name, data=rows)  
                pbar.update(end - start)

        try:
            client.create_index(
                collection_name=self.collection_name,  
                field_name=self.vector_field,
                index_params=self.index_params,
            )
        except Exception:
            pass

        self.logger.info("[milvus] Index ready on collection '%s'.", self.collection_name)

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int,
    ) -> List[List[str]]:

        client = self._client_connect()

        query_embeddings = np.asarray(query_embeddings, dtype=np.float32, order="C")
        if query_embeddings.ndim != 2:
            raise ValueError("[milvus] query embeddings must be 2-D.")

        # MilvusClient.search returns list[list[{id, distance, entity}]]
        try:
            res = client.search(
                collection_name=self.collection_name,
                data=query_embeddings.tolist(),
                limit=int(top_k),
                search_params=self.search_params,
                output_fields=[self.id_field],  # we only need ids
            )
        except Exception as exc:  
            raise RuntimeError(f"[milvus] Search failed: {exc}") from exc

        ret = []
        for hits in res:
            row = []
            for hit in hits:
                doc_id = hit.get("id") if isinstance(hit, dict) else getattr(hit, "id", None)
                did = int(doc_id)
                row.append(self.contents[did])
            ret.append(row)
        return ret

