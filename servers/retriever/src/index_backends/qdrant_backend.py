from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .base import BaseIndexBackend

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, QueryRequest, VectorParams
except ImportError:
    QdrantClient = Distance = PointStruct = QueryRequest = VectorParams = None

_INTERNAL_KEYS = frozenset({"text_field_name", "distance", "index_chunk_size", "collection_name"})


def _to_qdrant_id(val: Any) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_DNS, str(val))


class QdrantIndexBackend(BaseIndexBackend):
    """Qdrant-based index backend for vector similarity search."""

    def __init__(
        self,
        contents: Sequence[str],
        config: Optional[dict[str, Any]],
        logger,
        **_: Any,
    ) -> None:
        if QdrantClient is None:
            raise ImportError(
                "qdrant-client is not installed. "
                "Install it with `pip install qdrant-client`."
            )
        super().__init__(contents=[], config=config, logger=logger)

        self.collection_name: Optional[str] = self.config.get("collection_name")
        self.text_field: str = self.config.get("text_field_name", "contents")
        self.chunk_size: int = int(self.config.get("index_chunk_size", 50000))
        self._distance_str: str = self.config.get("distance", "Cosine").lower()
        self.client: Optional[QdrantClient] = None

    def _connect(self) -> QdrantClient:
        if self.client is None:
            kw = {k: v for k, v in self.config.items() if k not in _INTERNAL_KEYS and v is not None}
            self.client = QdrantClient(**kw) if kw else QdrantClient(":memory:")
        return self.client

    def _distance(self) -> Any:
        mapping = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "ip": Distance.DOT,
            "euclid": Distance.EUCLID,
            "l2": Distance.EUCLID,
            "manhattan": Distance.MANHATTAN,
        }
        dist = mapping.get(self._distance_str)
        if dist is None:
            self.logger.warning(
                "[qdrant] Unknown distance '%s'; defaulting to Cosine.",
                self._distance_str,
            )
            return Distance.COSINE
        return dist

    def _ensure_collection(self, name: str, dim: int, overwrite: bool) -> None:
        client = self._connect()
        if overwrite and client.collection_exists(name):
            client.delete_collection(name)
        if not client.collection_exists(name):
            client.create_collection(
                name, vectors_config=VectorParams(size=dim, distance=self._distance())
            )

    def load_index(self) -> None:
        self._connect()

    def build_index(
        self,
        *,
        embeddings: np.ndarray,
        ids: np.ndarray,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        col = kwargs.get("collection_name", self.collection_name)
        contents = kwargs.get("contents")
        metadatas = kwargs.get("metadatas") or []

        if not col:
            raise ValueError("[qdrant] 'collection_name' is required.")
        if not contents:
            raise ValueError("[qdrant] 'contents' is required.")

        embeddings = np.asarray(embeddings, dtype=np.float32, order="C")
        if embeddings.ndim != 2:
            raise ValueError("[qdrant] embeddings must be 2-D.")
        if len(ids) != len(embeddings):
            raise ValueError("[qdrant] ids and embeddings must have the same length.")

        self._ensure_collection(col, embeddings.shape[1], overwrite)
        client = self._connect()

        for start in range(0, len(embeddings), self.chunk_size):
            end = min(start + self.chunk_size, len(embeddings))
            client.upsert(
                collection_name=col,
                points=[
                    PointStruct(
                        id=_to_qdrant_id(ids[i]),
                        vector=embeddings[i].tolist(),
                        payload={
                            self.text_field: contents[i],
                            **(
                                metadatas[i]
                                if i < len(metadatas) and isinstance(metadatas[i], dict)
                                else {}
                            ),
                        },
                    )
                    for i in range(start, end)
                ],
            )

        self.logger.info("[qdrant] Indexed %d vectors into '%s'.", len(embeddings), col)

    def search(
        self, query_embeddings: np.ndarray, top_k: int, **kwargs: Any
    ) -> List[List[str]]:
        col = kwargs.get("collection_name", self.collection_name)
        if not col:
            raise ValueError("[qdrant] 'collection_name' is required.")

        query_embeddings = np.asarray(query_embeddings, dtype=np.float32, order="C")
        if query_embeddings.ndim != 2:
            raise ValueError("[qdrant] query embeddings must be 2-D.")

        try:
            responses = self._connect().query_batch_points(
                collection_name=col,
                requests=[
                    QueryRequest(query=row.tolist(), limit=top_k, with_payload=True)
                    for row in query_embeddings
                ],
            )
        except Exception as exc:
            raise RuntimeError(f"[qdrant] Search failed on '{col}': {exc}") from exc

        return [
            [str((hit.payload or {}).get(self.text_field, "")) for hit in resp.points]
            for resp in responses
        ]

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
