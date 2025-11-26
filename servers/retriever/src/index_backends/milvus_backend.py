from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
from urllib.parse import urlparse
from tqdm import tqdm

from .base import BaseIndexBackend

try:
    from pymilvus import MilvusClient, DataType
except ImportError:
    MilvusClient = None
    DataType = None 


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
        self.text_field: str = str(self.config.get("text_field_name", "content")) 
        
        self.metric_type: str = str(self.config.get("metric_type", "IP"))
        self.index_params: dict[str, Any] = dict(self.config.get("index_params"))
        self.search_params = dict(self.config.get("search_params"))

        self.id_max_length = int(self.config.get("id_max_length", 512))
        self.text_max_length = int(self.config.get("text_max_length", 60000))

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

    def _ensure_collection(
        self, 
        dim: int, 
        overwrite: bool, 
        collection_name: str,
        use_string_id: bool = False,
        store_content: bool = False
    ) -> None:
        client = self._client_connect()

        has_collection = client.has_collection(collection_name)

        if overwrite and has_collection:
            try:
                client.drop_collection(collection_name)
                self.logger.info(f"[milvus] Dropped existing collection '{collection_name}'.")
                has_collection = False
            except Exception as e:
                self.logger.warning(f"[milvus] Failed to drop collection: {e}")

        if has_collection:
            try:
                desc = client.describe_collection(collection_name)
                
                existing_dim = -1
                
                fields = desc.get("fields", []) if isinstance(desc, dict) else getattr(desc, "fields", [])
                
                for f in fields:
                    f_dict = f if isinstance(f, dict) else f.to_dict() if hasattr(f, "to_dict") else {}
                    params = f_dict.get("params", {})
                    if "dim" in params:
                        existing_dim = int(params["dim"])
                        break
                
                if existing_dim != -1 and existing_dim != dim:
                    err_msg = (
                        f"[milvus] Schema Mismatch! Existing collection '{collection_name}' has dim={existing_dim}, "
                        f"but input data has dim={dim}. "
                        f"Please set 'overwrite=True' to rebuild, or check your embedding model."
                    )
                    self.logger.error(err_msg)
                    raise ValueError(err_msg)
                
                self.logger.info(f"[milvus] Collection '{collection_name}' exists and schema check passed (or skipped). Appending data...")
                return

            except Exception as e:
                if "Schema Mismatch" in str(e):
                    raise e
                self.logger.warning(f"[milvus] Skip schema check due to error: {e}. Proceeding to insert...")
                return
        
        self.logger.info(f"[milvus] Defining schema for '{collection_name}' (StringID={use_string_id}, StoreContent={store_content})...")

        schema = MilvusClient.create_schema(
            auto_id=False, 
            enable_dynamic_field=True,
            description="Created by UltraRAG Retriever"
        )

        if use_string_id:
            schema.add_field(
                field_name=self.id_field, 
                datatype=DataType.VARCHAR, 
                max_length=self.id_max_length, 
                is_primary=True
            )
        else:
            schema.add_field(
                field_name=self.id_field, 
                datatype=DataType.INT64, 
                is_primary=True
            )

        schema.add_field(
            field_name=self.vector_field, 
            datatype=DataType.FLOAT_VECTOR, 
            dim=dim
        )

        if store_content:
            schema.add_field(
                field_name=self.text_field,
                datatype=DataType.VARCHAR,
                max_length=self.text_max_length,
                description="Original document content"
            )

        index_params = client.prepare_index_params()
        
        index_params.add_index(
            field_name=self.vector_field,
            metric_type=self.metric_type,
            index_type=self.index_params.get("index_type", "AUTOINDEX"),
            params=self.index_params.get("params", {})
        )

        try:
            client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params
            )
            self.collection_loaded = True
            self.logger.info(f"[milvus] Successfully created collection '{collection_name}'.")
            
        except Exception as e:
            self.logger.error(f"[milvus] Failed to create collection: {e}")
            raise RuntimeError(f"Milvus create collection failed: {e}")



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
        **kwargs: Any
    ) -> None:
        
        client = self._client_connect()
        target_collection = kwargs.get("collection_name", self.collection_name)

        passed_contents = kwargs.get("contents", None)
        passed_metadatas = kwargs.get("metadatas", None)
        store_content = passed_contents is not None and len(passed_contents) > 0
        
        embeddings = np.asarray(embeddings, dtype=np.float32, order="C")
        ids = np.array(ids)
        use_string_id = False
        if ids.dtype.kind in {'U', 'S', 'O'}: # Unicode, String, Object
            use_string_id = True
        else:
            ids = ids.astype(np.int64)
        if embeddings.ndim != 2:
            raise ValueError("[milvus] embeddings must be a 2-D array.")
        if ids.shape[0] != embeddings.shape[0]:
            raise ValueError("[milvus] ids must align with embeddings.")

        dim = int(embeddings.shape[1])

        self._ensure_collection(
            dim=dim, 
            overwrite=overwrite, 
            collection_name=target_collection,
            use_string_id=use_string_id,
            store_content=store_content
        )

        total = embeddings.shape[0]
        self.logger.info(f"[milvus] Inserting {total} vectors into '{target_collection}'.")
        
        index_chunk_size = int(self.config.get("index_chunk_size"))
        with tqdm(
            total=total,
            desc="[milvus] Indexing: ",
            unit="vec",
        ) as pbar:
            for start in range(0, total, index_chunk_size):
                end = min(start + index_chunk_size, total)

                batch_ids = ids[start:end].tolist()
                batch_vecs = embeddings[start:end].tolist()

                rows = []
                for i, (doc_id, vec) in enumerate(zip(batch_ids, batch_vecs)):
                    row = {
                        self.id_field: doc_id,  
                        self.vector_field: vec
                    }
                    
                    if store_content and passed_contents:
                        global_idx = start + i
                        if global_idx < len(passed_contents):
                            row[self.text_field] = passed_contents[global_idx]
                    
                    if passed_metadatas:
                        global_idx = start + i
                        if global_idx < len(passed_metadatas):
                            meta = passed_metadatas[global_idx]
                            if isinstance(meta, dict):
                                row.update(meta)

                    rows.append(row)

                client.insert(collection_name=target_collection, data=rows)
                pbar.update(end - start)

        try:
            client.flush(target_collection)
            self.logger.info(f"[milvus] Flushed collection '{target_collection}'.")
        except Exception as e:
            self.logger.warning(f"[milvus] Flush failed: {e}")

        try:
            client.load_collection(target_collection)
        except Exception:
            pass
        
        
        self.logger.info("[milvus] Index ready on collection '%s'.", target_collection)

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int,
        **kwargs: Any
    ) -> List[List[str]]:

        client = self._client_connect()
        target_collection = kwargs.get("collection_name", self.collection_name)

        query_embeddings = np.asarray(query_embeddings, dtype=np.float32, order="C")
        if query_embeddings.ndim != 2:
            raise ValueError("[milvus] query embeddings must be 2-D.")
        
        output_fields = [self.id_field]
        fetch_content_from_db = False
        
        if not self.contents:
            output_fields.append(self.text_field)
            fetch_content_from_db = True

        try:
            res = client.search(
                collection_name=target_collection,
                data=query_embeddings.tolist(),
                limit=int(top_k),
                search_params=self.search_params,
                output_fields=output_fields, 
                consistency_level="Strong"
            )
        except Exception as exc:  
            raise RuntimeError(f"[milvus] Search failed on '{target_collection}': {exc}") from exc

        ret = []
        for hits in res:
            row = []
            for hit in hits:
                entity = hit.get("entity") if isinstance(hit, dict) else getattr(hit, "entity", {})
                
                if fetch_content_from_db:
                    content = entity.get(self.text_field)
                    if content is None and isinstance(hit, dict):
                         content = hit.get(self.text_field)
                    
                    row.append(content if content is not None else "")
                else:
                    doc_id = hit.get("id") if isinstance(hit, dict) else getattr(hit, "id", None)
                    try:
                        did = int(doc_id)
                        row.append(self.contents[did])
                    except (ValueError, IndexError):
                        row.append("") 
            ret.append(row)
        return ret

