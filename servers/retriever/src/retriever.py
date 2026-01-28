import asyncio
import gc
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiohttp
import orjson
import numpy as np
from tqdm import tqdm
from PIL import Image
import uuid

from fastmcp.exceptions import ValidationError, NotFoundError, ToolError
from ultrarag.server import UltraRAG_MCP_Server
from index_backends import BaseIndexBackend, create_index_backend
from websearch_backends import create_websearch_backend

app = UltraRAG_MCP_Server("retriever")


class Retriever:
    def __init__(self, mcp_inst: UltraRAG_MCP_Server):
        mcp_inst.tool(
            self.retriever_init,
            output="model_name_or_path,backend_configs,batch_size,corpus_path,gpu_ids,is_multimodal,backend,index_backend,index_backend_configs,is_demo,collection_name->None",
        )
        mcp_inst.tool(
            self.retriever_embed,
            output="embedding_path,overwrite,is_multimodal->None",
        )
        mcp_inst.tool(
            self.retriever_index,
            output="embedding_path,overwrite,collection_name,corpus_path->None",
        )
        mcp_inst.tool(
            self.retriever_search,
            output="q_ls,top_k,query_instruction,collection_name->ret_psg",
        )
        mcp_inst.tool(
            self.retriever_batch_search,
            output="batch_query_list,top_k,query_instruction,collection_name->ret_psg_ls",
        )
        mcp_inst.tool(
            self.bm25_index,
            output="overwrite->None",
        )
        mcp_inst.tool(
            self.bm25_search,
            output="q_ls,top_k->ret_psg",
        )
        mcp_inst.tool(
            self.retriever_deploy_search,
            output="retriever_url,q_ls,top_k,query_instruction->ret_psg",
        )
        mcp_inst.tool(
            self.retriever_websearch,
            output="q_ls,top_k,retrieve_thread_num,websearch_backend,websearch_backend_configs->ret_psg",
        )
        mcp_inst.tool(
            self.retriever_batch_websearch,
            output="batch_query_list,top_k,retrieve_thread_num,websearch_backend,websearch_backend_configs->ret_psg_ls",
        )

    def _drop_keys(self, d: Dict[str, Any], banned: List[str]) -> Dict[str, Any]:
        """Remove banned keys and None values from dictionary.

        Args:
            d: Dictionary to filter
            banned: List of keys to remove

        Returns:
            Filtered dictionary
        """
        return {k: v for k, v in (d or {}).items() if k not in banned and v is not None}

    async def retriever_init(
        self,
        model_name_or_path: str,
        backend_configs: Dict[str, Any],
        batch_size: int,
        corpus_path: str,
        gpu_ids: Optional[Union[str, int]] = None,
        is_multimodal: bool = False,
        backend: str = "sentence_transformers",
        index_backend: str = "faiss",
        index_backend_configs: Optional[Dict[str, Any]] = None,
        is_demo: bool = False,
        collection_name: str = "",
    ) -> None:
        """Initialize retriever with specified backend and index backend.

        Args:
            model_name_or_path: Model name or path for embedding
            backend_configs: Dictionary of backend-specific configurations
            batch_size: Batch size for embedding generation
            corpus_path: Path to corpus file (JSONL format)
            gpu_ids: Comma-separated GPU IDs (e.g., "0,1")
            is_multimodal: Whether to use multimodal (image) embeddings
            backend: Backend name ("infinity", "sentence_transformers", "openai", or "bm25")
            index_backend: Index backend name ("faiss" or "milvus")
            index_backend_configs: Dictionary of index backend configurations
            is_demo: Whether to run in demo mode (forces OpenAI + Milvus)
            collection_name: Collection name for Milvus backend

        Raises:
            ImportError: If required dependencies are not installed
            ValueError: If required config is missing
            ValidationError: If demo mode requirements are not met
        """
        self.is_demo = is_demo
        self.batch_size = batch_size
        self.corpus_path = corpus_path
        self.backend_configs = backend_configs
        self.index_backend_configs = index_backend_configs or {}

        if self.is_demo:
            app.logger.info("[retriever] Initializing in DEMO mode.")
            self.backend = "openai"
            self.index_backend_name = "milvus"

            if "openai" not in self.backend_configs:
                raise ValidationError(
                    "is_demo=True requires 'openai' in backend_configs."
                )
            if "milvus" not in self.index_backend_configs:
                raise ValidationError(
                    "is_demo=True requires 'milvus' in index_backend_configs."
                )

            app.logger.info(
                "[retriever] Demo mode enforced: Backend=OpenAI, Index=Milvus."
            )
        else:
            self.backend = backend.lower()
            self.index_backend_name = index_backend.lower()

            if self.index_backend_name == "milvus":
                app.logger.warning(
                    "[retriever] Using Milvus in non-demo mode is not recommended in this simplified architecture."
                )

        self.cfg = self.backend_configs.get(self.backend, {})

        if gpu_ids is None:
            self.gpu_ids = None
            self.device = "cpu"
            self.device_num = 1
            app.logger.info("[retriever] gpu_ids is None, treat as CPU-only mode.")
        else:
            gpu_ids = str(gpu_ids)
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
            self.gpu_ids = gpu_ids
            self.device = "cuda"
            self.device_num = len(gpu_ids.split(","))
            app.logger.info(
                "[retriever] Set CUDA_VISIBLE_DEVICES=%s, device_num=%d",
                gpu_ids,
                self.device_num,
            )

        if self.backend == "infinity":
            try:
                from infinity_emb import AsyncEngineArray, EngineArgs
            except ImportError:
                err_msg = "infinity_emb is not installed. Please install it with `pip install infinity-emb`."
                app.logger.error(err_msg)
                raise ImportError(err_msg)

            infinity_engine_args = EngineArgs(
                model_name_or_path=model_name_or_path,
                batch_size=self.batch_size,
                device=self.device,
                **self.cfg,
            )
            self.model = AsyncEngineArray.from_args([infinity_engine_args])[0]

        elif self.backend == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                err_msg = (
                    "sentence_transformers is not installed. "
                    "Please install it with `pip install sentence-transformers`."
                )
                app.logger.error(err_msg)
                raise ImportError(err_msg)
            self.st_encode_params = (
                self.cfg.get("sentence_transformers_encode", {}) or {}
            )
            st_params = self._drop_keys(
                self.cfg, banned=["sentence_transformers_encode"]
            )

            self.model = SentenceTransformer(
                model_name_or_path=model_name_or_path,
                device=self.device,
                **st_params,
            )

        elif self.backend == "openai":
            try:
                from openai import AsyncOpenAI
            except ImportError:
                err_msg = (
                    "openai is not installed. "
                    "Please install it with `pip install openai`."
                )
                app.logger.error(err_msg)
                raise ImportError(err_msg)

            model_name = self.cfg.get("model_name")
            base_url = self.cfg.get("base_url")
            api_key = self.cfg.get("api_key") or os.environ.get("RETRIEVER_API_KEY")

            if not model_name:
                err_msg = "[openai] model_name is required"
                app.logger.error(err_msg)
                raise ValueError(err_msg)
            if not isinstance(base_url, str) or not base_url:
                err_msg = "[openai] base_url must be a non-empty string"
                app.logger.error(err_msg)
                raise ValueError(err_msg)

            try:
                self.model = AsyncOpenAI(base_url=base_url, api_key=api_key)
                self.model_name = model_name
                info_msg = (
                    f"[openai] OpenAI client initialized "
                    f"(model='{model_name}', base='{base_url}')"
                )
                app.logger.info(info_msg)
            except Exception as e:
                err_msg = f"[openai] Failed to initialize OpenAI client: {e}"
                app.logger.error(err_msg)
                raise RuntimeError(err_msg) from e
        elif self.backend == "bm25":
            try:
                import bm25s
            except ImportError:
                err_msg = (
                    "bm25s is not installed. "
                    "Please install it with `pip install bm25s`."
                )
                app.logger.error(err_msg)
                raise ImportError(err_msg)

            try:
                self.model = bm25s.BM25(backend="numba")
            except Exception as e:
                warn_msg = (
                    f"Failed to initialize BM25 model with backend 'numba': {e}. "
                    "Falling back to 'numpy' backend."
                )
                app.logger.warning(warn_msg)
                self.model = bm25s.BM25(backend="numpy")
            lang = self.cfg.get("lang", "en")
            try:
                self.tokenizer = bm25s.tokenization.Tokenizer(stopwords=lang)
            except Exception as e:
                err_msg = (
                    f"Failed to initialize BM25 tokenizer for language '{lang}': {e}"
                )
                app.logger.error(err_msg)
                raise RuntimeError(err_msg)
        else:
            error_msg = (
                f"Unsupported backend: {backend}. "
                "Supported backends: 'infinity', 'sentence_transformers', 'openai'"
            )
            app.logger.error(error_msg)
            raise ValueError(error_msg)

        self.contents = []

        should_load_corpus_to_memory = (self.backend == "bm25") or (
            self.index_backend_name == "faiss"
        )
        if should_load_corpus_to_memory and corpus_path and os.path.exists(corpus_path):
            app.logger.info(
                f"[retriever] Loading corpus to memory for {self.index_backend_name}/BM25..."
            )
            corpus_path_obj = Path(corpus_path)
            corpus_dir = corpus_path_obj.parent
            file_size = os.path.getsize(corpus_path)

            with open(corpus_path, "rb") as f:
                with tqdm(
                    total=file_size,
                    desc="Loading corpus",
                    unit="B",
                    unit_scale=True,
                    ncols=100,
                ) as pbar:
                    bytes_read = 0
                    for i, line in enumerate(f):
                        pbar.update(len(line))
                        bytes_read += len(line)
                        try:
                            item = orjson.loads(line)
                        except orjson.JSONDecodeError as e:
                            raise ToolError(f"Invalid JSON on line {i}: {e}") from e
                        if not is_multimodal or self.backend == "bm25":
                            if "contents" not in item:
                                error_msg = f"Line {i}: missing key 'contents'. full item={item}"
                                app.logger.error(error_msg)
                                raise ValueError(error_msg)

                            self.contents.append(item["contents"])
                        else:
                            if "image_path" not in item:
                                error_msg = f"Line {i}: missing key 'image_path'. full item={item}"
                                app.logger.error(error_msg)
                                raise ValueError(error_msg)

                            rel = str(item["image_path"])
                            abs_path = str((corpus_dir / rel).resolve())
                            self.contents.append(abs_path)
                    if bytes_read < file_size:
                        pbar.update(file_size - bytes_read)
                    pbar.refresh()
        else:
            if self.is_demo:
                app.logger.info(
                    "[retriever] Demo/Milvus mode: Skipping memory corpus load (Data stored in DB)."
                )
            elif not os.path.exists(corpus_path):
                app.logger.warning(f"[retriever] Corpus path not found: {corpus_path}")

        self.index_backend: Optional[BaseIndexBackend] = None
        if self.backend in ["infinity", "sentence_transformers", "openai"]:
            index_backend_cfg = self.index_backend_configs.get(
                self.index_backend_name, {}
            )

            if self.index_backend_name == "milvus":
                index_backend_cfg["collection_name"] = collection_name

            self.index_backend = create_index_backend(
                name=self.index_backend_name,
                contents=self.contents,
                logger=app.logger,
                config=index_backend_cfg,
                device_num=self.device_num,
            )
            app.logger.info(
                "[index] Initialized backend '%s'.", self.index_backend_name
            )
            try:
                self.index_backend.load_index()
            except Exception as exc:
                warn_msg = (
                    f"[index] Failed to load existing index using backend "
                    f"'{self.index_backend_name}': {exc}"
                )
                app.logger.warning(warn_msg)

        elif self.backend == "bm25":
            bm25_save_path = self.cfg.get("save_path", None)
            if bm25_save_path and os.path.exists(bm25_save_path):
                self.model = self.model.load(
                    bm25_save_path, mmap=True, load_corpus=False
                )
                self.tokenizer.load_stopwords(bm25_save_path)
                self.tokenizer.load_vocab(bm25_save_path)
                self.model.corpus = self.contents
                self.model.backend = "numba"
                info_msg = "[bm25] Index loaded successfully."
                app.logger.info(info_msg)
            else:
                if bm25_save_path and not os.path.exists(bm25_save_path):
                    warn_msg = f"{bm25_save_path} does not exist."
                    app.logger.warning(warn_msg)
                info_msg = "[bm25] no index_path provided. Retriever initialized without index."
                app.logger.info(info_msg)

    async def retriever_embed(
        self,
        embedding_path: Optional[str] = None,
        overwrite: bool = False,
        is_multimodal: bool = False,
    ) -> None:
        """Generate embeddings for corpus contents.

        Args:
            embedding_path: Path to save embeddings (.npy file)
            overwrite: Whether to overwrite existing embeddings
            is_multimodal: Whether to generate image embeddings

        Raises:
            ValidationError: If embedding_path format is invalid
            RuntimeError: If embedding generation fails
            ValueError: If backend doesn't support multimodal or is unsupported
        """
        if getattr(self, "is_demo", False):
            warn_msg = (
                "[retriever] 'retriever_embed' is ignored in Demo mode. "
                "Embeddings are generated on-the-fly during 'retriever_index'."
            )
            app.logger.warning(warn_msg)
            return

        if self.backend == "bm25":
            app.logger.info(
                "[retriever] BM25 backend does not support dense embedding generation. Skipping."
            )
            return

        embeddings = None

        if embedding_path is not None:
            if not embedding_path.endswith(".npy"):
                err_msg = (
                    f"Embedding save path must end with .npy, "
                    f"now the path is {embedding_path}"
                )
                app.logger.error(err_msg)
                raise ValidationError(err_msg)
            output_dir = os.path.dirname(embedding_path)
        else:
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))
            output_dir = os.path.join(project_root, "output", "embedding")
            embedding_path = os.path.join(output_dir, "embedding.npy")

        if not overwrite and os.path.exists(embedding_path):
            app.logger.info("Embedding already exists, skipping")
            return

        os.makedirs(output_dir, exist_ok=True)

        if self.backend == "infinity":
            async with self.model:
                if is_multimodal:
                    data = []
                    for i, p in enumerate(self.contents):
                        try:
                            with Image.open(p) as im:
                                data.append(im.convert("RGB").copy())
                        except Exception as e:
                            err_msg = f"Failed to load image at index {i}: {p} ({e})"
                            app.logger.error(err_msg)
                            raise RuntimeError(err_msg)
                    call = self.model.image_embed
                else:
                    data = self.contents
                    call = self.model.embed

                eff_bs = self.batch_size * self.device_num
                n = len(data)
                pbar = tqdm(total=n, desc="[infinity] Embedding:")
                embeddings = []
                for i in range(0, n, eff_bs):
                    chunk = data[i : i + eff_bs]
                    vecs, _ = (
                        await call(images=chunk)
                        if is_multimodal
                        else await call(sentences=chunk)
                    )
                    embeddings.extend(vecs)
                    pbar.update(len(chunk))
                pbar.close()

        elif self.backend == "sentence_transformers":
            if self.device == "cpu":
                device_param = "cpu"
                is_multi_gpu = False
            else:
                if self.device_num > 1:
                    device_param = [f"cuda:{i}" for i in range(self.device_num)]
                    is_multi_gpu = True
                else:
                    device_param = "cuda:0"
                    is_multi_gpu = False

            normalize = bool(self.st_encode_params.get("normalize_embeddings", False))
            csz = int(self.st_encode_params.get("encode_chunk_size", 256))
            psg_prompt_name = self.st_encode_params.get("psg_prompt_name", None)
            psg_task = self.st_encode_params.get("psg_task", None)

            if is_multimodal:
                data = []
                for p in self.contents:
                    with Image.open(p) as im:
                        data.append(im.convert("RGB").copy())
            else:
                data = self.contents

            if is_multi_gpu:
                app.logger.info(
                    f"[st] Starting multi-process pool on {len(device_param)} devices..."
                )
                pool = self.model.start_multi_process_pool()
                try:

                    def _encode_all():
                        return self.model.encode(
                            data,
                            pool=pool,
                            batch_size=self.batch_size,
                            chunk_size=csz,
                            show_progress_bar=True,
                            normalize_embeddings=normalize,
                            precision="float32",
                            prompt_name=psg_prompt_name,
                            task=psg_task,
                        )

                    embeddings = await asyncio.to_thread(_encode_all)
                finally:
                    self.model.stop_multi_process_pool(pool)
            else:

                def _encode_single():
                    return self.model.encode(
                        data,
                        device=device_param,
                        batch_size=self.batch_size,
                        show_progress_bar=True,
                        normalize_embeddings=normalize,
                        precision="float32",
                        prompt_name=psg_prompt_name,
                        task=psg_task,
                    )

                embeddings = await asyncio.to_thread(_encode_single)

        elif self.backend == "openai":
            if is_multimodal:
                err_msg = (
                    "openai backend does not support image embeddings in this path."
                )
                app.logger.error(err_msg)
                raise ValueError(err_msg)

            embeddings: list = []
            with tqdm(
                total=len(self.contents),
                desc="[openai] Embedding:",
                unit="item",
            ) as pbar:
                for start in range(0, len(self.contents), self.batch_size):
                    chunk = self.contents[start : start + self.batch_size]
                    resp = await self.model.embeddings.create(
                        model=self.model_name,
                        input=chunk,
                    )
                    embeddings.extend([d.embedding for d in resp.data])
                    pbar.update(len(chunk))
        else:
            err_msg = f"Unsupported backend: {self.backend}"
            app.logger.error(err_msg)
            raise ValueError(err_msg)

        if embeddings is None:
            raise RuntimeError("Embedding generation failed: embeddings is None")
        embeddings = np.array(embeddings, dtype=np.float32)
        np.save(embedding_path, embeddings)

        del embeddings
        gc.collect()
        app.logger.info("embedding success")

    async def retriever_index(
        self,
        embedding_path: str,
        overwrite: bool = False,
        collection_name: str = "",
        corpus_path: str = "",
    ) -> None:
        """Build index from embeddings or corpus (for demo mode).

        Args:
            embedding_path: Path to embeddings file (.npy) for non-demo mode
            overwrite: Whether to overwrite existing index
            collection_name: Collection name for Milvus backend
            corpus_path: Corpus file path (required for demo mode)

        Raises:
            ValidationError: If demo mode requirements are not met
            NotFoundError: If embedding file not found (non-demo mode)
            RuntimeError: If index backend is not initialized or indexing fails
            ValueError: If backend is BM25 or other unsupported backend
        """

        target_collection = collection_name

        if getattr(self, "is_demo", False):
            target_path = (
                corpus_path if corpus_path else getattr(self, "corpus_path", "")
            )

            if not target_path or not os.path.exists(target_path):
                msg = f"[Demo] corpus_path is required. Please provide it in call or init."
                if target_path:
                    msg += f" (File not found: {target_path})"
                raise ValidationError(msg)

            if not target_path.endswith(".jsonl"):
                raise ValidationError(
                    f"[Demo] Corpus file must be a JSONL file (.jsonl). Got: {target_path}"
                )

            app.logger.info(
                f"[Demo] Indexing JSONL: {target_path} -> Collection: {target_collection}"
            )

            milvus_cfg = self.index_backend_configs.get("milvus", {})

            configured_pk = milvus_cfg.get("id_field_name", "id")
            configured_vec = milvus_cfg.get("vector_field_name", "vector")
            configured_text = milvus_cfg.get("text_field_name", "contents")

            banned_keys = {
                # hard
                "contents",
                "content",
                "text",
                "embedding",
                "id",
                "_id",
                "pk",
                "uuid",
                # active
                configured_pk,
                configured_vec,
                configured_text,
            }

            texts = []
            metadatas = []
            file_name = os.path.basename(target_path)

            try:
                with open(target_path, "rb") as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = orjson.loads(line)
                        except orjson.JSONDecodeError:
                            app.logger.warning(
                                f"[Demo] Skipping invalid JSON on line {i}"
                            )
                            continue

                        raw_content = item.get("contents")
                        if raw_content is None:
                            continue

                        if not isinstance(raw_content, str):
                            content = str(raw_content)
                        else:
                            content = raw_content

                        if not content.strip():
                            continue
                        if content:
                            texts.append(content)
                            meta = {
                                k: v for k, v in item.items() if k not in banned_keys
                            }
                            meta["file_name"] = file_name
                            meta["segment_id"] = i
                            metadatas.append(meta)
            except Exception as e:
                raise ValidationError(f"[Demo] Failed to read JSONL file: {e}")

            if not texts:
                app.logger.warning("[Demo] No valid text chunks found in file.")
                return

            app.logger.info(f"[Demo] Embedding {len(texts)} chunks...")
            all_embeddings = []

            cached_dim = None

            for text in tqdm(texts, desc="[Demo] Processing"):
                try:
                    resp = await self.model.embeddings.create(
                        model=self.model_name, input=[text]
                    )

                    vec = resp.data[0].embedding
                    all_embeddings.append(vec)

                    if cached_dim is None:
                        cached_dim = len(vec)

                except Exception as e:
                    if cached_dim is not None:
                        app.logger.warning(
                            f"[Demo] Item failed. Filling with ZERO vector. Error: {str(e)[:100]}..."
                        )
                        all_embeddings.append([0.0] * cached_dim)
                    else:
                        app.logger.error(
                            f"[Demo] CRITICAL: First item failed! Cannot determine embedding dimension. Error: {e}"
                        )
                        raise e

            embeddings_np = np.array(all_embeddings, dtype=np.float32)

            ids = np.array([str(uuid.uuid4()) for _ in range(len(texts))])

            app.logger.info(
                f"[Demo] Inserting into collection '{target_collection}'..."
            )
            try:
                self.index_backend.build_index(
                    embeddings=embeddings_np,
                    ids=ids,
                    overwrite=overwrite,
                    collection_name=target_collection,
                    contents=texts,
                    metadatas=metadatas,
                )
            except Exception as e:
                app.logger.error(f"[Demo] Indexing failed: {e}")
                raise ToolError(f"Demo indexing failed: {e}")

            return
        else:
            if self.backend == "bm25":
                err_msg = "BM25 backend does not support vector index building via retriever_index."
                app.logger.error(err_msg)
                raise ValueError(err_msg)

            if self.index_backend is None:
                err_msg = (
                    "Vector index backend is not initialized. "
                    "Ensure retriever_init completed successfully."
                )
                app.logger.error(err_msg)
                raise RuntimeError(err_msg)

            if not os.path.exists(embedding_path):
                app.logger.error(f"Embedding file not found: {embedding_path}")
                raise NotFoundError(f"Embedding file not found: {embedding_path}")

            embedding = np.load(embedding_path)
            vec_ids = np.arange(embedding.shape[0]).astype(np.int64)

            try:
                self.index_backend.build_index(
                    embeddings=embedding,
                    ids=vec_ids,
                    overwrite=overwrite,
                )
            except ValueError as exc:
                raise ValidationError(str(exc)) from exc
            finally:
                del embedding
                gc.collect()

            info_msg = f"[{self.index_backend_name}] Indexing success."
            app.logger.info(info_msg)

    async def retriever_search(
        self,
        query_list: List[str],
        top_k: int = 5,
        query_instruction: str = "",
        collection_name: str = "",
    ) -> Dict[str, List[List[str]]]:
        """Search for passages using query embeddings.

        Args:
            query_list: List of query strings
            top_k: Number of top passages to return per query
            query_instruction: Optional instruction to prepend to queries
            collection_name: Collection name for Milvus backend

        Returns:
            Dictionary with 'ret_psg' containing retrieved passages

        Raises:
            RuntimeError: If index backend is not initialized
            ValueError: If backend is unsupported
        """

        if isinstance(query_list, str):
            query_list = [query_list]
        queries = [f"{query_instruction}{query}" for query in query_list]

        target_collection = collection_name
        if getattr(self, "is_demo", False):
            app.logger.info(
                f"[Demo] Searching query in collection: '{target_collection}'"
            )

        query_embedding = None

        if self.backend == "infinity":
            async with self.model:
                query_embedding, _ = await self.model.embed(sentences=queries)
        elif self.backend == "sentence_transformers":
            if self.device == "cpu":
                device_param = "cpu"
                is_multi_gpu = False
            else:
                if self.device_num > 1:
                    device_param = [f"cuda:{i}" for i in range(self.device_num)]
                    is_multi_gpu = True
                else:
                    device_param = "cuda:0"
                    is_multi_gpu = False

            normalize = bool(self.st_encode_params.get("normalize_embeddings", False))
            q_prompt_name = self.st_encode_params.get("q_prompt_name", "")
            q_task = self.st_encode_params.get("q_task", None)

            if is_multi_gpu:
                pool = self.model.start_multi_process_pool()
                try:

                    def _encode_all():
                        return self.model.encode(
                            queries,
                            pool=pool,
                            batch_size=self.batch_size,
                            show_progress_bar=True,
                            normalize_embeddings=normalize,
                            precision="float32",
                            prompt_name=q_prompt_name,
                            task=q_task,
                        )

                    query_embedding = await asyncio.to_thread(_encode_all)
                finally:
                    self.model.stop_multi_process_pool(pool)
            else:

                def _encode_single():
                    return self.model.encode(
                        queries,
                        device=device_param,
                        batch_size=self.batch_size,
                        show_progress_bar=True,
                        normalize_embeddings=normalize,
                        precision="float32",
                        prompt_name=q_prompt_name,
                        task=q_task,
                    )

                query_embedding = await asyncio.to_thread(_encode_single)

        elif self.backend == "openai":
            query_embedding = []
            for i in tqdm(
                range(0, len(queries), self.batch_size),
                desc="[openai] Embedding:",
                unit="batch",
            ):
                chunk = queries[i : i + self.batch_size]
                resp = await self.model.embeddings.create(
                    model=self.model_name, input=chunk
                )
                query_embedding.extend([d.embedding for d in resp.data])

        else:
            error_msg = f"Unsupported backend: {self.backend}"
            app.logger.error(error_msg)
            raise ValueError(error_msg)

        query_embedding = np.array(query_embedding, dtype=np.float32)

        if not getattr(self, "is_demo", False):
            info_msg = f"query embedding shape: {query_embedding.shape}"
            app.logger.info(info_msg)

        if self.index_backend is None:
            err_msg = (
                "Vector index backend is not initialized. "
                "Ensure retriever_init completed successfully."
            )
            app.logger.error(err_msg)
            raise RuntimeError(err_msg)

        rets = self.index_backend.search(
            query_embedding, top_k, collection_name=target_collection
        )

        return {"ret_psg": rets}

    async def retriever_batch_search(
        self,
        batch_query_list: List[List[str]],
        top_k: int = 5,
        query_instruction: str = "",
        collection_name: str = "",
    ) -> Dict[str, List[List[List[str]]]]:
        """Search for passages for multiple batches of queries.

        Args:
            batch_query_list: List of query lists (one batch per list)
            top_k: Number of top passages to return per query
            query_instruction: Optional instruction to prepend to queries
            collection_name: Collection name for Milvus backend

        Returns:
            Dictionary with 'ret_psg_ls' containing retrieved passages for each batch
        """

        ret_psg_ls = []
        for query_list in batch_query_list:
            if not query_list:
                ret_psg_ls.append([])
                continue

            result = await self.retriever_search(
                query_list=query_list,
                top_k=top_k,
                query_instruction=query_instruction,
                collection_name=collection_name,
            )

            ret_psg = result.get("ret_psg", [])
            ret_psg_ls.append(ret_psg)

        return {"ret_psg_ls": ret_psg_ls}

    async def retriever_deploy_search(
        self,
        retriever_url: str,
        query_list: List[str],
        top_k: int = 5,
        query_instruction: str = "",
    ) -> Dict[str, List[List[str]]]:
        """Search using remote retriever deployment.

        Args:
            retriever_url: URL of remote retriever service
            query_list: List of query strings
            top_k: Number of top passages to return per query
            query_instruction: Optional instruction to prepend to queries

        Returns:
            Dictionary with 'ret_psg' containing retrieved passages

        Raises:
            ToolError: If remote retriever call fails or response is invalid
        """
        from urllib.parse import urlparse, urlunparse

        url = retriever_url.strip()
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"http://{url}"

        url_obj = urlparse(url)
        api_url = urlunparse(url_obj._replace(path="/search", query="", fragment=""))

        app.logger.info(f"[remote_retriever] Calling remote retriever at: {api_url}")

        payload: Dict[str, Any] = {
            "query_list": query_list,
            "top_k": top_k,
            "query_instruction": query_instruction,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, json=payload) as response:
                if response.status != 200:
                    err_text = await response.text()
                    err_msg = (
                        f"[remote_retriever] Failed to call {api_url}, "
                        f"status={response.status}, body={err_text}"
                    )
                    app.logger.error(err_msg)
                    raise ToolError(err_msg)

                response_data = await response.json()
                app.logger.debug(
                    f"[remote_retriever] status={response.status}, keys={list(response_data.keys())}"
                )

                if "ret_psg" not in response_data:
                    err_msg = (
                        f"[remote_retriever] Response missing 'ret_psg' field: "
                        f"{response_data}"
                    )
                    app.logger.error(err_msg)
                    raise ToolError(err_msg)

                return {"ret_psg": response_data["ret_psg"]}

    async def bm25_index(
        self,
        overwrite: bool = False,
    ) -> None:
        """Build BM25 index from corpus.

        Args:
            overwrite: Whether to overwrite existing index
        """
        bm25_save_path = self.cfg.get("save_path", None)
        if bm25_save_path:
            output_dir = os.path.dirname(bm25_save_path)
        else:
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))
            output_dir = os.path.join(project_root, "output", "index")
            bm25_save_path = os.path.join(output_dir, "bm25")

        if not overwrite and os.path.exists(bm25_save_path):
            info_msg = (
                f"Index file already exists: {bm25_save_path}. "
                "Set overwrite=True to overwrite."
            )
            app.logger.info(info_msg)
            return

        if overwrite and os.path.exists(bm25_save_path):
            os.remove(bm25_save_path)

        corpus_tokens = self.tokenizer.tokenize(self.contents, return_as="tuple")
        self.model.index(corpus_tokens)
        self.model.save(bm25_save_path, corpus=None)
        self.tokenizer.save_stopwords(bm25_save_path)
        self.tokenizer.save_vocab(bm25_save_path)
        info_msg = "[bm25] Indexing success."
        app.logger.info(info_msg)

    async def bm25_search(
        self,
        query_list: List[str],
        top_k: int = 5,
    ) -> Dict[str, List[List[str]]]:
        """Search using BM25 index.

        Args:
            query_list: List of query strings
            top_k: Number of top passages to return per query

        Returns:
            Dictionary with 'ret_psg' containing retrieved passages
        """
        results = []
        q_toks = self.tokenizer.tokenize(
            query_list,
            return_as="tuple",
            update_vocab=False,
        )
        results, scores = self.model.retrieve(q_toks, k=top_k)
        results = results.tolist() if isinstance(results, np.ndarray) else results
        scores = scores.tolist() if isinstance(scores, np.ndarray) else scores
        return {"ret_psg": results}

    async def retriever_websearch(
        self,
        query_list: List[str],
        top_k: Optional[int] = 5,
        retrieve_thread_num: Optional[int] = 1,
        websearch_backend: str = "tavily",
        websearch_backend_configs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[List[str]]]:
        """Unified web search tool with multiple backends.

        Args:
            query_list: List of query strings
            top_k: Number of top results to return per query
            retrieve_thread_num: Maximum number of concurrent workers
            websearch_backend: Backend name ("tavily", "exa", "zhipuai")
            websearch_backend_configs: Backend configuration dictionary

        Returns:
            Dictionary with 'ret_psg' containing retrieved passages
        """
        if isinstance(query_list, str):
            query_list = [query_list]
        queries = [str(q) for q in (query_list or [])]
        if not queries:
            return {"ret_psg": []}

        backend_name = (websearch_backend or "tavily").lower()
        backend_cfgs = websearch_backend_configs or {}
        if not isinstance(backend_cfgs, dict):
            raise ValueError("websearch_backend_configs must be a dict")

        backend_cfg = backend_cfgs.get(backend_name, {})
        backend = create_websearch_backend(
            name=backend_name, logger=app.logger, config=backend_cfg
        )
        ret_psg = await backend.search(
            query_list=queries,
            top_k=top_k,
            retrieve_thread_num=retrieve_thread_num or 1,
        )
        return {"ret_psg": ret_psg}

    async def retriever_batch_websearch(
        self,
        batch_query_list: List[List[str]],
        top_k: Optional[int] = 5,
        retrieve_thread_num: Optional[int] = 1,
        websearch_backend: str = "tavily",
        websearch_backend_configs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[List[List[str]]]]:
        """Batch web search for SurveyCPM-style queries.

        Args:
            batch_query_list: List of query lists (one batch per list)
            top_k: Number of top results to return per query
            retrieve_thread_num: Maximum number of concurrent workers
            websearch_backend: Backend name ("tavily", "exa", "zhipuai")
            websearch_backend_configs: Backend configuration dictionary

        Returns:
            Dictionary with 'ret_psg_ls' containing retrieved passages for each batch
        """
        if not batch_query_list:
            return {"ret_psg_ls": []}

        backend_name = (websearch_backend or "tavily").lower()
        backend_cfgs = websearch_backend_configs or {}
        if not isinstance(backend_cfgs, dict):
            raise ValueError("websearch_backend_configs must be a dict")
        backend_cfg = backend_cfgs.get(backend_name, {})
        backend = create_websearch_backend(
            name=backend_name, logger=app.logger, config=backend_cfg
        )

        ret_psg_ls: List[List[List[str]]] = []
        for query_list in batch_query_list:
            if not query_list:
                ret_psg_ls.append([])
                continue
            if isinstance(query_list, str):
                query_list = [query_list]
            queries = [str(q) for q in query_list]
            ret_psg = await backend.search(
                query_list=queries,
                top_k=top_k,
                retrieve_thread_num=retrieve_thread_num or 1,
            )
            ret_psg_ls.append(ret_psg)

        return {"ret_psg_ls": ret_psg_ls}


if __name__ == "__main__":
    Retriever(app)
    app.run(transport="stdio")
