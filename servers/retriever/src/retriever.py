import asyncio
import gc
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import orjson
import numpy as np
from tqdm import tqdm
from PIL import Image

from fastmcp.exceptions import ValidationError, NotFoundError, ToolError
from ultrarag.server import UltraRAG_MCP_Server
from index_backends import BaseIndexBackend, create_index_backend

app = UltraRAG_MCP_Server("retriever")


class Retriever:
    def __init__(self, mcp_inst: UltraRAG_MCP_Server):
        mcp_inst.tool(
            self.retriever_init,
            output="model_name_or_path,backend_configs,batch_size,corpus_path,gpu_ids,is_multimodal,backend,index_backend,index_backend_configs->None",
        )
        mcp_inst.tool(
            self.retriever_embed,
            output="embedding_path,overwrite,is_multimodal->None",
        )
        mcp_inst.tool(
            self.retriever_index,
            output="embedding_path,overwrite->None",
        )
        mcp_inst.tool(
            self.bm25_index,
            output="overwrite->None",
        )
        mcp_inst.tool(
            self.retriever_search,
            output="q_ls,top_k,query_instruction->ret_psg",
        )
        mcp_inst.tool(
            self.retriever_deploy_search,
            output="retriever_url,q_ls,top_k,query_instruction->ret_psg",
        )
        mcp_inst.tool(
            self.retriever_search_colbert_maxsim,
            output="q_ls,embedding_path,top_k,query_instruction->ret_psg",
        )
        mcp_inst.tool(
            self.bm25_search,
            output="q_ls,top_k->ret_psg",
        )
        mcp_inst.tool(
            self.retriever_exa_search,
            output="q_ls,top_k,retrieve_thread_num->ret_psg",
        )
        mcp_inst.tool(
            self.retriever_tavily_search,
            output="q_ls,top_k,retrieve_thread_num->ret_psg",
        )
        mcp_inst.tool(
            self.retriever_zhipuai_search,
            output="q_ls,top_k,retrieve_thread_num->ret_psg",
        )

    def _drop_keys(self, d: Dict[str, Any], banned: List[str]) -> Dict[str, Any]:
        return {k: v for k, v in (d or {}).items() if k not in banned and v is not None}

    def retriever_init(
        self,
        model_name_or_path: str,
        backend_configs: Dict[str, Any],
        batch_size: int,
        corpus_path: str,
        gpu_ids: Optional[object] = None,
        is_multimodal: bool = False,
        backend: str = "sentence_transformers",
        index_backend: str = "faiss",
        index_backend_configs: Optional[Dict[str, Any]] = None,
    ):

        self.backend = backend.lower()
        self.index_backend_name = index_backend.lower()
        self.index_backend_configs = index_backend_configs or {}
        self.index_backend: Optional[BaseIndexBackend] = None

        self.batch_size = batch_size
        self.backend_configs = backend_configs

        cfg = self.backend_configs.get(self.backend, {})
        self.cfg = cfg

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
                **cfg,
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
            self.st_encode_params = cfg.get("sentence_transformers_encode", {}) or {}
            st_params = self._drop_keys(cfg, banned=["sentence_transformers_encode"])

            self.model = SentenceTransformer(
                model_name_or_path=model_name_or_path,
                device=self.device,
                **st_params,
            )

        elif self.backend == "openai":
            try:
                from openai import AsyncOpenAI, OpenAIError
            except ImportError:
                err_msg = (
                    "openai is not installed. "
                    "Please install it with `pip install openai`."
                )
                app.logger.error(err_msg)
                raise ImportError(err_msg)

            model_name = cfg.get("model_name")
            base_url = cfg.get("base_url")
            api_key = cfg.get("api_key") or os.environ.get("RETRIEVER_API_KEY")

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
                info_msg = f"[openai] OpenAI client initialized (model='{model_name}', base='{base_url}')"
                app.logger.info(info_msg)
            except OpenAIError as e:
                err_msg = f"[openai] Failed to initialize OpenAI client: {e}"
                app.logger.error(err_msg)
                raise OpenAIError(err_msg)
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
            lang = cfg.get("lang", "en")
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
                            error_msg = (
                                f"Line {i}: missing key 'contents'. full item={item}"
                            )
                            app.logger.error(error_msg)
                            raise ValueError(error_msg)

                        self.contents.append(item["contents"])
                    else:
                        if "image_path" not in item:
                            error_msg = (
                                f"Line {i}: missing key 'image_path'. full item={item}"
                            )
                            app.logger.error(error_msg)
                            raise ValueError(error_msg)

                        rel = str(item["image_path"])
                        abs_path = str((corpus_dir / rel).resolve())
                        self.contents.append(abs_path)
                if bytes_read < file_size:
                    pbar.update(file_size - bytes_read)
                pbar.refresh() 

        if self.backend in ["infinity", "sentence_transformers", "openai"]:
            index_backend_cfg = self.index_backend_configs.get(
                self.index_backend_name, {}
            )
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
            bm25_save_path = cfg.get("save_path", None)
            if bm25_save_path and os.path.exists(bm25_save_path):
                self.model = self.model.load(bm25_save_path, mmap=True, load_corpus=False)
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
    ):
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
            if self.device_num == 1:
                device_param = "cuda:0"
            else:
                device_param = [f"cuda:{i}" for i in range(self.device_num)]
            normalize = bool(self.st_encode_params.get("normalize_embeddings", False))
            csz = int(self.st_encode_params.get("encode_chunk_size", 10000))
            psg_prompt_name = self.st_encode_params.get("psg_prompt_name", None)
            psg_task = self.st_encode_params.get("psg_task", None)

            if is_multimodal:
                data = []
                for p in self.contents:
                    with Image.open(p) as im:
                        data.append(im.convert("RGB").copy())
            else:
                data = self.contents

            if isinstance(device_param, list) and len(device_param) > 1:
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

    def retriever_index(
        self,
        embedding_path: str,
        overwrite: bool = False,
    ):
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

    def bm25_index(
        self,
        overwrite: bool = False,
    ):
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

    async def retriever_search(
        self,
        query_list: List[str],
        top_k: int = 5,
        query_instruction: str = "",
    ) -> Dict[str, List[List[str]]]:

        if isinstance(query_list, str):
            query_list = [query_list]
        queries = [f"{query_instruction}{query}" for query in query_list]

        if self.backend == "infinity":
            async with self.model:
                query_embedding, _ = await self.model.embed(sentences=queries)
        elif self.backend == "sentence_transformers":
            if self.device_num == 1:
                device_param = "cuda:0"
            else:
                device_param = [f"cuda:{i}" for i in range(self.device_num)]
            normalize = bool(self.st_encode_params.get("normalize_embeddings", False))
            q_prompt_name = self.st_encode_params.get("q_prompt_name", "")
            q_task = self.st_encode_params.get("psg_task", None)

            if isinstance(device_param, list) and len(device_param) > 1:
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

        info_msg = f"query embedding shape: {query_embedding.shape}"
        app.logger.info(info_msg)
        
        if self.index_backend is None:
            err_msg = (
                "Vector index backend is not initialized. "
                "Ensure retriever_init completed successfully."
            )
            app.logger.error(err_msg)
            raise RuntimeError(err_msg)

        rets = self.index_backend.search(query_embedding, top_k)

        return {"ret_psg": rets}
    
    async def retriever_deploy_search(
        self,
        retriever_url: str,
        query_list: List[str],
        top_k: int = 5,
        query_instruction: str = "",
    ) -> Dict[str, List[List[str]]]:
        from urllib.parse import urlparse, urlunparse
        import aiohttp

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

    async def retriever_search_colbert_maxsim(
        self,
        query_list: List[str],
        embedding_path: str,
        top_k: int = 5,
        query_instruction: str = "",
    ) -> Dict[str, List[List[str]]]:
        try:
            import torch
        except ImportError:
            err_msg = (
                "torch is not installed. Please install it with `pip install torch`."
            )
            app.logger.error(err_msg)
            raise ImportError(err_msg)

        if self.backend not in ["infinity"]:
            error_msg = (
                "retriever_search_colbert_maxsim only supports 'infinity' backend "
                "with ColBERT/ColPali multi-vector models. "
                "Use retriever_search or other backend-specific retrieval functions instead."
            )
            app.logger.error(error_msg)
            raise ValueError(error_msg)

        if isinstance(query_list, str):
            query_list = [query_list]
        queries = [f"{query_instruction}{query}" for query in query_list]

        async with self.model:
            query_embedding, _ = await self.model.embed(sentences=queries)

        doc_embeddings = np.load(embedding_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if (
            isinstance(doc_embeddings, np.ndarray)
            and doc_embeddings.dtype != object
            and doc_embeddings.ndim == 3
        ):
            docs_tensor = torch.from_numpy(
                doc_embeddings.astype("float32", copy=False)
            ).to(device)
        elif isinstance(doc_embeddings, np.ndarray) and doc_embeddings.dtype == object:
            try:
                stacked = np.stack(
                    [np.asarray(x, dtype=np.float32) for x in doc_embeddings.tolist()],
                    axis=0,
                )
                docs_tensor = torch.from_numpy(stacked).to(device)
            except Exception:
                error_msg = (
                    f"Document embeddings in {embedding_path} have inconsistent shapes, "
                    "cannot stack into (N,Kd,D). "
                    f"Check your retriever_embed."
                )
                app.logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            error_msg = (
                f"Unexpected doc_embeddings format: type={type(doc_embeddings)}, "
                f"shape={getattr(doc_embeddings, 'shape', None)}"
            )
            app.logger.error(error_msg)
            raise ValueError(error_msg)

        def _l2norm(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
            return t / t.norm(dim=-1, keepdim=True).clamp_min(eps)

        N, _, D_docs = docs_tensor.shape
        docs_tensor = _l2norm(docs_tensor)
        k_pick = min(top_k, N)

        results = []
        for q_np in query_embedding:
            q = torch.as_tensor(
                q_np,
                dtype=torch.float32,
                device=device,
            )
            if q.shape[-1] != D_docs:
                error_msg = (
                    f"Dimension mismatch: query D={q.shape[-1]} vs doc D={D_docs}"
                )
                app.logger.error(error_msg)
                raise ValueError(error_msg)

            q = _l2norm(q)
            sim = torch.einsum("qd,nkd->nqk", q, docs_tensor)
            sim_max = sim.max(dim=2).values
            scores = sim_max.sum(dim=1)

            top_idx = torch.topk(scores, k=k_pick, largest=True).indices.tolist()
            results.append([self.contents[i] for i in top_idx])
        return {"ret_psg": results}

    async def bm25_search(
        self,
        query_list: List[str],
        top_k: int = 5,
    ) -> Dict[str, List[List[str]]]:
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

    async def _parallel_search(
        self,
        query_list: List[str],
        retrieve_thread_num: int,
        desc: str,
        worker_factory,
    ) -> Dict[str, List[List[str]]]:
        sem = asyncio.Semaphore(retrieve_thread_num)

        async def _wrap(i: int, q: str):
            async with sem:
                return await worker_factory(i, q)

        tasks = [asyncio.create_task(_wrap(i, q)) for i, q in enumerate(query_list)]
        ret: List[List[str]] = [None] * len(query_list)

        iterator = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc)
        for fut in iterator:
            idx, psg_ls = await fut
            ret[idx] = psg_ls
        return {"ret_psg": ret}

    async def retriever_exa_search(
        self,
        query_list: List[str],
        top_k: Optional[int] | None = 5,
        retrieve_thread_num: Optional[int] | None = 1,
    ) -> Dict[str, List[List[str]]]:

        try:
            from exa_py import AsyncExa
            from exa_py.api import Result
        except ImportError:
            err_msg = (
                "exa_py is not installed. Please install it with `pip install exa_py`."
            )
            app.logger.error(err_msg)
            raise ImportError(err_msg)

        exa_api_key = os.environ.get("EXA_API_KEY", "")
        exa = AsyncExa(api_key=exa_api_key if exa_api_key else "EMPTY")

        async def worker_factory(idx: int, q: str):
            retries, delay = 3, 1.0
            for attempt in range(retries):
                try:
                    resp = await exa.search_and_contents(
                        q, num_results=top_k, text=True
                    )
                    results: List[Result] = getattr(resp, "results", []) or []
                    psg_ls: List[str] = [(r.text or "") for r in results]
                    return idx, psg_ls
                except Exception as e:
                    status = getattr(getattr(e, "response", None), "status_code", None)
                    if status == 401 or "401" in str(e):
                        err_msg = (
                            "Unauthorized (401): Invalid or missing EXA_API_KEY. "
                            "Please set it to use Exa."
                        )
                        app.logger.error(err_msg)
                        raise ToolError(err_msg) from e
                    warn_msg = f"[exa][retry {attempt+1}] failed (idx={idx}): {e}"
                    app.logger.warning(warn_msg)
                    await asyncio.sleep(delay)
            return idx, []

        return await self._parallel_search(
            query_list=query_list,
            retrieve_thread_num=retrieve_thread_num or 1,
            desc="EXA Searching:",
            worker_factory=worker_factory,
        )

    async def retriever_tavily_search(
        self,
        query_list: List[str],
        top_k: Optional[int] | None = 5,
        retrieve_thread_num: Optional[int] | None = 1,
    ) -> Dict[str, List[List[str]]]:

        try:
            from tavily import (
                AsyncTavilyClient,
                BadRequestError,
                UsageLimitExceededError,
                InvalidAPIKeyError,
                MissingAPIKeyError,
            )
        except ImportError:
            err_msg = "tavily is not installed. Please install it with `pip install tavily-python`."
            app.logger.error(err_msg)
            raise ImportError(err_msg)

        tavily_api_key = os.environ.get("TAVILY_API_KEY", "")
        if not tavily_api_key:
            err_msg = (
                "TAVILY_API_KEY environment variable is not set. "
                "Please set it to use Tavily."
            )
            app.logger.error(err_msg)
            raise MissingAPIKeyError(err_msg)
        tavily = AsyncTavilyClient(api_key=tavily_api_key)

        async def worker_factory(idx: int, q: str):
            retries, delay = 3, 1.0
            for attempt in range(retries):
                try:
                    resp = await tavily.search(query=q, max_results=top_k)
                    results: List[Dict[str, Any]] = resp["results"]
                    psg_ls: List[str] = [(r.get("content") or "") for r in results]
                    return idx, psg_ls
                except UsageLimitExceededError as e:
                    err_msg = f"Usage limit exceeded: {e}"
                    app.logger.error(err_msg)
                    raise ToolError(err_msg) from e
                except InvalidAPIKeyError as e:
                    err_msg = f"Invalid API key: {e}"
                    app.logger.error(err_msg)
                    raise ToolError(err_msg) from e
                except (BadRequestError, Exception) as e:
                    warn_msg = f"[tavily][retry {attempt+1}] failed (idx={idx}): {e}"
                    app.logger.warning(warn_msg)
                    await asyncio.sleep(delay)
            return idx, []

        return await self._parallel_search(
            query_list=query_list,
            retrieve_thread_num=retrieve_thread_num or 1,
            desc="Tavily Searching:",
            worker_factory=worker_factory,
        )

    async def retriever_zhipuai_search(
        self,
        query_list: List[str],
        top_k: Optional[int] | None = 5,
        retrieve_thread_num: Optional[int] | None = 1,
    ) -> Dict[str, List[List[str]]]:

        zhipuai_api_key = os.environ.get("ZHIPUAI_API_KEY", "")
        if not zhipuai_api_key:
            err_msg = (
                "ZHIPUAI_API_KEY environment variable is not set. "
                "Please set it to use ZhipuAI."
            )
            app.logger.error(err_msg)
            raise ToolError(err_msg)

        retrieval_url = "https://open.bigmodel.cn/api/paas/v4/web_search"
        headers = {
            "Authorization": f"Bearer {zhipuai_api_key}",
            "Content-Type": "application/json",
        }

        session = aiohttp.ClientSession()

        async def worker_factory(idx: int, q: str):
            retries, delay = 3, 1.0
            for attempt in range(retries):
                try:
                    payload = {
                        "search_query": q,
                        "search_engine": "search_std",  # [search_std, search_pro, search_pro_sogou, search_pro_quark]
                        "search_intent": False,
                        "count": top_k,  # [10,20,30,40,50]
                        "search_recency_filter": "noLimit",  # [oneDay, oneWeek, oneMonth, oneYear, noLimit]
                        "content_size": "medium",  # [medium, high]
                    }
                    async with session.post(
                        retrieval_url, json=payload, headers=headers
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        results: List[Dict[str, Any]] = data.get("search_result", [])
                        psg_ls: List[str] = [(r.get("content") or "") for r in results]
                        # Respect top_k
                        return idx, (psg_ls[:top_k] if top_k is not None else psg_ls)
                except (aiohttp.ClientError, Exception) as e:
                    warn_msg = f"[zhipuai][retry {attempt+1}] failed (idx={idx}): {e}"
                    app.logger.warning(warn_msg)
                    await asyncio.sleep(delay)
            return idx, []

        try:
            return await self._parallel_search(
                query_list=query_list,
                retrieve_thread_num=retrieve_thread_num or 1,
                desc="ZhipuAI Searching:",
                worker_factory=worker_factory,
            )
        finally:
            await session.close()


if __name__ == "__main__":
    Retriever(app)
    app.run(transport="stdio")
