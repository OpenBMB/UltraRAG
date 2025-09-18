import os
from urllib.parse import urlparse, urlunparse
from typing import Any, Dict, List, Optional

import requests
import aiohttp
import asyncio
import jsonlines
import numpy as np
import pandas as pd
from tqdm import tqdm
from flask import Flask, jsonify, request
from openai import AsyncOpenAI, OpenAIError


from fastmcp.exceptions import NotFoundError, ToolError, ValidationError
from ultrarag.server import UltraRAG_MCP_Server
from pathlib import Path

app = UltraRAG_MCP_Server("retriever")
retriever_app = Flask(__name__)


class Retriever:
    def __init__(self, mcp_inst: UltraRAG_MCP_Server):
        mcp_inst.tool(
            self.retriever_init,
            output="retriever_path,corpus_path,index_path,faiss_use_gpu,infinity_kwargs,cuda_devices,is_multimodal->None",
        )
        mcp_inst.tool(
            self.retriever_init_openai,
            output="corpus_path,openai_model,api_base,api_key,faiss_use_gpu,index_path,cuda_devices->None",
        )
        mcp_inst.tool(
            self.retriever_embed,
            output="embedding_path,overwrite,is_multimodal->None",
        )
        mcp_inst.tool(
            self.retriever_embed_openai,
            output="embedding_path,overwrite->None",
        )
        mcp_inst.tool(
            self.retriever_index,
            output="embedding_path,index_path,overwrite,index_chunk_size->None",
        )
        mcp_inst.tool(
            self.retriever_index_lancedb,
            output="embedding_path,lancedb_path,table_name,overwrite->None",
        )
        mcp_inst.tool(
            self.retriever_search,
            output="q_ls,top_k,query_instruction,use_openai->ret_psg",
        )
        mcp_inst.tool(
            self.retriever_search_maxsim,
            output="q_ls,embedding_path,top_k,query_instruction->ret_psg",
        )
        mcp_inst.tool(
            self.retriever_search_lancedb,
            output="q_ls,top_k,query_instruction,use_openai,lancedb_path,table_name,filter_expr->ret_psg",
        )
        mcp_inst.tool(
            self.retriever_deploy_service,
            output="retriever_url->None",
        )
        mcp_inst.tool(
            self.retriever_deploy_search,
            output="retriever_url,q_ls,top_k,query_instruction->ret_psg",
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

    def retriever_init(
        self,
        retriever_path: str,
        corpus_path: str,
        index_path: Optional[str] = None,
        faiss_use_gpu: bool = False,
        infinity_kwargs: Optional[Dict[str, Any]] = None,
        cuda_devices: Optional[str] = None,
        is_multimodal: bool = False,
    ):

        try:
            import faiss
        except ImportError:
            err_msg = "faiss is not installed. Please install it with `conda install -c pytorch faiss-cpu` or `conda install -c pytorch faiss-gpu`."
            app.logger.error(err_msg)
            raise ImportError(err_msg)

        try:
            from infinity_emb.log_handler import LOG_LEVELS
            from infinity_emb import AsyncEngineArray, EngineArgs
        except ImportError:
            err_msg = "infinity_emb is not installed. Please install it with `pip install infinity-emb`."
            app.logger.error(err_msg)
            raise ImportError(err_msg)

        self.faiss_use_gpu = faiss_use_gpu
        app.logger.setLevel(LOG_LEVELS["warning"])

        if cuda_devices is not None:
            assert isinstance(cuda_devices, str), "cuda_devices should be a string"
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

        infinity_kwargs = infinity_kwargs or {}
        self.model = AsyncEngineArray.from_args(
            [EngineArgs(model_name_or_path=retriever_path, **infinity_kwargs)]
        )[0]

        self.contents = []
        corpus_path_obj = Path(corpus_path)
        corpus_dir = corpus_path_obj.parent
        if not is_multimodal:
            with jsonlines.open(corpus_path, mode="r") as reader:
                self.contents = [item["contents"] for item in reader]
        else:
            with jsonlines.open(corpus_path, mode="r") as reader:
                for i, item in enumerate(reader):
                    if "image_path" not in item:
                        raise ValueError(
                            f"Line {i}: expected key 'image_path' in multimodal corpus JSONL, got keys={list(item.keys())}"
                        )
                    rel = str(item["image_path"])
                    abs_path = str((corpus_dir / rel).resolve())
                    self.contents.append(abs_path)

        self.faiss_index = None
        if index_path is not None and os.path.exists(index_path):
            cpu_index = faiss.read_index(index_path)

            if self.faiss_use_gpu:
                co = faiss.GpuMultipleClonerOptions()
                co.shard = True
                co.useFloat16 = True
                try:
                    self.faiss_index = faiss.index_cpu_to_all_gpus(cpu_index, co)
                    app.logger.info(f"Loaded index to GPU(s).")
                except RuntimeError as e:
                    app.logger.error(
                        f"GPU index load failed: {e}. Falling back to CPU."
                    )
                    self.faiss_use_gpu = False
                    self.faiss_index = cpu_index
            else:
                self.faiss_index = cpu_index
                app.logger.info("Loaded index on CPU.")

            app.logger.info(f"Retriever index path has already been built")
        else:
            app.logger.warning(f"Cannot find path: {index_path}")
            self.faiss_index = None
            app.logger.info(f"Retriever initialized")

    def retriever_init_openai(
        self,
        corpus_path: str,
        openai_model: str,
        api_base: str,
        api_key: str = None,
        faiss_use_gpu: bool = False,
        index_path: Optional[str] = None,
        cuda_devices: Optional[str] = None,
    ):
        try:
            import faiss
        except ImportError:
            err_msg = "faiss is not installed. Please install it with `conda install -c pytorch faiss-cpu` or `conda install -c pytorch faiss-gpu`."
            app.logger.error(err_msg)
            raise ImportError(err_msg)

        if not openai_model:
            raise ValueError("openai_model must be provided.")
        if not api_base or not isinstance(api_base, str):
            raise ValueError("api_base must be a non-empty string.")

        api_key = os.environ.get("RETRIEVER_API_KEY") or api_key

        if not api_key or not isinstance(api_key, str):
            raise ValueError("api_key must be a non-empty string.")

        self.faiss_use_gpu = faiss_use_gpu
        if cuda_devices is not None:
            assert isinstance(cuda_devices, str), "cuda_devices should be a string"
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

        self.faiss_index = None
        if index_path is not None and os.path.exists(index_path):
            cpu_index = faiss.read_index(index_path)

            if self.faiss_use_gpu:
                co = faiss.GpuMultipleClonerOptions()
                co.shard = True
                co.useFloat16 = True
                try:
                    self.faiss_index = faiss.index_cpu_to_all_gpus(cpu_index, co)
                    app.logger.info(f"Loaded index to GPU(s).")
                except RuntimeError as e:
                    app.logger.error(
                        f"GPU index load failed: {e}. Falling back to CPU."
                    )
                    self.faiss_use_gpu = False
                    self.faiss_index = cpu_index
            else:
                self.faiss_index = cpu_index
                app.logger.info("Loaded index on CPU.")

            app.logger.info(f"Retriever index path has already been built")
        else:
            app.logger.warning(f"Cannot find path: {index_path}")
            self.faiss_index = None
            app.logger.info(f"Retriever initialized")

        self.contents = []
        with jsonlines.open(corpus_path, mode="r") as reader:
            self.contents = [item["contents"] for item in reader]

        try:
            self.openai_model = openai_model
            self.client = AsyncOpenAI(base_url=api_base, api_key=api_key)
            app.logger.info(
                f"OpenAI client initialized with model '{openai_model}' and base '{api_base}'"
            )
        except OpenAIError as e:
            app.logger.error(f"Failed to initialize OpenAI client: {e}")

    async def retriever_embed(
        self,
        embedding_path: Optional[str] = None,
        overwrite: bool = False,
        is_multimodal: bool = False,
    ):

        if embedding_path is not None:
            if not embedding_path.endswith(".npy"):
                err_msg = f"Embedding save path must end with .npy, now the path is {embedding_path}"
                app.logger.error(err_msg)
                raise ValidationError(err_msg)
            output_dir = os.path.dirname(embedding_path)
        else:
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))
            output_dir = os.path.join(project_root, "output", "embedding")
            embedding_path = os.path.join(output_dir, "embedding.npy")

        if not overwrite and os.path.exists(embedding_path):
            app.logger.info("embedding already exists, skipping")
            return

        os.makedirs(output_dir, exist_ok=True)

        async with self.model:
            if is_multimodal:
                from PIL import Image

                images = []
                for i, p in enumerate(self.contents):
                    try:
                        with Image.open(p) as im:
                            images.append(im.convert("RGB").copy())
                    except Exception as e:
                        err_msg = f"Failed to load image at index {i}: {p} ({e})"
                        app.logger.error(err_msg)
                        raise RuntimeError(err_msg)
                embeddings, usage = await self.model.image_embed(images=images)
            else:
                embeddings, usage = await self.model.embed(sentences=self.contents)

        embeddings = np.array(embeddings, dtype=np.float16)
        np.save(embedding_path, embeddings)
        app.logger.info("embedding success")

    async def retriever_embed_openai(
        self,
        embedding_path: Optional[str] = None,
        overwrite: bool = False,
    ):
        if embedding_path is not None:
            if not embedding_path.endswith(".npy"):
                err_msg = f"Embedding save path must end with .npy, now the path is {embedding_path}"
                app.logger.error(err_msg)
                raise ValidationError(err_msg)
            output_dir = os.path.dirname(embedding_path)
        else:
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))
            output_dir = os.path.join(project_root, "output", "embedding")
            embedding_path = os.path.join(output_dir, "embedding.npy")

        if not overwrite and os.path.exists(embedding_path):
            app.logger.info("embedding already exists, skipping")

        os.makedirs(output_dir, exist_ok=True)

        async def openai_embed(texts):
            embeddings = []
            for text in texts:
                response = await self.client.embeddings.create(
                    input=text, model=self.openai_model
                )
                embeddings.append(response.data[0].embedding)
            return embeddings

        embeddings = await openai_embed(self.contents)

        embeddings = np.array(embeddings, dtype=np.float16)
        np.save(embedding_path, embeddings)
        app.logger.info("embedding success")

    def retriever_index(
        self,
        embedding_path: str,
        index_path: Optional[str] = None,
        overwrite: bool = False,
        index_chunk_size: int = 50000,
    ):
        """
        Build a Faiss index from an embedding matrix.

        Args:
            embedding_path (str): .npy file of shape (N, dim), dtype float32.
            index_path (str, optional): where to save .index file.
            overwrite (bool): overwrite existing index.
            index_chunk_size (int): batch size for add_with_ids.
        """
        try:
            import faiss
        except ImportError:
            err_msg = "faiss is not installed. Please install it with `conda install -c pytorch faiss-cpu` or `conda install -c pytorch faiss-gpu`."
            app.logger.error(err_msg)
            raise ImportError(err_msg)

        if not os.path.exists(embedding_path):
            app.logger.error(f"Embedding file not found: {embedding_path}")
            NotFoundError(f"Embedding file not found: {embedding_path}")

        if index_path is not None:
            if not index_path.endswith(".index"):
                app.logger.error(
                    f"Parameter index_path must end with .index now is {index_path}"
                )
                ValidationError(
                    f"Parameter index_path must end with .index now is {index_path}"
                )
            output_dir = os.path.dirname(index_path)
        else:
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))
            output_dir = os.path.join(project_root, "output", "index")
            index_path = os.path.join(output_dir, "index.index")

        if not overwrite and os.path.exists(index_path):
            app.logger.info("Index already exists, skipping")
            return

        os.makedirs(output_dir, exist_ok=True)

        embedding = np.load(embedding_path)
        dim = embedding.shape[1]
        vec_ids = np.arange(embedding.shape[0]).astype(np.int64)

        # with cpu
        cpu_flat = faiss.IndexFlatIP(dim)
        cpu_index = faiss.IndexIDMap2(cpu_flat)

        # chunk to write
        total = embedding.shape[0]
        for start in range(0, total, index_chunk_size):
            end = min(start + index_chunk_size, total)
            cpu_index.add_with_ids(embedding[start:end], vec_ids[start:end])

        # with gpu
        if self.faiss_use_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = True
            try:
                gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co)
                index = gpu_index
                app.logger.info("Using GPU for indexing with sharding")
            except RuntimeError as e:
                app.logger.warning(f"GPU indexing failed ({e}); fall back to CPU")
                self.faiss_use_gpu = False
                index = cpu_index
        else:
            index = cpu_index

        # save
        faiss.write_index(cpu_index, index_path)

        if self.faiss_index is None:
            self.faiss_index = index

        app.logger.info("Indexing success")

    def retriever_index_lancedb(
        self,
        embedding_path: str,
        lancedb_path: str,
        table_name: str,
        overwrite: bool = False,
    ):
        """
        Build a Faiss index from an embedding matrix.

        Args:
            embedding_path (str): .npy file of shape (N, dim), dtype float32.
            lancedb_path (str): directory path to store LanceDB tables.
            table_name (str): the name of the LanceDB table.
            overwrite (bool): overwrite existing index.
        """
        try:
            import lancedb
        except ImportError:
            err_msg = "lancedb is not installed. Please install it with `pip install lancedb`."
            app.logger.error(err_msg)
            raise ImportError(err_msg)

        if not os.path.exists(embedding_path):
            app.logger.error(f"Embedding file not found: {embedding_path}")
            NotFoundError(f"Embedding file not found: {embedding_path}")

        if lancedb_path is None:
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))
            lancedb_path = os.path.join(project_root, "output", "lancedb")

        os.makedirs(lancedb_path, exist_ok=True)
        db = lancedb.connect(lancedb_path)

        if table_name in db.table_names() and not overwrite:
            info_msg = f"LanceDB table '{table_name}' already exists, skipping"
            app.logger.info(info_msg)
            return {"status": info_msg}
        elif table_name in db.table_names() and overwrite:
            import shutil

            shutil.rmtree(os.path.join(lancedb_path, table_name))
            app.logger.info(f"Overwriting LanceDB table '{table_name}'")

        embedding = np.load(embedding_path)
        ids = [str(i) for i in range(len(embedding))]
        data = [{"id": i, "vector": v} for i, v in zip(ids, embedding)]
        df = pd.DataFrame(data)
        db.create_table(table_name, data=df)

        app.logger.info("LanceDB indexing success")

    async def retriever_search(
        self,
        query_list: List[str],
        top_k: int = 5,
        query_instruction: str = "",
        use_openai: bool = False,
    ) -> Dict[str, List[List[str]]]:

        if isinstance(query_list, str):
            query_list = [query_list]
        queries = [f"{query_instruction}{query}" for query in query_list]

        if use_openai:

            async def openai_embed(texts):
                embeddings = []
                for text in texts:
                    response = await self.client.embeddings.create(
                        input=text, model=self.openai_model
                    )
                    embeddings.append(response.data[0].embedding)
                return embeddings

            query_embedding = await openai_embed(queries)
        else:
            async with self.model:
                query_embedding, usage = await self.model.embed(sentences=queries)
        query_embedding = np.array(query_embedding, dtype=np.float16)
        app.logger.info("query embedding finish")

        scores, ids = self.faiss_index.search(query_embedding, top_k)
        rets = []
        for i, query in enumerate(query_list):
            cur_ret = []
            for _, id in enumerate(ids[i]):
                cur_ret.append(self.contents[id])
            rets.append(cur_ret)
        app.logger.debug(f"ret_psg: {rets}")
        return {"ret_psg": rets}

    async def retriever_search_maxsim(
        self,
        query_list: List[str],
        embedding_path: str,
        top_k: int = 5,
        query_instruction: str = "",
    ) -> Dict[str, List[List[str]]]:
        import torch

        if isinstance(query_list, str):
            query_list = [query_list]
        queries = [f"{query_instruction}{query}" for query in query_list]

        async with self.model:
            query_embedding, usage = await self.model.embed(
                sentences=queries
            )  # (Q, Kq, D)

        doc_embeddings = np.load(embedding_path)  # (N, Kd, D)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if (
            isinstance(doc_embeddings, np.ndarray)
            and doc_embeddings.dtype != object
            and doc_embeddings.ndim == 3
        ):
            # (N,Kd,D)
            docs_tensor = torch.from_numpy(
                doc_embeddings.astype("float32", copy=False)
            ).to(device)
        elif isinstance(doc_embeddings, np.ndarray) and doc_embeddings.dtype == object:
            try:
                stacked = np.stack(
                    [np.asarray(x, dtype=np.float32) for x in doc_embeddings.tolist()],
                    axis=0,
                )  # (N,Kd,D)
                docs_tensor = torch.from_numpy(stacked).to(device)
            except Exception:
                raise ValueError(
                    f"Document embeddings in {embedding_path} have inconsistent shapes, cannot stack into (N,Kd,D). "
                    f"Check your retriever_embed."
                )
        else:
            raise ValueError(
                f"Unexpected doc_embeddings format: type={type(doc_embeddings)}, shape={getattr(doc_embeddings, 'shape', None)}"
            )

        results = []

        def _l2norm(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
            return t / t.norm(dim=-1, keepdim=True).clamp_min(eps)

        N, Kd, D_docs = docs_tensor.shape

        docs_tensor = _l2norm(docs_tensor)  # (N,Kd,D)

        results = []
        k_pick = min(top_k, N)

        for q_np in query_embedding:

            q = torch.as_tensor(q_np, dtype=torch.float32, device=device)  # (Kq,D)

            D_query = q.shape[-1]
            if D_query != D_docs:
                raise ValueError(f"Dim mismatch: query D={D_query} vs doc D={D_docs}")

            q = _l2norm(q)  # (Kq,D)

            # MaxSim
            # sim[n, i, j] = dot(q[i], docs_tensor[n, j])
            # (Kq,D) x (N,Kd,D) -> (N,Kq,Kd)
            sim = torch.einsum("qd,nkd->nqk", q, docs_tensor)
            # doc tokens -> (N,Kq)
            sim_max = sim.max(dim=2).values
            # query tokens-> (N,)
            scores = sim_max.sum(dim=1)

            top_idx = torch.topk(scores, k=k_pick, largest=True).indices.tolist()
            top_contents = [self.contents[i] for i in top_idx]
            results.append(top_contents)

        return {"ret_psg": results}

    async def retriever_search_lancedb(
        self,
        query_list: List[str],
        top_k: Optional[int] | None = None,
        query_instruction: str = "",
        use_openai: bool = False,
        lancedb_path: str = "",
        table_name: str = "",
        filter_expr: Optional[str] = None,
    ) -> Dict[str, List[List[str]]]:

        try:
            import lancedb
        except ImportError:
            err_msg = "lancedb is not installed. Please install it with `pip install lancedb`."
            app.logger.error(err_msg)
            raise ImportError(err_msg)

        if isinstance(query_list, str):
            query_list = [query_list]
        queries = [f"{query_instruction}{query}" for query in query_list]

        if use_openai:

            async def openai_embed(texts):
                embeddings = []
                for text in texts:
                    response = await self.client.embeddings.create(
                        input=text, model=self.openai_model
                    )
                    embeddings.append(response.data[0].embedding)
                return embeddings

            query_embedding = await openai_embed(queries)
        else:
            async with self.model:
                query_embedding, usage = await self.model.embed(sentences=queries)
        query_embedding = np.array(query_embedding, dtype=np.float16)
        app.logger.info("query embedding finish")

        rets = []

        if not lancedb_path:
            NotFoundError(f"`lancedb_path` must be provided.")
        db = lancedb.connect(lancedb_path)
        self.lancedb_table = db.open_table(table_name)
        for i, query_vec in enumerate(query_embedding):
            q = self.lancedb_table.search(query_vec).limit(top_k)
            if filter_expr:
                q = q.where(filter_expr)
            df = q.to_df()
            cur_ret = []
            for id_str in df["id"]:
                id_int = int(id_str)
                cur_ret.append(self.contents[id_int])
            rets.append(cur_ret)

        app.logger.debug(f"ret_psg: {rets}")
        return {"ret_psg": rets}

    async def retriever_deploy_service(
        self,
        retriever_url: str,
    ):
        # Ensure URL is valid, adding "http://" prefix if necessary
        retriever_url = retriever_url.strip()
        if not retriever_url.startswith("http://") and not retriever_url.startswith(
            "https://"
        ):
            retriever_url = f"http://{retriever_url}"

        url_obj = urlparse(retriever_url)
        retriever_host = url_obj.hostname
        retriever_port = (
            url_obj.port if url_obj.port else 8080
        )  # Default port if none provided

        @retriever_app.route("/search", methods=["POST"])
        async def deploy_retrieval_model():
            data = request.get_json()
            query_list = data["query_list"]
            top_k = data["top_k"]
            async with self.model:
                query_embedding, _ = await self.model.embed(sentences=query_list)
            query_embedding = np.array(query_embedding, dtype=np.float16)
            _, ids = self.faiss_index.search(query_embedding, top_k)

            rets = []
            for i, _ in enumerate(query_list):
                cur_ret = []
                for _, id in enumerate(ids[i]):
                    cur_ret.append(self.contents[id])
                rets.append(cur_ret)
            return jsonify({"ret_psg": rets})

        retriever_app.run(host=retriever_host, port=retriever_port)
        app.logger.info(f"employ embedding server at {retriever_url}")

    async def retriever_deploy_search(
        self,
        retriever_url: str,
        query_list: List[str],
        top_k: Optional[int] | None = None,
        query_instruction: str = "",
    ):
        # Validate the URL format
        url = retriever_url.strip()
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"http://{url}"
        url_obj = urlparse(url)
        api_url = urlunparse(url_obj._replace(path="/search"))
        app.logger.info(f"Calling url: {api_url}")

        if isinstance(query_list, str):
            query_list = [query_list]
        query_list = [f"{query_instruction}{query}" for query in query_list]

        payload = {"query_list": query_list}
        if top_k is not None:
            payload["top_k"] = top_k

        async with aiohttp.ClientSession() as session:
            async with session.post(
                api_url,
                json=payload,
            ) as response:
                if response.status == 200:
                    response_data = await response.json()
                    app.logger.debug(
                        f"status_code: {response.status}, response data: {response_data}"
                    )
                    return response_data
                else:
                    err_msg = (
                        f"Failed to call {retriever_url} with code {response.status}"
                    )
                    app.logger.error(err_msg)
                    raise ToolError(err_msg)

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
                        raise ToolError(
                            "Unauthorized (401): Invalid or missing EXA_API_KEY."
                        ) from e
                    app.logger.warning(
                        f"[Retry {attempt+1}] EXA failed (idx={idx}): {e}"
                    )
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
            raise MissingAPIKeyError(
                "TAVILY_API_KEY environment variable is not set. Please set it to use Tavily."
            )
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
                    app.logger.error(f"Usage limit exceeded: {e}")
                    raise ToolError(f"Usage limit exceeded: {e}") from e
                except InvalidAPIKeyError as e:
                    app.logger.error(f"Invalid API key: {e}")
                    raise ToolError(f"Invalid API key: {e}") from e
                except (BadRequestError, Exception) as e:
                    app.logger.warning(
                        f"[Retry {attempt+1}] Tavily failed (idx={idx}): {e}"
                    )
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
            raise ToolError(
                "ZHIPUAI_API_KEY environment variable is not set. Please set it to use ZhipuAI."
            )

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
                    app.logger.warning(
                        f"[Retry {attempt+1}] ZhipuAI failed (idx={idx}): {e}"
                    )
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
