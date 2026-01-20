import asyncio
import os
from typing import Any, Dict, List, Optional, Union

import aiohttp
from tqdm import tqdm

from ultrarag.server import UltraRAG_MCP_Server

app = UltraRAG_MCP_Server("reranker")


class Reranker:
    def __init__(self, mcp_inst: UltraRAG_MCP_Server):
        mcp_inst.tool(
            self.reranker_init,
            output="model_name_or_path,backend_configs,batch_size,gpu_ids,backend->None",
        )
        mcp_inst.tool(
            self.reranker_rerank,
            output="q_ls,ret_psg,top_k,query_instruction->rerank_psg",
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

    async def reranker_init(
        self,
        model_name_or_path: str,
        backend_configs: Dict[str, Any],
        batch_size: int,
        gpu_ids: Optional[Union[str, int]] = None,
        backend: str = "infinity",
    ) -> None:
        """Initialize reranker backend (infinity, sentence_transformers, or openai).

        Args:
            model_name_or_path: Model name or path
            backend_configs: Dictionary of backend-specific configurations
            batch_size: Batch size for reranking
            gpu_ids: Comma-separated GPU IDs (e.g., "0,1")
            backend: Backend name ("infinity", "sentence_transformers", or "openai")

        Raises:
            ImportError: If required dependencies are not installed
            ValueError: If required config is missing
        """
        self.backend = backend.lower()
        self.batch_size = batch_size
        self.backend_configs = backend_configs

        cfg = self.backend_configs.get(self.backend, {})

        gpu_ids_str = str(gpu_ids) if gpu_ids is not None else ""
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str

        self.device_num = len(gpu_ids_str.split(",")) if gpu_ids_str else 1

        if self.backend == "infinity":
            try:
                from infinity_emb import AsyncEngineArray, EngineArgs
            except ImportError:
                err_msg = "infinity_emb is not installed. Please install it with `pip install infinity-emb`."
                app.logger.error(err_msg)
                raise ImportError(err_msg)

            device = str(cfg.get("device", "")).strip().lower()
            if not device:
                warn_msg = f"[infinity] device is not set, default to `cpu`"
                app.logger.warning(warn_msg)
                device = "cpu"

            if device == "cpu":
                info_msg = "[infinity] device=cpu, gpu_ids is ignored"
                app.logger.info(info_msg)
                self.device_num = 1

            app.logger.info(
                f"[infinity] device={device}, gpu_ids={gpu_ids_str}, device_num={self.device_num}"
            )

            infinity_engine_args = EngineArgs(
                model_name_or_path=model_name_or_path,
                batch_size=self.batch_size,
                **cfg,
            )
            self.model = AsyncEngineArray.from_args([infinity_engine_args])[0]

        elif self.backend == "sentence_transformers":
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                err_msg = (
                    "sentence_transformers is not installed. "
                    "Please install it with `pip install sentence-transformers`."
                )
                app.logger.error(err_msg)
                raise ImportError(err_msg)
            self.st_encode_params = cfg.get("sentence_transformers_encode", {}) or {}
            st_params = self._drop_keys(cfg, banned=["sentence_transformers_encode"])

            device = str(cfg.get("device", "")).strip().lower()
            if not device:
                warn_msg = (
                    f"[sentence_transformers] device is not set, default to `cpu`"
                )
                app.logger.warning(warn_msg)
                device = "cpu"

            if device == "cpu":
                info_msg = "[sentence_transformers] device=cpu, gpu_ids is ignored"
                app.logger.info(info_msg)
                self.device_num = 1

            app.logger.info(
                f"[sentence_transformers] device={device}, gpu_ids={gpu_ids_str}, device_num={self.device_num}"
            )

            self.model = CrossEncoder(
                model_name_or_path=model_name_or_path,
                **st_params,
            )

        elif self.backend == "openai":
            model_name = cfg.get("model_name")
            base_url = cfg.get("base_url")
            concurrency = cfg.get("concurrency", 1)

            if not model_name:
                err_msg = "[openai] model_name is required"
                app.logger.error(err_msg)
                raise ValueError(err_msg)
            if not isinstance(base_url, str) or not base_url:
                err_msg = "[openai] base_url must be a non-empty string"
                app.logger.error(err_msg)
                raise ValueError(err_msg)

            self.rerank_url = base_url
            self.model_name = model_name
            self.concurrency = concurrency

    async def reranker_rerank(
        self,
        query_list: List[str],
        passages_list: List[List[str]],
        top_k: int = 5,
        query_instruction: str = "",
    ) -> Dict[str, List[Any]]:
        """Rerank passages for queries using the initialized reranker backend.

        Args:
            query_list: List of query strings
            passages_list: List of passage lists (one per query)
            top_k: Number of top passages to return per query
            query_instruction: Optional instruction to prepend to queries

        Returns:
            Dictionary with 'rerank_psg' containing reranked passages

        Raises:
            ValueError: If query_list and passages_list have different lengths
            RuntimeError: If reranking fails (for openai backend)
        """
        if len(query_list) != len(passages_list):
            err_msg = (
                f"[reranker] query_list and passages_list must have same length, "
                f"but got {len(query_list)} and {len(passages_list)}"
            )
            app.logger.error(err_msg)
            raise ValueError(err_msg)

        formatted_queries = [f"{query_instruction}{q}" for q in query_list]

        if self.backend == "infinity":

            async def rerank_single(query: str, docs: List[str]) -> List[str]:
                ranking, _ = await self.model.rerank(
                    query=query,
                    docs=docs,
                    top_n=top_k,
                )
                return [d.document for d in ranking]

            async with self.model:
                reranked_results = await asyncio.gather(
                    *[
                        rerank_single(query, docs)
                        for query, docs in zip(formatted_queries, passages_list)
                    ]
                )

        elif self.backend == "sentence_transformers":

            def _rank_all(query: str, docs: List[str]) -> List[str]:
                ranks = self.model.rank(
                    query,
                    docs,
                    top_k=top_k,
                    batch_size=self.batch_size,
                    return_documents=True,
                    show_progress_bar=False,
                )
                return [rank["text"] for rank in ranks]

            reranked_results = []
            for query, docs in tqdm(
                zip(formatted_queries, passages_list),
                total=len(formatted_queries),
                desc="Reranking",
            ):
                reranked_results.append(await asyncio.to_thread(_rank_all, query, docs))

        elif self.backend == "openai":
            semaphore = asyncio.Semaphore(self.concurrency)

            async def rerank_single_oa(
                session: aiohttp.ClientSession,
                query: str,
                docs: List[str],
            ) -> List[str]:
                payload = {
                    "model": self.model_name,
                    "query": query,
                    "documents": docs,
                    "top_n": top_k,
                }
                async with semaphore:
                    async with session.post(self.rerank_url, json=payload) as resp:
                        if resp.status != 200:
                            err_msg = f"[{resp.status}] {await resp.text()}"
                            raise RuntimeError(err_msg)

                        data = await resp.json()
                        results = data.get("results", [])
                        ret = []
                        for item in results:
                            doc_val = item.get("document")
                            if isinstance(doc_val, dict):
                                doc_val = doc_val.get("text", "")
                            if isinstance(doc_val, str):
                                ret.append(doc_val)
                        return ret

            async with aiohttp.ClientSession() as session:
                reranked_results = await asyncio.gather(
                    *[
                        rerank_single_oa(session, query, docs)
                        for query, docs in zip(formatted_queries, passages_list)
                    ]
                )

        return {"rerank_psg": reranked_results}


provider = Reranker(app)

if __name__ == "__main__":
    app.run(transport="stdio")
