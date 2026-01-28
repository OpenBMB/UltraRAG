from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional, Sequence

import aiohttp
from fastmcp.exceptions import ToolError

from .base import BaseWebSearchBackend


class ZhipuaiWebSearchBackend(BaseWebSearchBackend):
    """Web search backend for ZhipuAI."""

    def __init__(self, config: Optional[Dict[str, Any]], logger, **kwargs: Any) -> None:
        super().__init__(config=config, logger=logger, **kwargs)
        api_key = self.config.get("api_key") or os.environ.get("ZHIPUAI_API_KEY", "")
        if not api_key:
            err_msg = (
                "ZHIPUAI_API_KEY environment variable is not set. "
                "Please set it to use ZhipuAI."
            )
            self.logger.error(err_msg)
            raise ToolError(err_msg)

        self._api_key = api_key
        self._base_url = self.config.get(
            "base_url", "https://open.bigmodel.cn/api/paas/v4/web_search"
        )
        self._search_engine = self.config.get("search_engine", "search_std")
        self._search_intent = self.config.get("search_intent", False)
        self._search_recency_filter = self.config.get(
            "search_recency_filter", "noLimit"
        )
        self._content_size = self.config.get("content_size", "medium")

    async def search(
        self,
        query_list: Sequence[str],
        top_k: Optional[int] = 5,
        retrieve_thread_num: Optional[int] = 1,
    ) -> List[List[str]]:
        queries = [str(q) for q in query_list]
        if not queries:
            return []

        retries = int(self.config.get("retries", 3))
        base_delay = float(self.config.get("base_delay", 1.0))
        search_kwargs = dict(self.config.get("search_kwargs", {}) or {})
        result_key = self.config.get("result_key", "search_result")
        content_key = self.config.get("content_key", "content")
        desc = self.config.get("progress_desc", "ZhipuAI Searching:")
        effective_top_k = None if top_k is None else int(top_k)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        timeout = self.config.get("timeout", None)
        timeout_cfg = (
            aiohttp.ClientTimeout(total=float(timeout)) if timeout else None
        )

        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:

            async def worker_factory(idx: int, q: str):
                delay = base_delay
                for attempt in range(retries):
                    try:
                        payload = {
                            "search_query": q,
                            "search_engine": self._search_engine,
                            "search_intent": self._search_intent,
                            "search_recency_filter": self._search_recency_filter,
                            "content_size": self._content_size,
                        }
                        if effective_top_k is not None:
                            payload["count"] = effective_top_k
                        else:
                            count = self.config.get("count", None)
                            if count is not None:
                                payload["count"] = count

                        if search_kwargs:
                            payload.update(search_kwargs)

                        async with session.post(
                            self._base_url, json=payload, headers=headers
                        ) as resp:
                            resp.raise_for_status()
                            data = await resp.json()
                            results: List[Dict[str, Any]] = (
                                data.get(result_key, []) or []
                            )
                            psg_ls: List[str] = [
                                (r.get(content_key) or "") for r in results
                            ]
                            if effective_top_k is not None:
                                psg_ls = psg_ls[:effective_top_k]
                            return idx, psg_ls
                    except (aiohttp.ClientError, Exception) as e:
                        warn_msg = (
                            f"[zhipuai][retry {attempt+1}] failed (idx={idx}): {e}"
                        )
                        self.logger.warning(warn_msg)
                        await asyncio.sleep(delay)
                        delay *= 2
                return idx, []

            return await self._parallel_search(
                query_list=queries,
                retrieve_thread_num=retrieve_thread_num or 1,
                desc=desc,
                worker_factory=worker_factory,
            )
