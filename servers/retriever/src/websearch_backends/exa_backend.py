from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional, Sequence

from fastmcp.exceptions import ToolError

from .base import BaseWebSearchBackend


class ExaWebSearchBackend(BaseWebSearchBackend):
    """Web search backend for Exa."""

    def __init__(self, config: Optional[Dict[str, Any]], logger, **kwargs: Any) -> None:
        super().__init__(config=config, logger=logger, **kwargs)
        try:
            from exa_py import AsyncExa
        except ImportError:
            err_msg = (
                "exa_py is not installed. Please install it with `pip install exa_py`."
            )
            self.logger.error(err_msg)
            raise ImportError(err_msg)

        api_key = self.config.get("api_key") or os.environ.get("EXA_API_KEY", "")
        self._client = AsyncExa(api_key=api_key if api_key else "EMPTY")

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
        desc = self.config.get("progress_desc", "EXA Searching:")
        effective_top_k = None if top_k is None else int(top_k)

        async def worker_factory(idx: int, q: str):
            delay = base_delay
            for attempt in range(retries):
                try:
                    params = {"text": True}
                    params.update(search_kwargs)
                    if effective_top_k is not None:
                        params["num_results"] = effective_top_k
                    resp = await self._client.search_and_contents(q, **params)
                    results = getattr(resp, "results", []) or []
                    psg_ls: List[str] = [(r.text or "") for r in results]
                    return idx, psg_ls
                except Exception as e:
                    status = getattr(getattr(e, "response", None), "status_code", None)
                    if status == 401 or "401" in str(e):
                        err_msg = (
                            "Unauthorized (401): Invalid or missing EXA_API_KEY. "
                            "Please set it to use Exa."
                        )
                        self.logger.error(err_msg)
                        raise ToolError(err_msg) from e
                    warn_msg = f"[exa][retry {attempt+1}] failed (idx={idx}): {e}"
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
