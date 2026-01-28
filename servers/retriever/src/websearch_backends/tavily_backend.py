from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional, Sequence

from fastmcp.exceptions import ToolError

from .base import BaseWebSearchBackend


class TavilyWebSearchBackend(BaseWebSearchBackend):
    """Web search backend for Tavily."""

    def __init__(self, config: Optional[Dict[str, Any]], logger, **kwargs: Any) -> None:
        super().__init__(config=config, logger=logger, **kwargs)
        try:
            from tavily import (
                AsyncTavilyClient,
                BadRequestError,
                UsageLimitExceededError,
                InvalidAPIKeyError,
                MissingAPIKeyError,
            )
        except ImportError:
            err_msg = (
                "tavily is not installed. Please install it with `pip install tavily-python`."
            )
            self.logger.error(err_msg)
            raise ImportError(err_msg)

        self._BadRequestError = BadRequestError
        self._UsageLimitExceededError = UsageLimitExceededError
        self._InvalidAPIKeyError = InvalidAPIKeyError
        self._MissingAPIKeyError = MissingAPIKeyError

        api_key = self.config.get("api_key") or os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            err_msg = (
                "TAVILY_API_KEY environment variable is not set. "
                "Please set it to use Tavily."
            )
            self.logger.error(err_msg)
            raise MissingAPIKeyError(err_msg)
        self._client = AsyncTavilyClient(api_key=api_key)

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
        desc = self.config.get("progress_desc", "Tavily Searching:")
        effective_top_k = None if top_k is None else int(top_k)

        async def worker_factory(idx: int, q: str):
            delay = base_delay
            for attempt in range(retries):
                try:
                    params = dict(search_kwargs)
                    if effective_top_k is not None and "max_results" not in params:
                        params["max_results"] = effective_top_k
                    resp = await self._client.search(query=q, **params)
                    results: List[Dict[str, Any]] = resp.get("results", []) or []
                    psg_ls: List[str] = [(r.get("content") or "") for r in results]
                    return idx, psg_ls
                except self._UsageLimitExceededError as e:
                    err_msg = f"Usage limit exceeded: {e}"
                    self.logger.error(err_msg)
                    raise ToolError(err_msg) from e
                except self._InvalidAPIKeyError as e:
                    err_msg = f"Invalid API key: {e}"
                    self.logger.error(err_msg)
                    raise ToolError(err_msg) from e
                except (self._BadRequestError, Exception) as e:
                    warn_msg = f"[tavily][retry {attempt+1}] failed (idx={idx}): {e}"
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
