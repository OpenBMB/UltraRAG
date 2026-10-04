from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional, Sequence

import aiohttp
from fastmcp.exceptions import ToolError

from .base import BaseWebSearchBackend


class SerplyWebSearchBackend(BaseWebSearchBackend):
    """Web search backend for Serply (Google search results)."""

    def __init__(self, config: Optional[Dict[str, Any]], logger, **kwargs: Any) -> None:
        super().__init__(config=config, logger=logger, **kwargs)
        api_key = self.config.get("api_key") or os.environ.get("SERPLY_API_KEY", "")
        if not api_key:
            err_msg = (
                "SERPLY_API_KEY environment variable is not set. "
                "Please set it to use Serply."
            )
            self.logger.error(err_msg)
            raise ToolError(err_msg)

        self._api_key = api_key
        self._base_url = self.config.get("base_url", "https://api.serply.io/v1/search")

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
        content_key = self.config.get("content_key", "description")
        effective_top_k = None if top_k is None else int(top_k)

        headers = {"X-Api-Key": self._api_key, "User-Agent": "UltraRAG"}

        timeout = self.config.get("timeout", None)
        timeout_cfg = (
            aiohttp.ClientTimeout(total=float(timeout)) if timeout else None
        )

        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:

            async def worker_factory(idx: int, q: str):
                delay = base_delay
                for attempt in range(retries):
                    try:
                        params: Dict[str, Any] = {"q": q, **search_kwargs}
                        if effective_top_k is not None:
                            params["num"] = effective_top_k

                        async with session.get(
                            self._base_url, params=params, headers=headers
                        ) as resp:
                            if resp.status in (401, 403):
                                err_msg = f"Unauthorized ({resp.status}): invalid SERPLY_API_KEY."
                                self.logger.error(err_msg)
                                raise ToolError(err_msg)
                            resp.raise_for_status()
                            data = await resp.json()
                            results: List[Dict[str, Any]] = data.get("results", []) or []
                            psg_ls: List[str] = [(r.get(content_key) or "") for r in results]
                            return idx, psg_ls[:effective_top_k]
                    except ToolError:
                        raise
                    except (aiohttp.ClientError, Exception) as e:
                        warn_msg = f"[serply][retry {attempt+1}] failed (idx={idx}): {e}"
                        self.logger.warning(warn_msg)
                        await asyncio.sleep(delay)
                        delay *= 2
                return idx, []

            return await self._parallel_search(
                query_list=queries,
                retrieve_thread_num=retrieve_thread_num or 1,
                desc="Serply Searching:",
                worker_factory=worker_factory,
            )
