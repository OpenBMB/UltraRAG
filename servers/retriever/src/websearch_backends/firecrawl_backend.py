from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional, Sequence

import aiohttp
from fastmcp.exceptions import ToolError

from .base import BaseWebSearchBackend

_USER_AGENT = "UltraRAG"
_INTEGRATION = "_ultrarag"


def _passage(result: Dict[str, Any]) -> str:
    """Turn one Firecrawl web result into a passage: page markdown first, description as fallback."""
    return result.get("markdown") or result.get("description") or ""


def _retry_after(value: Optional[str], default: float) -> float:
    try:
        return max(0.0, float(value)) if value else default
    except ValueError:
        return default


class FirecrawlWebSearchBackend(BaseWebSearchBackend):
    """Web search backend for the Firecrawl Search API (keyed, FIRECRAWL_API_KEY)."""

    def __init__(self, config: Optional[Dict[str, Any]], logger, **kwargs: Any) -> None:
        super().__init__(config=config, logger=logger, **kwargs)
        api_key = self.config.get("api_key") or os.environ.get("FIRECRAWL_API_KEY", "")
        if not api_key:
            err_msg = (
                "FIRECRAWL_API_KEY environment variable is not set. "
                "Please set it to use Firecrawl."
            )
            self.logger.error(err_msg)
            raise ToolError(err_msg)

        self._api_key = api_key
        self._base_url = self.config.get(
            "base_url", "https://api.firecrawl.dev/v2/search"
        )

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
        effective_top_k = None if top_k is None else int(top_k)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": _USER_AGENT,
            "Content-Type": "application/json",
        }

        timeout = self.config.get("timeout", None)
        timeout_cfg = aiohttp.ClientTimeout(total=float(timeout)) if timeout else None

        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:

            async def worker_factory(idx: int, q: str):
                delay = base_delay
                for attempt in range(retries):
                    wait = delay
                    try:
                        payload: Dict[str, Any] = {
                            "query": q,
                            "integration": _INTEGRATION,
                            **search_kwargs,
                        }
                        if effective_top_k is not None:
                            payload["limit"] = effective_top_k

                        async with session.post(
                            self._base_url, json=payload, headers=headers
                        ) as resp:
                            if resp.status in (401, 403):
                                err_msg = f"Unauthorized ({resp.status}): invalid FIRECRAWL_API_KEY."
                                self.logger.error(err_msg)
                                raise ToolError(err_msg)
                            if resp.status == 402:
                                err_msg = "Firecrawl returned 402: the account is out of credits."
                                self.logger.error(err_msg)
                                raise ToolError(err_msg)
                            if resp.status == 429:
                                wait = _retry_after(
                                    resp.headers.get("Retry-After"), delay
                                )
                                raise RuntimeError("rate limited (429)")
                            resp.raise_for_status()
                            data = await resp.json()
                            web_results: List[Dict[str, Any]] = (
                                data.get("data") or {}
                            ).get("web") or []
                            psg_ls = [
                                p for p in (_passage(r) for r in web_results) if p
                            ]
                            return idx, psg_ls[:effective_top_k]
                    except ToolError:
                        raise
                    except (aiohttp.ClientError, Exception) as e:
                        warn_msg = (
                            f"[firecrawl][retry {attempt + 1}] failed (idx={idx}): {e}"
                        )
                        self.logger.warning(warn_msg)
                        await asyncio.sleep(wait)
                        delay *= 2
                return idx, []

            return await self._parallel_search(
                query_list=queries,
                retrieve_thread_num=retrieve_thread_num or 1,
                desc="Firecrawl Searching:",
                worker_factory=worker_factory,
            )
