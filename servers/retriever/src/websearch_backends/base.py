from __future__ import annotations

import abc
import asyncio
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm


class BaseWebSearchBackend(abc.ABC):
    """Abstract base class for web search backends."""

    def __init__(self, config: Optional[Dict[str, Any]], logger, **_: Any) -> None:
        self.config: Dict[str, Any] = dict(config or {})
        self.logger = logger

    async def _parallel_search(
        self,
        query_list: Sequence[str],
        retrieve_thread_num: int,
        desc: str,
        worker_factory: Callable[[int, str], Awaitable[Tuple[int, List[str]]]],
    ) -> List[List[str]]:
        if not query_list:
            return []

        concurrency = max(1, int(retrieve_thread_num or 1))
        sem = asyncio.Semaphore(concurrency)

        async def _wrap(i: int, q: str):
            async with sem:
                return await worker_factory(i, q)

        tasks = [asyncio.create_task(_wrap(i, q)) for i, q in enumerate(query_list)]
        ret: List[List[str]] = [[] for _ in range(len(query_list))]

        iterator = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc)
        for fut in iterator:
            idx, psg_ls = await fut
            ret[idx] = psg_ls
        return ret

    @abc.abstractmethod
    async def search(
        self,
        query_list: Sequence[str],
        top_k: Optional[int] = 5,
        retrieve_thread_num: Optional[int] = 1,
    ) -> List[List[str]]:
        """Search for passages for a list of queries."""
        ...

    def close(self) -> None:
        """Optional hook for releasing resources."""
        return None
