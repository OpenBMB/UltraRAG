"""Lifecycle helpers for concurrent requests owned by one pipeline step."""

import asyncio
from collections.abc import Iterable
from typing import Any


async def cancel_tasks(tasks: Iterable[asyncio.Task[Any]]) -> None:
    """Cancel and drain sibling requests before their owning step returns."""
    owned = list(tasks)
    for task in owned:
        if not task.done():
            task.cancel()
    await asyncio.gather(*owned, return_exceptions=True)
