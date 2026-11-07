"""A failed embedding batch must not leave sibling API requests running."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from ultrarag.execution import LocalExecutionClient


def test_failed_batch_cancels_and_drains_other_requests() -> None:
    source = Path(__file__).resolve().parents[3] / "servers/retriever/src/retriever.py"
    adapter = LocalExecutionClient(
        {"mcpServers": {}}, {"retriever": {"path": str(source)}}
    )

    async def exercise() -> None:
        async with adapter:
            provider = adapter.local["retriever"].tools["retriever_init"].fn.__self__
            provider.backend = "openai"
            provider.model_name = "test"
            started = asyncio.Event()
            finished = asyncio.Event()

            async def create(**kwargs):
                if kwargs["input"] == ["fail"]:
                    await started.wait()
                    raise ValueError("embedding failure")
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    finished.set()

            provider.model = SimpleNamespace(embeddings=SimpleNamespace(create=create))
            with pytest.raises(ValueError, match="embedding failure"):
                await provider._api_embed_texts(
                    ["fail", "pending"], batch_size=1, concurrency=2, desc="test"
                )
            assert finished.is_set()

    asyncio.run(exercise())
