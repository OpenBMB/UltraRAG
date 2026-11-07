"""Opt-in, small live API parity check for local execution and stdio MCP.

Credentials come only from ULTRARAG_AUDIT_API_KEY, never command arguments or
saved parameter files. Run against two checkouts and compare the JSON outputs.
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


async def exercise(
    root: Path, mode: str, model: str, embedding_model: str, backend: str = "openai"
) -> dict[str, Any]:
    import numpy as np

    from ultrarag.client import create_execution_client, local_server_launch_config

    server_cfg = {
        name: {"path": str(root / "servers" / name / "src" / f"{name}.py")}
        for name in ("generation", "retriever")
    }
    mcp_cfg = {
        "mcpServers": {
            name: local_server_launch_config(cfg["path"])
            for name, cfg in server_cfg.items()
        }
    }
    endpoint = {
        "base_url": os.environ["ULTRARAG_AUDIT_API_BASE"],
        "api_key": os.environ["ULTRARAG_AUDIT_API_KEY"],
    }
    routed_model = f"openai/{model}" if backend == "litellm" else model
    routed_embedding = (
        f"openai/{embedding_model}" if backend == "litellm" else embedding_model
    )
    passages = [
        "Paris is the capital of France.",
        "Beijing is the capital of China.",
        "The Pacific is the largest ocean.",
    ]
    with tempfile.TemporaryDirectory(prefix="ultrarag-api-") as temporary:
        directory = Path(temporary)
        corpus = directory / "corpus.jsonl"
        corpus.write_text(
            "".join(json.dumps({"contents": p}) + "\n" for p in passages),
            encoding="utf-8",
        )
        embedding_path = directory / "embedding.npy"
        async with create_execution_client(
            mcp_cfg, server_cfg, no_mcp=mode == "local"
        ) as adapter:
            await adapter.call_tool(
                "retriever_retriever_init",
                {
                    "model_name_or_path": routed_embedding,
                    "backend_configs": {
                        backend: {
                            **endpoint,
                            "model_name": routed_embedding,
                            "concurrency": 2,
                        }
                    },
                    "batch_size": 2,
                    "corpus_path": str(corpus),
                    "backend": backend,
                    "index_backend": "faiss",
                    "gpu_ids": None,
                    "index_backend_configs": {
                        "faiss": {
                            "index_use_gpu": False,
                            "index_path": str(directory / "index.index"),
                        }
                    },
                },
            )
            await adapter.call_tool(
                "retriever_retriever_embed",
                {"embedding_path": str(embedding_path), "overwrite": True},
            )
            vectors = np.load(embedding_path)
            assert vectors.shape[0] == len(passages) and np.isfinite(vectors).all()
            await adapter.call_tool(
                "retriever_retriever_index",
                {"embedding_path": str(embedding_path), "overwrite": True},
            )
            result = await adapter.call_tool(
                "retriever_retriever_search",
                {
                    "query_list": [
                        "What is the capital of France?",
                        "What is the capital of China?",
                    ],
                    "top_k": 2,
                },
            )
            retrieval = result.data["ret_psg"]
            assert retrieval[0][0] == passages[0], retrieval
            assert retrieval[1][0] == passages[1], retrieval
            await adapter.call_tool(
                "generation_generation_init",
                {
                    "backend_configs": {
                        backend: {
                            **endpoint,
                            "model_name": routed_model,
                            "concurrency": 2,
                            "retries": 1,
                        }
                    },
                    "sampling_params": {"temperature": 0, "max_tokens": 128},
                    "extra_params": {
                        "chat_template_kwargs": {"enable_thinking": False}
                    },
                    "backend": backend,
                },
            )
            answers = await adapter.call_tool(
                "generation_generate",
                {
                    "prompt_ls": [
                        f"Context: {docs[0]}\nReturn the capital city name only."
                        for docs in retrieval
                    ],
                },
            )
            text = answers.data["ans_ls"]
            assert "Paris" in text[0] and "Beijing" in text[1], text
        return {
            "mode": mode,
            "model": model,
            "embedding_model": embedding_model,
            "embedding_shape": list(vectors.shape),
            "embedding_sha256": hashlib.sha256(vectors.tobytes()).hexdigest(),
            "retrieval": retrieval,
            "answers": text,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--backend", choices=("openai", "litellm"), default="openai")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("ULTRARAG_AUDIT_API_KEY", "ULTRARAG_AUDIT_API_BASE"):
        if not os.environ.get(name):
            parser.error(f"Set {name} in the environment")
    root = args.repository.resolve()
    sys.path.insert(0, str(root / "src"))
    os.environ["PYTHONPATH"] = os.pathsep.join([str(root / "src"), str(root)])
    os.environ["LITELLM_TELEMETRY"] = "False"
    # Import native FAISS before entering the event loop, as in server startup.
    import faiss  # noqa: F401

    async def run() -> list[dict[str, Any]]:
        return [
            await exercise(root, mode, args.model, args.embedding_model, args.backend)
            for mode in ("mcp", "local")
        ]

    args.output.write_text(
        json.dumps(asyncio.run(run()), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved live API parity results to {args.output}")


if __name__ == "__main__":
    main()
