import os
import sys
import argparse
from typing import List, Dict, Any, Optional

import orjson
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


RETRIEVER_SRC = os.path.join(PROJECT_ROOT, "servers", "retriever", "src")
if RETRIEVER_SRC not in sys.path:
    sys.path.insert(0, RETRIEVER_SRC)

from servers.retriever.src.retriever import Retriever, app  




def load_retriever_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise RuntimeError(f"Config file does not exist: {path}")

    with open(path, "rb") as f:
        cfg = orjson.loads(f.read())

    return cfg



class SearchRequest(BaseModel):
    query_list: List[str]
    top_k: int = 5
    query_instruction: str = ""
    collection_name: str = ""


class SearchResponse(BaseModel):
    ret_psg: List[List[str]]



fastapi_app = FastAPI(title="UltraRAG Retriever Service")

retriever: Optional[Retriever] = None   
retriever_cfg: Optional[Dict[str, Any]] = None


@fastapi_app.on_event("startup")
async def startup_event():

    global retriever, retriever_cfg

    assert retriever_cfg is not None, "retriever_cfg is not set"

    app.logger.info(f"[http retriever] Using configuration: {retriever_cfg}")

    retriever = Retriever(app)

    retriever.retriever_init(
        model_name_or_path=retriever_cfg["model_name_or_path"],
        backend_configs=retriever_cfg["backend_configs"],
        batch_size=retriever_cfg.get("batch_size", 32),
        corpus_path=retriever_cfg["corpus_path"],
        gpu_ids=retriever_cfg.get("gpu_ids", "0"),
        is_multimodal=retriever_cfg.get("is_multimodal", False),
        backend=retriever_cfg.get("backend", "sentence_transformers"),
        index_backend=retriever_cfg.get("index_backend", "faiss"),
        index_backend_configs=retriever_cfg.get("index_backend_configs", {}),
        is_demo=retriever_cfg.get("is_demo", False),
        collection_name=retriever_cfg.get("collection_name", ""),
    )

    app.logger.info("[http retriever] retriever_init completed (corpus & index loaded)")


@fastapi_app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):

    global retriever

    assert retriever is not None, "Retriever is not initialized"

    rets = await retriever.retriever_search(
        query_list=req.query_list,
        top_k=req.top_k,
        query_instruction=req.query_instruction,
        collection_name=req.collection_name,
    )

    return SearchResponse(ret_psg=rets["ret_psg"])


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Retriever HTTP Service")
    parser.add_argument(
        "--config_path",
        type=str,
        default='script/deploy_retriever_config.json',
        help="Path to retriever_config.json",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind the HTTP server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=64501,
        help="Port to bind the HTTP server",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    retriever_cfg = load_retriever_config(args.config_path)

    uvicorn.run(
        fastapi_app,
        host=args.host,
        port=args.port,
        reload=False,
    )