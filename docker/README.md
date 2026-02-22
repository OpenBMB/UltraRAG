# UltraRAG One-Click Compose

This folder provides two compose setups:

- `docker-compose.yml` (GPU): UI + Milvus + vLLM(LLM) + vLLM(Embedding)
- `docker-compose.cpu.yml` (CPU-only): UI + Milvus (no vLLM)

GPU setup includes:

- UltraRAG UI
- Milvus (standalone with etcd + minio)
- vLLM OpenAI-compatible LLM endpoint
- vLLM OpenAI-compatible Embedding endpoint

## 1) Prepare env

```bash
cd /Users/rqq/UltraRAG/docker
cp .env.example .env
```

Edit `.env` at least for these fields:

- `LLM_MODEL_PATH`
- `LLM_MODEL_NAME`
- `EMB_MODEL_PATH`
- `EMB_MODEL_NAME`
- Optional UI auto-fill defaults:
  - `ULTRARAG_UI_DEFAULT_AI_BASE_URL`
  - `ULTRARAG_UI_DEFAULT_AI_MODEL`
  - `ULTRARAG_UI_DEFAULT_EMB_BASE_URL`
  - `ULTRARAG_UI_DEFAULT_EMB_MODEL_NAME`

## 2) Start all services

```bash
docker compose up -d --build
```

Open UI:

- [http://localhost:5050](http://localhost:5050)

## 3) Recommended UI settings

Inside UltraRAG UI:

- Knowledge Base -> Milvus URI should be `tcp://milvus-standalone:19530`
  - Already pre-mounted from `docker/config/kb_config.json`.
- Embedding config:
  - `Base URL`: `http://vllm-emb:8000/v1`
  - `Model Name`: same as `EMB_MODEL_NAME` in `.env`
  - `API Key`: optional (for local vLLM you can use any non-empty value if required by UI)
- AI assistant / generation config:
  - `Base URL`: `http://vllm-llm:8000/v1`
  - `Model`: same as `LLM_MODEL_NAME` in `.env`
  - `API Key`: optional (for local vLLM you can use any non-empty value)

## 4) Stop

```bash
docker compose down
```

To remove volumes too:

```bash
docker compose down -v
```

## Notes

- `vllm-llm` and `vllm-emb` require NVIDIA GPU + working Docker GPU runtime.
- First startup may take a long time due to model/image downloads.
- Persistent data is under `docker/volumes/`.

---

## CPU-only setup (no vLLM)

If your machine has no NVIDIA GPU, use this flow.

### 1) Prepare env

```bash
cd /Users/rqq/UltraRAG/docker
cp .env.cpu.example .env
```

### 2) Start CPU stack

```bash
docker compose -f docker-compose.cpu.yml up -d --build
```

Open UI:

- [http://localhost:5050](http://localhost:5050)

### 3) UI settings for CPU mode

- Knowledge Base -> Milvus URI:
  - `tcp://milvus-standalone:19530`
  - Already pre-mounted from `docker/config/kb_config.json`.
- Embedding and Generation models:
  - Use external OpenAI-compatible API (OpenAI/Azure/other hosted endpoint).
  - You can prefill UI via `.env`:
    - `ULTRARAG_UI_DEFAULT_AI_*`
    - `ULTRARAG_UI_DEFAULT_EMB_*`
  - In UI you can still manually change provider `Base URL`, `Model`, `API Key`.
  - CPU compose does not start local vLLM endpoints.

### 4) Stop CPU stack

```bash
docker compose -f docker-compose.cpu.yml down
```
