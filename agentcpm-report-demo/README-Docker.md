## UltraRAG (v3-dev) Docker 一键运行

本工作区已包含 `UltraRAG/` 源码（分支：`v3-dev`），并在根目录提供了 `docker-compose.yml` 用于 **构建镜像** 与 **一键启动**。

参考 UltraRAG 官方仓库：[`OpenBMB/UltraRAG/tree/v3-dev`](https://github.com/OpenBMB/UltraRAG/tree/v3-dev)

### 0) 前置要求

- **Docker** + **Docker Compose** 可用
- （可选 GPU）安装 NVIDIA 驱动 + `nvidia-container-toolkit`

说明：如果你的环境支持 `docker compose` 插件，也可以把下文的 `docker-compose` 替换为 `docker compose`。

### 0.1)（可选但强烈推荐）配置 Docker Hub 镜像加速 / Registry Mirror

如果你遇到类似报错：

- `Get "https://registry-1.docker.io/v2/": context deadline exceeded`

说明当前机器无法直连 Docker Hub。解决方式之一是在 Docker daemon 上配置 **registry mirror**（镜像加速/代理仓库）。

在 **CentOS 7 / RHEL 系**上通常这样配置（需要 root）：

1) 创建/编辑 `/etc/docker/daemon.json`：

```json
{
  "registry-mirrors": [
    "https://<your-mirror-host>"
  ]
}
```

其中 `https://<your-mirror-host>` 替换成你们内网镜像地址，或可用的公共镜像加速地址。

2) 重启 Docker：

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

3) 验证是否生效：

```bash
docker info | grep -A2 -i "Registry Mirrors"
```

> 备注：如果你们环境还需要 HTTP/HTTPS 代理（proxy），也可以给 Docker 配置 systemd drop-in 的 `HTTP_PROXY/HTTPS_PROXY/NO_PROXY`；具体值按你们网络策略来。

### 1) 构建镜像

在本目录执行：

```bash
cd /home/yyk/yyk03/Workspace/AgentCPM-Report
docker-compose build
```

镜像会使用我们新增的 **`UltraRAG/Dockerfile.v3-dev`** 构建（Python 3.11 + `uv` 直接安装当前 `v3-dev` 源码），不依赖任何“官方旧镜像”。

#### （可选）Debian APT 源太慢：切换到 Aliyun / TUNA

如果你构建时 `apt-get update/apt-get install` 很慢，可以在本目录创建/编辑 `.env`，加入其中一个：

- `APT_MIRROR=aliyun`
- `APT_MIRROR=tuna`

然后重新构建：

```bash
docker-compose build --no-cache ultrarag
docker-compose build --no-cache ultrarag-ui
```

> 如果你是 CPU-only 环境（没有 CUDA），并且构建时在下载巨大的 CUDA 版 torch wheel，可以在 `.env` 里加：
>
> - `TORCH_INDEX_URL_BUILD=https://download.pytorch.org/whl/cpu`
>
> 如果你 **有 GPU** 并且希望安装 CUDA 版 torch wheel（示例：CUDA 12.1），可以设置：
>
> - `TORCH_INDEX_URL_BUILD=https://download.pytorch.org/whl/cu121`

#### GPU 默认配置（Faiss + Torch）

本仓库现在默认：

- retriever `gpu_ids="0"`
- FAISS `index_use_gpu=True`
- `PIP_EXTRA_PACKAGES` 默认安装 `faiss-gpu-cu12`

并且 `ultrarag` / `ultrarag-ui` / `ultrarag-ui-admin` 已加 `runtime: nvidia`（要求宿主机安装 `nvidia-container-toolkit`）。

（可选）如果你需要用到远程 MCP server（`npx mcp-remote`），构建时打开 Node.js 20：

```bash
ULTRARAG_INSTALL_NODE=1 docker-compose build
```

（可选）如果你要一键装可选依赖（注意：某些 GPU/FAISS 依赖在纯 CPU 环境可能会装不上）：

```bash
ULTRARAG_EXTRAS=all docker-compose build
```

### 2) 跑一个最小可用验证（sayhello）

```bash
docker-compose run --rm ultrarag
```

预期看到类似输出：`Hello, UltraRAG 2.0!`

### 3) 启动 Web UI

- **普通 UI（chat-only）**：

```bash
docker-compose up -d ultrarag-ui
```

访问：`http://localhost:5050`（或用 `ULTRARAG_UI_PORT` 改端口）

- **管理员 UI（带 pipeline builder）**：

```bash
docker-compose up -d ultrarag-ui-admin
```

访问：`http://localhost:5051`（或用 `ULTRARAG_ADMIN_UI_PORT` 改端口）

#### Embedding（避免 OpenAI 超时 / 避免拉取 Docker Hub 的 Infinity 镜像）

在一些内网/受限网络环境里：

- 可能无法访问 `https://api.openai.com`（导致 `retriever_index` 超时）
- 也可能无法访问 Docker Hub（导致 `michaelfeil/infinity:latest` 拉取失败）

因此本仓库默认改为 **in-process embedding**（不需要额外 `infinity-embed` 容器）：

- `RETRIEVER_BACKEND=infinity`（依赖 `pip install infinity-emb`，已加入默认 `PIP_EXTRA_PACKAGES`）

如果你是 **CPU-only** 环境（或容器内无法使用 CUDA runtime），建议在 `.env` 里加：

- `RETRIEVER_FORCE_CPU=1`

这会让 retriever/reranker 即使参数文件里写了 `gpu_ids` 也自动回落到 CPU，避免 `infinity_emb` 在无 GPU 环境下报错。

#### 如果遇到 tokenizer / tiktoken 报错（`Converting from Tiktoken failed`）

如果日志里出现类似：

- `Converting from Tiktoken failed ... tokenizer.model`

通常是 `transformers` 版本过新把 `tokenizer.model` 当成 tiktoken BPE 文件解析，而 `openbmb/MiniCPM-Embedding-Light` 实际提供的是 SentencePiece `tokenizer.model`（二进制）。

解决方式：

- 重新构建镜像（本仓库已默认 pin）：`transformers==4.37.2` + `sentencepiece`

### 3.1) Generation：使用 vLLM 镜像（OpenAI-compatible）而不是在 UltraRAG 内安装 vllm

如果你看到报错：

- `Error calling tool 'generation_init': vllm is not installed. Please install it with pip install vllm`

推荐做法是：**用 compose 的 `vllm` 服务提供 OpenAI-compatible API**，然后让 UltraRAG 的 generation 走 `openai` backend 指向它（这样 UltraRAG 容器里不需要安装 `vllm` 包）。

本仓库已把默认 generation 配置改为：

- `backend: openai`
- `base_url: http://vllm:8000/v1`

启动（GPU）：

```bash
docker-compose up -d vllm ultrarag-ui
```

### CPU 版（torch CPU + faiss-cpu + llama.cpp 推理）

本仓库提供了一个独立的 CPU compose：`docker-compose.cpu.yml`，关键点：

- **torch CPU**：构建时 `TORCH_INDEX_URL_BUILD=https://download.pytorch.org/whl/cpu`
- **faiss-cpu**：`PIP_EXTRA_PACKAGES` 使用 `faiss-cpu`
- **推理用 llama.cpp**：使用 `llama-cpp` 服务提供 OpenAI-compatible `/v1/chat/completions`
- **不需要 vllm**：UltraRAG generation 走 `openai` backend 指向 llama.cpp

准备一个 GGUF 模型（示例路径），在根目录 `.env` 里配置：

- `GGUF_MODEL_HOST_PATH=/path/to/your/model.gguf`
- （可选）`GGUF_MODEL_CONTAINER_PATH=/models/model.gguf`
- （可选）`LLAMA_CPP_IMAGE=ghcr.io/ggml-org/llama.cpp:server`
- （可选）`LLAMA_CPP_PORT=8000`

启动：

```bash
docker-compose -f docker-compose.cpu.yml up -d --build
```

#### 使用宿主机本地模型目录（bind mount）给 vLLM

如果你的模型已经在宿主机上（例如：`/home/yyk/yyk03/Workspace/models/surveycpm_1218/huggingface`），推荐在根目录 `.env` 里设置：

- `VLLM_MODEL_HOST_PATH=/home/yyk/yyk03/Workspace/models/surveycpm_1218/huggingface`
- `VLLM_MODEL=/models/surveycpm`
- `VLLM_SERVED_MODEL_NAME=agentcpm-report`

然后重启 vLLM：

```bash
docker-compose up -d --build vllm
```

如果你的环境可以拉取 Docker Hub 镜像，也可以继续使用外部 OpenAI-compatible embedding server（Infinity 容器），在 `.env` 里设置：

- `RETRIEVER_BACKEND=openai`
- `RETRIEVER_OPENAI_BASE_URL=http://infinity-embed:7997/v1`
- `RETRIEVER_OPENAI_MODEL=openbmb/MiniCPM-Embedding-Light`
- （可选）`INFINITY_IMAGE` / `INFINITY_HOST_PORT` / `INFINITY_DEVICE`

#### UI 启动前自动 prebuild（解决 “Parameters not found. Build first.”）

UI 运行某些 pipeline 前，需要先 `ultrarag build` 生成：

- `examples/parameter/<pipeline>_parameter.yaml`
- `examples/server/<pipeline>_server.yaml`

默认情况下，`ultrarag-ui` / `ultrarag-ui-admin` 会在启动前自动扫描并 prebuild 前端会用到的 pipelines（`examples/*.yaml` 和 `ui/pipelines/*.yaml`，并跳过内部隐藏 pipelines）。

为了避免每次 `docker-compose up`（尤其是 `docker-compose down` 后容器重建）都重复 prebuild，本仓库已把这些 build 产物目录挂载为 **命名卷** 持久化：

- `/ultrarag/examples/parameter`
- `/ultrarag/examples/server`
- `/ultrarag/ui/pipelines/parameter`
- `/ultrarag/ui/pipelines/server`

因此通常只会在 **第一次启动**（或你删除了这些卷、或 `ULTRARAG_PREBUILD_FORCE=1`）时执行真实 build，之后会快速跳过（`skip (exists)`）。

### 3.1)（可选）CPU 专用 compose：使用本地 `.gguf` 模型作为生成模型

如果你希望 **纯 CPU** 运行，并且生成模型是本地 `.gguf`（例如 `surveycpm_1218-Q4_K_M.gguf`），可以使用本仓库新增的 `docker-compose.cpu.yml`：

1) 在 `.env` 里配置模型文件路径（宿主机）：

```bash
GGUF_MODEL_HOST_PATH=/home/yyk/yyk03/Workspace/models/surveycpm_1218/surveycpm_1218-Q4_K_M.gguf
HF_ENDPOINT=https://hf-mirrors.com
```

2) 启动（CPU 版本会自动把 generation 指向 llama.cpp OpenAI-compatible server）：

```bash
docker-compose -f docker-compose.cpu.yml up -d --build ultrarag-ui
```

> 备注：`docker-compose.cpu.yml` 使用 `ghcr.io/ggml-org/llama.cpp:server` 作为 GGUF OpenAI-compatible server（旧的 `ghcr.io/ggerganov/llama.cpp:server` tag 已不可用）。
> 如果你的网络无法访问 GHCR，需要提前配置镜像加速/代理，或手工预拉取该镜像（也可在 `.env` 里设置 `LLAMA_CPP_IMAGE` 覆盖）。

你也可以手动控制：

- 只 prebuild 指定列表（空格分隔）：

```bash
ULTRARAG_PREBUILD_PIPELINES="examples/sayhello.yaml examples/DeepResearch.yaml ui/pipelines/xxx.yaml" \
docker-compose up -d --build ultrarag-ui
```

- 强制重建（即使产物已存在）：

```bash
ULTRARAG_PREBUILD_FORCE=1 docker-compose up -d --build ultrarag-ui
```

- 关闭 prebuild：

```bash
ULTRARAG_PREBUILD_DISABLE=1 docker-compose up -d --build ultrarag-ui
```

如果 UI 已经在跑，也可以直接在容器里 build 单个 pipeline：

```bash
docker-compose exec ultrarag-ui ultrarag build examples/DeepResearch.yaml
```

#### UI 里的 Milvus 状态说明

我们现在默认起 **Milvus Server**（compose 服务名 `milvus`），并把 UI 默认指向它：`http://milvus:19530`（可用 `ULTRARAG_MILVUS_URI` 覆盖）。

如需改 Milvus 对外部署端口（宿主机端口映射），改 `.env`：

- `MILVUS_PORT=19531`（示例）

然后起服务：

```bash
docker-compose up -d milvus
```

UI 容器内访问 Milvus **不受宿主机映射端口影响**，仍然用 `http://milvus:19530`（容器内网络）。

如需指向外部 Milvus Server，可设置环境变量（或在 UI 弹窗里改 URI）：

- `ULTRARAG_MILVUS_URI=http://<milvus_host>:19530`

### 4) 运行你自己的 pipeline

把 compose 的 `ultrarag` 服务 command 替换为你想跑的 YAML，例如：

```bash
docker-compose run --rm ultrarag ultrarag run examples/rag.yaml
```

### 4.1) 跑通 `examples/DeepResearch.yaml`

DeepResearch 默认配置会走 **vLLM + GPU**，并且 retriever 默认 FAISS GPU。我们这里采用 **docker-compose 联动 vLLM**：起一个 `vllm` 服务提供 OpenAI-compatible 接口，然后让 UltraRAG 的 generation 走 `openai` backend 指向 `http://vllm:8000/v1`（参数文件：`examples/parameter/DeepResearch_docker_parameter.yaml`）。

注意：本仓库环境的 `docker-compose` 版本对 `gpus: all` 语法不兼容，因此 `vllm` 服务使用了 `runtime: nvidia`。如需使用 GPU，请确保已安装 NVIDIA Container Toolkit。

先构建：

```bash
docker-compose build ultrarag-deepresearch vllm
docker-compose run --rm ultrarag-deepresearch
```

这条命令会在容器里先执行：

- `ultrarag build examples/DeepResearch.yaml`
- `ultrarag run examples/DeepResearch.yaml --param examples/parameter/DeepResearch_docker_parameter.yaml`

### 5) 环境变量（LLM Key 等）

建议在本目录创建 `.env`（Docker Compose 会自动读取）。你也可以先复制 `env.example`：

```bash
cp env.example .env
```

然后按需填写，例如：

```bash
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
HF_ENDPOINT=https://hf-mirrors.com
ULTRARAG_UI_PORT=5050
ULTRARAG_ADMIN_UI_PORT=5051
PIP_EXTRA_PACKAGES=faiss-cpu pymilvus chonkie tiktoken
VLLM_MODEL=OpenBMB/MiniCPM4-8B
VLLM_SERVED_MODEL_NAME=minicpm4-8b
VLLM_PORT=8000
```

### 6) 输出与持久化

pipeline 运行产生的 `output/` 已通过命名卷 `ultrarag_output` 持久化。


