FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON=python3.12 \
    PATH="/root/.local/bin:/ultrarag/.venv/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-dev \
        curl ca-certificates git build-essential && \
    update-ca-certificates && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /ultrarag
COPY . .

RUN uv sync --frozen --no-dev \
    --extra retriever --extra generation --extra corpus --extra evaluation

EXPOSE 5050

CMD ["ultrarag", "show", "ui", "--admin", "--port", "5050", "--host", "0.0.0.0"]