FROM nvidia/cuda:12.2.2-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON=python3.11 \
    PATH="/root/.local/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-distutils \
        curl ca-certificates git build-essential && \
    update-ca-certificates && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /ultrarag
COPY . .

RUN uv sync --system --frozen --no-dev \
    --extra retriever --extra generation --extra corpus --extra evaluation

EXPOSE 5050

CMD ["ultrarag", "show", "ui", "--admin", "--port", "5050", "--host", "0.0.0.0"]