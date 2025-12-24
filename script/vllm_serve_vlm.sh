#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --served-model-name qwen3-8b-vl \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 65503 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt '{"image":3}' \
    --enforce-eager