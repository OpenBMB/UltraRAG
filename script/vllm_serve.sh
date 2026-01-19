#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --served-model-name minicpm4-8b \
    --model OpenBMB/MiniCPM4-8B \
    --trust-remote-code \
    --host 127.0.0.1 \
    --port 65501 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 2 \
    --enforce-eager