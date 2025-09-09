#!/bin/bash

MODEL="OpenGVLab/InternVL2_5-8B"
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export NCCL_P2P_DISABLE=1
# export CUDA_VISIBLE_DEVICES="0"
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PORT=8000
vllm serve $MODEL \
    --port $PORT \
    --trust-remote-code \
    --limit-mm-per-prompt image=4 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.97 \
    --disable-log-requests
