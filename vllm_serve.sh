#!/bin/bash

# MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
# MODEL="OpenGVLab/InternVL2-8B",
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export NCCL_P2P_DISABLE=1
# export CUDA_VISIBLE_DEVICES="0"
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PORT=8000
vllm serve "OpenGVLab/InternVL2_5-8B" \
    --port 8000 \
    --trust-remote-code \
    --limit-mm-per-prompt image=4 \
    # --max-model-len 8192 \
    # --gpu-memory-utilization 0.97 \
    --disable-log-requests
