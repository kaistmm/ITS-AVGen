#!/bin/bash
# Reward server for inference time scaling
# Usage: bash vqa_server.sh [GPU_ID]
# Example: bash vqa_server.sh 0

GPU_ID=${1:-0}  # Default to GPU 0 if not specified

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=${GPU_ID} python reward_model/vqa_server.py \
    --gpu 0 \
    --addr 5001 \
    --reward_model VideoReward \
    --align_model JavisScore \
    --audio_model None