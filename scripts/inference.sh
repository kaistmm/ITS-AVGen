#!/bin/bash
# BON (Best-of-N) inference script
# Usage: bash scripts/inference.sh [GPU_ID] [NSHARD] [SHARD_ID]
# Examples:
#   Single GPU: bash scripts/inference.sh 0
#   Multi-GPU: for i in {0..3}; do bash scripts/inference.sh $i 4 $i & done; wait

GPU_ID=${1:-0}
NSHARD=${2:-1}
SHARD_ID=${3:-0}

cfg_file="configs/javisdit-v0-1/inference/sample_240p4s.py"
save_dir="samples/generated_bon_$(date +%Y%m%d_%H%M%S)"

resolution=240p
num_frames=4s
aspect_ratio="9:16"

mkdir -p ${save_dir}
cp ${cfg_file} ${save_dir}/config.py

echo "Running BON (Best-of-N) inference"
echo "GPU ${GPU_ID}, Shard ${SHARD_ID}/${NSHARD}"
echo "Output directory: ${save_dir}"

CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/inference.py ${cfg_file} \
    --resolution ${resolution} --num-frames ${num_frames} --aspect-ratio ${aspect_ratio} \
    --save-dir ${save_dir} --verbose 1 \
    --nshard ${NSHARD} --shard-id ${SHARD_ID}
