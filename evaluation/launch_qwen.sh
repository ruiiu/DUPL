#!/bin/bash

# -------- CONFIGURATION --------

SSH_USER="root"                        # SSH user for all nodes
CONDA_ENV="easyr1"                        # Conda environment name
NETWORK_INTERFACE="bond1"               # NIC for NCCL
RAY_GPU_COUNT=8                         # GPUs per node for Ray

# -------- START RAY HEAD --------
# echo "[HEAD] Starting Ray head node..."

# pkill -f python 
# ray stop > /dev/null 2>&1
# ray start --head --dashboard-host=0.0.0.0 --num-gpus=$RAY_GPU_COUNT

# sleep 3  # Wait for Ray head to start

MODEL="Qwen/Qwen2.5-72B-Instruct"
# MODEL="Qwen/Qwen3-32B"

# ray job submit \
#     --address=http://${HEAD_NODE_IP}:8265 \
#     --no-wait \

python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL} \
    --tensor-parallel-size 8 \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 16384 \
    --max-num-batched-tokens 32768 \
    --trust-remote-code 

python ../matrix_multiplication_gpus.py --gpus 8 --size 5000
