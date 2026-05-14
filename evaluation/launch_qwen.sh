#!/bin/bash

# -------- CONFIGURATION --------
# HEAD_NODE_IP="29.127.68.253"             # Head node IP
# HEAD_NODE_PORT="6379"
# Worker node IPs, "29.119.96.254" 29.232.224.137 "29.127.36.241" "29.191.211.78" 29.232.228.185, 29.127.80.107, 29.160.40.86, 29.160.43.142

SSH_USER="root"                        # SSH user for all nodes
CONDA_ENV="easyr1"                        # Conda environment name
NETWORK_INTERFACE="bond1"               # NIC for NCCL
RAY_GPU_COUNT=8                         # GPUs per node for Ray

# -------- START RAY HEAD --------
# echo "[HEAD] Starting Ray head node..."

export NCCL_SOCKET_IFNAME=$NETWORK_INTERFACE
export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"

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
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --max-num-batched-tokens 32768 \
    --trust-remote-code 

python ../matrix_multiplication_gpus.py --gpus 8 --size 5000

# ray job submit \
#     --address=http://${HEAD_NODE_IP}:8265 \
#     --no-wait \
#     -- python -m vllm.entrypoints.openai.api_server \
#     --model ${MODEL} \
#     --tensor-parallel-size 8 \
#     --host 0.0.0.0 \
#     --port 8000 \
#     --gpu-memory-utilization 0.9 \
#     --max-model-len 4096 \
#     --trust-remote-code
