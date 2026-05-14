#!/bin/bash
# -------- CONFIGURATION --------
HEAD_NODE_IP="29.119.84.75"             # Head node IP
HEAD_NODE_PORT="6379"
WORKER_NODES=()  # Worker node IPs, "29.119.96.254" 29.232.224.137 "29.127.36.241" "29.191.211.78" 29.232.228.185
#29.127.50.139, 29.127.80.107, 29.119.98.124
SSH_USER="root"                        # SSH user for all nodes
CONDA_ENV="easyr1"                        # Conda environment name
NETWORK_INTERFACE="bond1"               # NIC for NCCL
RAY_GPU_COUNT=8                         # GPUs per node for Ray

# -------- START RAY HEAD --------
echo "[HEAD] Starting Ray head node..."

export NCCL_SOCKET_IFNAME=$NETWORK_INTERFACE
export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"

pkill -f python 
# ray stop > /dev/null 2>&1
# ray start --head --dashboard-host=0.0.0.0 --num-gpus=$RAY_GPU_COUNT

sleep 3  # Wait for Ray head to start

# -------- START RAY WORKERS --------
for NODE in "${WORKER_NODES[@]}"; do
  echo "[WORKER] Connecting to $NODE and starting Ray worker..."
  ssh -p 36000 ${SSH_USER}@$NODE "
    conda activate $CONDA_ENV
    export NCCL_SOCKET_IFNAME=$NETWORK_INTERFACE
    export http_proxy="http://star-proxy.oa.com:3128"
    export https_proxy="http://star-proxy.oa.com:3128"
    ray start --address=${HEAD_NODE_IP}:${HEAD_NODE_PORT} --num-gpus=$RAY_GPU_COUNT
  "
done

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

# ray job submit \
#     --address=http://${HEAD_NODE_IP}:8265 \
#     --no-wait \
#     -- \
    python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=XenoZLH/MMRL30k@train \
    data.val_files=XenoZLH/MMRL30k@k12_test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.n=5 \
    trainer.experiment_name=grpo_3b_steps_200_n5 \
    algorithm.adv_estimator=grpo \
    algorithm.kl_coef=0.0 \
    trainer.n_gpus_per_node=$RAY_GPU_COUNT \

    nohup python ../matrix_multiplication_gpus.py --gpus 8 --size 5000 > /dev/null 2>&1 &


