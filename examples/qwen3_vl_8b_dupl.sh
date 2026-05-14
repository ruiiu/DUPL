#!/bin/bash
# -------- CONFIGURATION --------

RAY_GPU_COUNT=8                         # GPUs per node for Ray

export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"

MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=XenoZLH/MMRL30k@train \
    data.val_files=XenoZLH/MMRL30k@k12_test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3_8b_dupl \
    algorithm.adv_estimator=grpo_dupl \
    algorithm.dupl.enabled=true \
    algorithm.dupl.gaussian_noise_std=0.2 \
    algorithm.dupl.sampling_strategy=adaptive \
    algorithm.use_entropy_shaping=true \
    algorithm.entropy_alpha=0.4 \
    algorithm.entropy_kappa=2.0 \
    algorithm.disable_kl=true \
    trainer.n_gpus_per_node=$RAY_GPU_COUNT 

python ../matrix_multiplication_gpus.py --gpus 8 --size 5000