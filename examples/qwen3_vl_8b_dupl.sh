#!/bin/bash

MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=XenoZLH/MMRL30k@train \
    data.val_files=XenoZLH/MMRL30k@k12_test \
    data.format_prompt=./examples/format_prompt/math_qwen3.jinja \
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
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1
