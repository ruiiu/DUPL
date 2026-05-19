#!/bin/bash

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=XenoZLH/MMRL30k@train \
    data.val_files=XenoZLH/MMRL30k@k12_test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    algorithm.disable_kl=true \
    algorithm.online_filtering=true \
    algorithm.adv_estimator=grpo_dupl \
    algorithm.dupl.enabled=true \
    algorithm.dupl.gaussian_noise_std=0.2 \
    algorithm.dupl.sampling_strategy=adaptive \
    algorithm.use_entropy_shaping=true \
    algorithm.entropy_alpha=0.4 \
    algorithm.entropy_kappa=2.0 \
    trainer.experiment_name=dupl_dapo_7b \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1
