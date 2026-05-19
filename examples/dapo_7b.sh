#!/bin/bash

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=XenoZLH/MMRL30k@train \
    data.val_files=XenoZLH/MMRL30k@k12_test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    algorithm.adv_estimator=grpo \
    algorithm.kl_coef=0.0 \
    algorithm.disable_kl=true \
    algorithm.online_filtering=true \
    trainer.experiment_name=dapo_7b \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1
