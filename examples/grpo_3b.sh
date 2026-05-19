#!/bin/bash

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=XenoZLH/MMRL30k@train \
    data.val_files=XenoZLH/MMRL30k@k12_test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.n=5 \
    trainer.experiment_name=grpo_3b_steps_200_n5 \
    algorithm.adv_estimator=grpo \
    algorithm.kl_coef=0.0 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1
