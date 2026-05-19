#!/bin/bash

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=XenoZLH/MMRL30k@train \
    data.val_files=XenoZLH/MMRL30k@k12_test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=MMRL30k_grpo_n_8 \
    algorithm.adv_estimator=grpo \
    worker.rollout.n=8 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=1
