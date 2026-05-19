#!/bin/bash

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=XenoZLH/MMRL30k@train \
    data.val_files=XenoZLH/MMRL30k@k12_test \
    worker.actor.model.model_path=${MODEL_PATH} \
    algorithm.adv_estimator=grpo \
    algorithm.use_entropy_shaping=true \
    algorithm.entropy_alpha=0.4 \
    algorithm.entropy_kappa=2.0 \
    algorithm.kl_coef=0.0 \
    trainer.experiment_name=entropy_3b_steps_200_n8 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1
