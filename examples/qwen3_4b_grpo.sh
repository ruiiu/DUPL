#!/bin/bash

MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=XenoZLH/MMRL30k@train \
    data.val_files=XenoZLH/MMRL30k@k12_test \
    data.format_prompt=./examples/format_prompt/math_qwen3.jinja \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3_4b_grpo \
    algorithm.adv_estimator=grpo \
    algorithm.kl_coef=0.0 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1
