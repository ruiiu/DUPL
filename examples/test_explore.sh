#!/bin/bash

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=XenoZLH/MMRL30k@train \
    data.val_files=XenoZLH/MMRL30k@k12_test \
    worker.actor.model.model_path=${MODEL_PATH} \
    algorithm.adv_estimator=grpo_exploration \
    algorithm.exploration.use_exploration=true \
    algorithm.exploration.use_ngram_diversity=true \
    algorithm.exploration.use_prediction_consistency=true \
    algorithm.exploration.beta=0.4 \
    algorithm.exploration.gamma=0.2 \
    algorithm.exploration.ngram_size=2 \
    algorithm.kl_coef=0.0 \
    worker.rollout.n=8 \
    trainer.experiment_name=MMRL30k_grpo_explore_freeze_beta0.4_gamma0.2_n8 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_epochs=1
