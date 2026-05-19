#!/bin/bash

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=XenoZLH/MMRL30k@train \
    data.val_files=XenoZLH/MMRL30k@k12_test \
    worker.actor.model.model_path=${MODEL_PATH} \
    algorithm.adv_estimator=grpo_exploration \
    algorithm.use_entropy_shaping=true \
    algorithm.entropy_alpha=0.4 \
    algorithm.entropy_kappa=2.0 \
    algorithm.kl_coef=0.0 \
    algorithm.exploration.use_exploration=true \
    algorithm.exploration.beta=0.3 \
    algorithm.exploration.gamma=0.2 \
    algorithm.exploration.use_ngram_diversity=true \
    algorithm.exploration.use_prediction_consistency=true \
    algorithm.exploration.ngram_size=2 \
    worker.rollout.n=8 \
    trainer.experiment_name=MMRL30k_grpo_entropy_explore_beta0.3_gamma0.2_n8 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.max_steps=20
