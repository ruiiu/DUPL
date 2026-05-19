#!/bin/bash

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=XenoZLH/MMRL30k@train \
    data.val_files=XenoZLH/MMRL30k@k12_test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=dupl_forward_kl_3b_steps_200_n8 \
    algorithm.adv_estimator=grpo_dupl \
    algorithm.dupl.enabled=true \
    algorithm.dupl.augmentation_strength=1.0 \
    algorithm.dupl.gaussian_noise_std=0.2 \
    algorithm.dupl.sampling_strategy=adaptive \
    algorithm.dupl.fixed_prob=0.5 \
    algorithm.dupl.initial_aug_prob=1.0 \
    algorithm.dupl.final_aug_prob=0.0 \
    algorithm.dupl.enable_kl_transition=false \
    algorithm.dupl.exploration_ratio=0.5 \
    algorithm.dupl.transition_ratio=0.25 \
    algorithm.dupl.kl_penalty_weight=1.0 \
    algorithm.dupl.top_k_for_kl=100 \
    algorithm.dupl.use_forward_kl_only=true \
    algorithm.use_entropy_shaping=true \
    algorithm.entropy_alpha=0.4 \
    algorithm.entropy_kappa=2.0 \
    algorithm.disable_kl=true \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1
