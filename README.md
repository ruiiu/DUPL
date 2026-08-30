# DUPL

Dual-Path Uncertainty Learning (DUPL) for robust multimodal math reasoning.

This repository is an [EasyR1](https://github.com/hiyouga/EasyR1)/[veRL](https://github.com/volcengine/verl) fork. The original [EasyR1](https://github.com/hiyouga/EasyR1) training stack is still used for distributed GRPO, FSDP, vLLM rollout, checkpointing, and logging, but this fork adds a DUPL path for perceptual uncertainty estimation and uncertainty-aware advantage shaping.

## What This Approach Does

DUPL trains a vision-language model on math and visual reasoning prompts by comparing behavior on two views of the same sample:

- **Raw path**: the original image and prompt are rolled out normally.
- **Augmented path**: the image is perturbed with controlled visual augmentations such as flips, small rotations, color jitter, and Gaussian noise.
- **Perceptual uncertainty**: the trainer estimates how much the policy changes between the raw and augmented paths. In this codebase the main signal is based on raw-vs-augmented rollout log-probability divergence.
- **DUPL advantage**: the `grpo_dupl` estimator combines outcome rewards with perceptual uncertainty and an optional KL-style consistency or exploration penalty.
- **Curriculum**: DUPL can start with stronger augmentation/exploration and anneal toward lower augmentation probability or stronger consistency.

The implementation is trainer-level, not only a reward-function wrapper. The dual-path batch is created during training and additional tensors such as `perceptual_uncertainty` are merged back into the batch before advantage computation.

## Main Code Paths

| File | Purpose |
| --- | --- |
| `verl/models/uncertainty/perceptual_uncertainty.py` | Visual augmentor, uncertainty estimation, dual-path utilities. |
| `verl/workers/fsdp_workers.py` | Runs raw and augmented paths and computes per-sample perceptual uncertainty. |
| `verl/trainer/ray_trainer.py` | Inserts DUPL processing into the [EasyR1](https://github.com/hiyouga/EasyR1) training loop and logs DUPL metrics. |
| `verl/trainer/core_algos.py` | Adds `AdvantageEstimator.GRPO_DUPL` and `compute_grpo_outcome_advantage_dupl`. |
| `verl/trainer/config.py` | Defines the `algorithm.dupl` configuration block and validates `grpo_dupl`. |
| `examples/config.yaml` | Base training config with DUPL fields. |
| `examples/reward_function/math.py` | Boxed-answer math reward used by the provided scripts. |

## Supported Experiments

The scripts in `examples/` cover DUPL and baselines for Qwen2.5/Qwen3 models:

| Script | What it runs |
| --- | --- |
| `examples/qwen3_vl_8b_dupl.sh` | Qwen3-VL-8B DUPL on `XenoZLH/MMRL30k`. |
| `examples/qwen3_vl_4b_dupl.sh` | Qwen3-VL-4B DUPL on `XenoZLH/MMRL30k`. |
| `examples/dupl_7b.sh` | Qwen2.5-VL-7B DUPL recipe. |
| `examples/dupl_3b.sh` | Qwen2.5-VL-3B DUPL recipe. |
| `examples/grpo_7b.sh`, `examples/grpo_3b.sh` | Standard GRPO baselines. |
| `examples/entropy_7b.sh`, `examples/entropy_3b.sh` | Entropy-shaping baselines. |
| `examples/dupl_7b_no_kl.sh` | DUPL without KL penalty. |
| `examples/dupl_7b_forward_kl.sh` | DUPL with forward-KL-only behavior. |
| `examples/dupl_7b_fix_prob.sh` | DUPL with fixed augmentation probability. |
| `examples/dupl_7b_no_entropy.sh` | DUPL without entropy shaping. |

The scripts may include local proxy exports or local post-run GPU checks. Remove or edit those lines if they are not valid in your environment.

## Installation

Use the same environment style as EasyR1: Python 3.9+, PyTorch, transformers, flash-attn, vLLM, Ray, and FSDP-capable GPUs.

```bash
cd DUPL
pip install -e .
```

If your cluster uses ModelScope or a Hugging Face mirror, set those environment variables before launching training:

```bash
export USE_MODELSCOPE_HUB=1
export HF_ENDPOINT=https://hf-mirror.com
```

## Run DUPL

The most direct Qwen3-VL run is:

```bash
cd DUPL
bash examples/qwen3_vl_8b_dupl.sh
```

That script expands to the important overrides below:

```bash
python3 -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files=XenoZLH/MMRL30k@train \
  data.val_files=XenoZLH/MMRL30k@k12_test \
  data.format_prompt=./examples/format_prompt/math_qwen3.jinja \
  worker.actor.model.model_path=Qwen/Qwen3-VL-8B-Instruct \
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
```

For Qwen2.5-VL:

```bash
bash examples/dupl_7b.sh
bash examples/dupl_3b.sh
```

## Configuration Reference

The central DUPL fields live under `algorithm.dupl` in `examples/config.yaml`.

| Field | Meaning |
| --- | --- |
| `enabled` | Enables the dual-path processing hook. If true, config validation switches the estimator to `grpo_dupl` when needed. |
| `augmentation_strength` | Scales color jitter and Gaussian noise intensity. |
| `gaussian_noise_std` | Base standard deviation for Gaussian image noise. |
| `uncertainty_alpha` | Weight of the uncertainty term in DUPL advantage shaping. |
| `uncertainty_kappa` | Temperature/normalizer for uncertainty scaling. |
| `sampling_strategy` | Augmentation schedule strategy. Provided scripts use `adaptive` or fixed-probability ablations. |
| `fixed_prob` | Augmentation probability when using fixed sampling. |
| `initial_aug_prob`, `final_aug_prob` | Start and end probabilities for adaptive augmentation. |
| `kl_penalty_weight` | Weight for the raw-vs-augmented KL penalty. |
| `top_k_for_kl` | Number of top tokens considered by the KL approximation. |
| `exploration_ratio` | Fraction of training treated as early exploration. |
| `transition_ratio` | Fraction used for transition between exploration and consistency. |
| `enable_kl_transition` | Enables schedule-based KL sign or behavior transition. |
| `use_kl_penalty` | Turns the dual-path KL penalty on or off. |
| `use_forward_kl_only` | Uses forward KL only for the penalty. |

Two generic shaping knobs are often used with DUPL:

- `algorithm.use_entropy_shaping=true`
- `algorithm.entropy_alpha=0.4`
- `algorithm.entropy_kappa=2.0`

## Data Format

The default config uses [EasyR1](https://github.com/hiyouga/EasyR1)-style Hugging Face dataset references:

```yaml
data:
  train_files: hiyouga/math12k@train
  val_files: hiyouga/math12k@test
  prompt_key: problem
  answer_key: answer
  image_key: images
  format_prompt: ./examples/format_prompt/math.jinja
```

The Qwen3-VL DUPL script overrides this to:

```yaml
data.train_files: XenoZLH/MMRL30k@train
data.val_files: XenoZLH/MMRL30k@k12_test
data.format_prompt: ./examples/format_prompt/math_qwen3.jinja
```

For custom data, provide an [EasyR1](https://github.com/hiyouga/EasyR1)-compatible text or image-text dataset with the prompt, answer, and image columns named by the config.

## Evaluation And Checkpoint Merge

Merge a saved actor checkpoint to Hugging Face format:

```bash
python3 scripts/model_merger.py \
  --local_dir checkpoints/DUPL/<experiment_name>/global_step_<step>/actor
```

Run the local evaluation launcher after editing model and data paths for your environment:

```bash
bash evaluation/eval.sh
```

## Citation

This framework builds on [EasyR1](https://github.com/hiyouga/EasyR1) and [veRL](https://github.com/volcengine/verl). If you use DUPL, please cite:

```bibtex
@misc{liu2026dualuncertaintyguidedpolicylearning,
      title={Dual-Uncertainty Guided Policy Learning for Multimodal Reasoning}, 
      author={Rui Liu and Dian Yu and Tong Zheng and Runpeng Dai and Zongxia Li and Wenhao Yu and Zhenwen Liang and Linfeng Song and Haitao Mi and Pratap Tokekar and Dong Yu},
      year={2026},
      eprint={2510.01444},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.01444}, 
}
```
