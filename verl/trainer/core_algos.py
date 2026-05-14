# Copyright 2022 The HuggingFace Team
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union
import re

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import torch_functional as VF

from mathruler.grader import extract_boxed_content


if TYPE_CHECKING:
    from .config import AlgorithmConfig


class KLController(ABC):
    kl_coef: float
    """KL coefficient."""

    @abstractmethod
    def update(self, current_kl: float, n_steps: int):
        """Update kl_coef according to current KL."""
        ...


class AdaptiveKLController(KLController):
    """Adaptive KL controller described in: https://arxiv.org/pdf/1909.08593.pdf

    Copied from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L54"""

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: float):
        self.kl_coef = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.kl_coef *= mult


class FixedKLController(KLController):
    """Fixed KL controller.

    Copeid from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L72"""

    def __init__(self, init_kl_coef: float):
        self.kl_coef = init_kl_coef

    def update(self, current_kl: float, n_steps: int):
        pass


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    GRPO_EXPLORATION = "grpo_exploration"
    GRPO_DUPL = "grpo_dupl"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"


ADV_ESTIMATOR_MAP: dict[str, Any] = {}


def get_kl_controller(algorithm_config: "AlgorithmConfig") -> KLController:
    """Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L319"""
    if algorithm_config.kl_type == "fixed":
        kl_ctrl = FixedKLController(init_kl_coef=algorithm_config.kl_coef)
    elif algorithm_config.kl_type == "adaptive":
        assert algorithm_config.kl_horizon > 0, f"horizon must be larger than 0. Got {algorithm_config.kl_horizon}."
        kl_ctrl = AdaptiveKLController(
            init_kl_coef=algorithm_config.kl_coef,
            target_kl=algorithm_config.kl_target,
            horizon=algorithm_config.kl_horizon,
        )
    else:
        raise ValueError(f"Unknown kl type: {algorithm_config.kl_type}.")

    return kl_ctrl


def register_adv_estimator(name: AdvantageEstimator):
    """Decorator to register a advantage estimator function with a given name."""

    def decorator(fn):
        wrapped_fn = torch.no_grad()(fn)
        ADV_ESTIMATOR_MAP[getattr(name, "value", name)] = wrapped_fn
        return wrapped_fn

    return decorator


def compute_advantage_return(name: AdvantageEstimator, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute advantage and return for a given advantage estimator."""
    return ADV_ESTIMATOR_MAP[getattr(name, "value", name)](**kwargs)


@register_adv_estimator(AdvantageEstimator.GAE)
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapted from https://github.com/huggingface/trl/blob/v0.16.0/trl/trainer/ppo_trainer.py#L513

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). The token after eos tokens have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    nextvalues = 0
    lastgaelam = 0
    advantages_reversed = []
    gen_len = token_level_rewards.shape[-1]
    for t in reversed(range(gen_len)):
        delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
        gaelam = delta + gamma * lam * lastgaelam

        if response_mask[:, t]:  # skip values and TD-error on observation tokens
            nextvalues = values[:, t]
            lastgaelam = gaelam

        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    advantages = VF.masked_whiten(advantages, response_mask)
    return advantages, returns


@register_adv_estimator(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)`
            shape: (bs,)
        eps: `(float)`
            epsilon value to avoid division by zero

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor(id2score[idx]))

    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@register_adv_estimator(AdvantageEstimator.GRPO_EXPLORATION)
def compute_grpo_outcome_advantage_with_exploration(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index,
    response_texts: List[str] = None,
    ground_truths: List[str] = None,
    beta: float = 0.1,
    gamma: float = 0.1,
    use_ngram_diversity: bool = True,
    use_prediction_consistency: bool = True,
    ngram_size: int = 2,
    eps: float = 1e-6,
    log_probs: torch.Tensor = None,
    alpha: float = 0.4,
    kappa: float = 2.0,
    use_entropy_shaping: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO with exploration bonuses and optionally entropy-based advantage shaping.
    """
    if response_texts is not None and ground_truths is not None:
        scores = compute_grpo_enhanced_rewards(
            token_level_rewards=token_level_rewards,
            response_texts=response_texts,
            ground_truths=ground_truths,
            index=index,
            beta=beta,
            gamma=gamma,
            use_ngram_diversity=use_ngram_diversity,
            use_prediction_consistency=use_prediction_consistency,
            ngram_size=ngram_size,
            correctness_threshold=0.5,
        )
    else:
        scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean, id2std = {}, {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor(id2score[idx]))
    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

    returns = scores.unsqueeze(-1) * response_mask
    advantages = returns

    if use_entropy_shaping and log_probs is not None:
        entropy = compute_token_entropy(log_probs, response_mask)
        entropy_shaping_term = compute_entropy_advantage_shaping(advantages, entropy, response_mask, alpha, kappa)
        advantages = advantages + entropy_shaping_term

    return advantages, returns


@torch.no_grad()
def compute_grpo_outcome_advantage_dupl(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    raw_log_probs: Optional[torch.Tensor] = None,
    aug_log_probs: Optional[torch.Tensor] = None,
    use_augmented_branch: Union[bool, torch.Tensor] = True,
    eps: float = 1e-6,
    alpha: float = 0.4,
    kappa: float = 2.0,
    use_entropy_shaping: bool = False,
    kl_penalty_weight: float = 0.1,
    current_step: int = 0,
    total_training_steps: int = 200,
    exploration_ratio: float = 0.45,
    transition_ratio: float = 0.23,
    enable_kl_transition: bool = False,
    use_kl_penalty: bool = True,
    use_forward_kl_only: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Compute GRPO outcome-based advantages with DUPL (dual-path uncertainty learning) and curriculum KL penalty.
    """
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor(id2score[idx]))
    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

    returns = scores.unsqueeze(-1) * response_mask
    advantages = returns
    metadata = {}

    kl_penalty = torch.zeros_like(advantages)
    if use_kl_penalty and raw_log_probs is not None and aug_log_probs is not None:
        raw_probs = torch.exp(raw_log_probs)
        aug_probs = torch.exp(aug_log_probs)
        forward_kl = F.kl_div(torch.log(aug_probs + 1e-8), raw_probs, reduction='none')

        if use_forward_kl_only:
            kl_penalty = forward_kl
        else:
            inverse_kl = F.kl_div(torch.log(raw_probs + 1e-8), aug_probs, reduction='none')
            kl_penalty = (forward_kl + inverse_kl) / 2.0

        kl_penalty = kl_penalty.detach() * response_mask
        metadata["kl_penalty"] = kl_penalty.mean().item()
        metadata["kl_penalty_std"] = kl_penalty.std().item()

        if isinstance(use_augmented_branch, torch.Tensor):
            use_augmented_mask = use_augmented_branch.float().unsqueeze(-1)
        else:
            use_augmented_mask = float(use_augmented_branch)

        should_apply_kl = use_augmented_branch.any().item() if isinstance(use_augmented_branch, torch.Tensor) else use_augmented_branch

        if should_apply_kl:
            exploration_steps = int(total_training_steps * exploration_ratio)
            transition_steps = int(total_training_steps * transition_ratio)

            if not enable_kl_transition:
                kl_multiplier = 1.0
                metadata["kl_penalty_mode"] = "exploration_fixed"
            elif current_step < exploration_steps:
                kl_multiplier = 1.0
                metadata["kl_penalty_mode"] = "exploration"
            elif current_step < exploration_steps + transition_steps:
                transition_progress = (current_step - exploration_steps) / max(transition_steps, 1)
                kl_multiplier = 1.0 - 2.0 * transition_progress
                metadata["kl_penalty_mode"] = "transition"
            else:
                kl_multiplier = -1.0
                metadata["kl_penalty_mode"] = "consistency"

            kl_term = kl_penalty_weight * kl_multiplier * kl_penalty
            advantage_magnitude_term = torch.abs(advantages) / 2.0
            kl_penalty_contribution = torch.min(torch.abs(kl_term), advantage_magnitude_term) * torch.sign(kl_term)
            kl_penalty_contribution = kl_penalty_contribution * use_augmented_mask
            advantages = advantages + kl_penalty_contribution

            metadata["current_step"] = current_step
            metadata["total_training_steps"] = total_training_steps
            if enable_kl_transition:
                metadata["kl_multiplier"] = kl_multiplier
            metadata["kl_contribution_mean"] = kl_penalty_contribution.mean().item()
            metadata["kl_contribution_std"] = kl_penalty_contribution.std().item()

    if raw_log_probs is not None:
        raw_entropy = compute_token_entropy(raw_log_probs, response_mask)
    else:
        raw_entropy = None
    if aug_log_probs is not None:
        aug_entropy = compute_token_entropy(aug_log_probs, response_mask)
    else:
        aug_entropy = None

    if use_entropy_shaping:
        if isinstance(use_augmented_branch, torch.Tensor):
            if aug_entropy is not None and raw_entropy is not None:
                entropy = torch.where(use_augmented_branch.unsqueeze(-1), aug_entropy, raw_entropy)
            else:
                entropy = aug_entropy if aug_entropy is not None else raw_entropy
        else:
            entropy = aug_entropy if use_augmented_branch else raw_entropy

        if entropy is not None:
            entropy_shaping_term = compute_entropy_advantage_shaping(advantages, entropy, response_mask, alpha, kappa)
            advantages = advantages + entropy_shaping_term

    return advantages, returns, metadata


@register_adv_estimator(AdvantageEstimator.RLOO)
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)`
            shape: (bs,)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2sum = {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        id2sum[idx] = torch.sum(torch.tensor(id2score[idx]))

    for i in range(bsz):
        sample_num = len(id2score[index[i]])
        assert sample_num > 1, "RLOO needs rollout.n > 1."
        baseline = (id2sum[index[i]] - scores[i]) / (sample_num - 1)
        scores[i] = scores[i] - baseline

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@register_adv_estimator(AdvantageEstimator.REINFORCE_PLUS_PLUS)
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    returns = torch.zeros_like(token_level_rewards)
    running_return = 0
    for t in reversed(range(token_level_rewards.shape[1])):
        running_return = token_level_rewards[:, t] + gamma * running_return
        returns[:, t] = running_return
        # Reset after EOS
        running_return = running_return * response_mask[:, t]

    advantages = VF.masked_whiten(returns, response_mask)
    return advantages, returns


@register_adv_estimator(AdvantageEstimator.REMAX)
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    advantages = (token_level_rewards.sum(dim=-1) - reward_baselines) * response_mask
    returns = (token_level_rewards * response_mask).flip(dims=(-1,)).cumsum(dim=-1).flip(dims=(-1,))
    return advantages, returns


def compute_rewards(
    token_level_scores: torch.Tensor,
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_ratio: float,
) -> torch.Tensor:
    kl = log_probs - ref_log_probs
    return token_level_scores - kl * kl_ratio


def average_loss(
    values: torch.Tensor, mask: torch.Tensor, mode: Literal["token", "seq"], eps: float = 1e-8
) -> torch.Tensor:
    """Average the policy loss.

    Args:
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        mask: `(torch.Tensor)`
            shape: (bs, response_length)
        mode: `(Literal["token", "seq"])`
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means
        eps: `(float)`
            epsilon value

    Returns:
        loss: `a scalar torch.Tensor`
    """
    if mode == "token":
        return VF.masked_mean(values, mask, eps=eps)
    elif mode == "seq":
        return ((values * mask).sum(-1) / (mask.sum(-1) + eps)).mean()
    else:
        raise NotImplementedError(f"Unknown mode: {mode}.")


def compute_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_dual: float,
    tau_positive: float,
    tau_negative: float,
    loss_type: Literal["default", "gspo", "gspo_token", "cispo", "sapo"],
    loss_avg_mode: Literal["token", "seq"],
    **kwargs,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the clipped policy objective and related metrics for PPO.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L568

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        clip_ratio_low: (float)
            The lower clip range used in PPO. See https://arxiv.org/abs/1707.06347
        clip_ratio_high: (float)
            The higher clip range used in DAPO. See https://arxiv.org/pdf/2503.14476
        clip_ratio_dual: (float)
            The dual clip range used in Dual-clip PPO. See https://arxiv.org/pdf/1912.09729
        tau_positive: (float)
            The temperature for control the positive tokens' clipping in SAPO. See https://arxiv.org/pdf/2511.20347
        tau_negative: (float)
            The temperature for control the negative tokens' clipping in SAPO. See https://arxiv.org/pdf/2511.20347
        loss_avg_mode: (Literal["token", "seq"])
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac_higher: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a higher value
        pg_clipfrac_lower: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a lower value
        ppo_kl: (float)
            a float number indicating the mean KL divergence between the old policy and the new policy
        entropy_loss: (float)
            a float number indicating the mean entropy loss

    """
    negative_approx_kl = log_probs - old_log_probs
    if loss_type in ["gspo", "gspo_token"]:
        # compute sequence-level importance ratio
        negative_approx_kl_in_seq = VF.masked_mean(negative_approx_kl, response_mask, dim=-1)
        # combined ratio at token level
        if loss_type == "gspo_token":
            log_importance_ratio = negative_approx_kl_in_seq.detach().unsqueeze(-1) + log_probs - log_probs.detach()
        else:
            log_importance_ratio = negative_approx_kl_in_seq.unsqueeze(-1) * response_mask
    else:
        log_importance_ratio = negative_approx_kl

    # clamp the ratio before exp to avoid nan grad
    # see: https://github.com/pytorch/pytorch/issues/10729
    ratio = torch.exp(torch.clamp(log_importance_ratio, -20.0, 20.0))
    clipped_ratio = torch.exp(
        torch.clamp(log_importance_ratio, np.log(1.0 - clip_ratio_low), np.log(1.0 + clip_ratio_high))
    )

    # pg metrics
    metrics = {"ppo_kl": -negative_approx_kl}
    # use negative log probs as an estimator of entropy loss
    metrics["entropy_loss"] = average_loss(-log_probs, response_mask, mode=loss_avg_mode)

    if loss_type == "cispo":
        final_pg_loss = -advantages * log_probs * clipped_ratio.detach()
    elif loss_type == "sapo":
        positive_token_mask =  (advantages >= 0).float()
        negative_token_mask =  (advantages < 0).float()
        gate_negative = 4.0 / tau_negative * torch.sigmoid(tau_negative * (ratio - 1.0))
        gate_positive = 4.0 / tau_positive * torch.sigmoid(tau_positive * (ratio - 1.0))
        final_pg_loss = -advantages * (positive_token_mask * gate_positive + negative_token_mask * gate_negative)
    else:
        pg_loss = -advantages * ratio  # -ratio * A
        pg_loss2 = -advantages * clipped_ratio  # -clip(ratio, 1-clip_low, 1+clip_high) * A
        pg_loss3 = -advantages * clip_ratio_dual  # -clip_dual * A

        clipped_pg_loss_higher = torch.max(pg_loss, pg_loss2)  # clip if pg_loss < pg_loss2
        metrics["pg_clipfrac_higher"] = (pg_loss < pg_loss2).float()
        clipped_pg_loss_lower = torch.min(clipped_pg_loss_higher, pg_loss3)  # clip if pg_loss > pg_loss3 and adv < 0
        final_pg_loss = torch.where(advantages < 0, clipped_pg_loss_lower, clipped_pg_loss_higher)
        metrics["pg_clipfrac_lower"] = (clipped_pg_loss_higher > pg_loss3).float() * (advantages < 0).float()

    final_pg_loss = average_loss(final_pg_loss, response_mask, mode=loss_avg_mode)
    metrics = {k: VF.masked_mean(v, response_mask).detach().item() for k, v in metrics.items()}
    return final_pg_loss, metrics


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_avg_mode: Literal["token", "seq"],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the value loss.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L556

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange_value: (float)
            The clip range for value net used in PPO. See https://arxiv.org/abs/1707.06347
        loss_avg_mode: (Literal["token", "seq"])
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped
        vpred_mean: a float
            The mean of predicted values

    """
    vpredclipped = torch.clamp(vpreds, values - cliprange_value, values + cliprange_value)
    vf_loss1 = torch.square(vpreds - returns)
    vf_loss2 = torch.square(vpredclipped - returns)
    clipped_vf_losses = torch.max(vf_loss1, vf_loss2)  # clip if vf_loss1 < vf_loss2
    vf_loss = 0.5 * average_loss(clipped_vf_losses, response_mask, mode=loss_avg_mode)
    metrics = {
        "vf_clipfrac": VF.masked_mean((vf_loss1 < vf_loss2).float(), response_mask).detach().item(),
        "vpred_mean": VF.masked_mean(vpreds, response_mask).detach().item(),
    }
    return vf_loss, metrics


def compute_kl(
    log_probs: torch.FloatTensor,
    ref_log_probs: torch.FloatTensor,
    kl_penalty: Literal["kl", "abs", "mse", "low_var_kl", "full"],
) -> torch.Tensor:
    """Compute KL divergence given log_probs and ref_log_probs.

    Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L1150

    Args:
        log_probs: torch.Tensor
        ref_log_probs: torch.Tensor
        kl_penalty: str ("kl", "abs", "mse", "low_var_kl", "full")

    Returns:
        kl_div: torch.Tensor

    """
    log_probs, ref_log_probs = log_probs.float(), ref_log_probs.float()
    if kl_penalty == "kl":
        return log_probs - ref_log_probs

    if kl_penalty == "abs":
        return (log_probs - ref_log_probs).abs()

    if kl_penalty == "mse":
        return 0.5 * (log_probs - ref_log_probs).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # URL http://joschu.net/blog/kl-approx.html
    if kl_penalty == "low_var_kl":
        # For numerical stability
        kl = (ref_log_probs - log_probs).clamp(-20.0, 20.0)
        kld = (kl.exp() - kl - 1).contiguous()
        return torch.clamp(kld, min=-10.0, max=10.0)

    if kl_penalty == "full":
        return F.kl_div(ref_log_probs, log_probs, log_target=True, reduction="none").sum(-1)

    raise NotImplementedError(f"Unknown KL penalty: {kl_penalty}.")


@torch.no_grad()
def compute_token_entropy(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-token entropy directly from log probabilities of generated tokens.
    H = -P * log(P)
    """
    probs = torch.exp(log_probs)
    entropy = -probs * log_probs
    entropy = entropy * response_mask
    return entropy


@torch.no_grad()
def compute_entropy_advantage_shaping(
    advantages: torch.Tensor,
    entropy: torch.Tensor,
    response_mask: torch.Tensor,
    alpha: float = 0.4,
    kappa: float = 2.0,
) -> torch.Tensor:
    """
    Compute entropy-based advantage shaping term:
    psi(H_t) = min(alpha * H_t^detach, |A_t| / kappa)
    """
    entropy_detached = entropy.detach()
    entropy_term = alpha * entropy_detached
    advantage_magnitude_term = torch.abs(advantages) / kappa
    psi = torch.min(entropy_term, advantage_magnitude_term)
    psi = psi * response_mask
    return psi


def tokenize_response(response: str, exclude_final_prediction: bool = True) -> List[str]:
    """Tokenize response into words for n-gram analysis."""
    if exclude_final_prediction:
        response = re.sub(r'\\boxed\{[^}]*\}$', '', response.strip())
    text = re.sub(r"<think>|</think>|\\boxed\{[^}]*\}", " ", response)
    text = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = text.split()
    return [token for token in tokens if token.strip()]


def generate_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Generate n-grams from a list of tokens."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def compute_ngram_diversity_reward(responses: List[str], n: int = 2, exclude_final_prediction: bool = True) -> float:
    """Calculate N-gram diversity reward across all responses."""
    if not responses:
        return 0.0
    if len(set(responses)) == 1:
        return 0.0
    all_ngrams = []
    for response in responses:
        tokens = tokenize_response(response, exclude_final_prediction=exclude_final_prediction)
        ngrams = generate_ngrams(tokens, n)
        all_ngrams.extend(ngrams)
    if not all_ngrams:
        return 0.0
    unique_ngrams = len(set(all_ngrams))
    total_ngrams = len(all_ngrams)
    return unique_ngrams / total_ngrams


def compute_prediction_consistency_reward(responses: List[str], ground_truths: List[str]) -> List[float]:
    """Calculate prediction consistency reward for each response."""
    if len(responses) <= 1:
        return [0.0] * len(responses)
    answers = []
    for response in responses:
        answer = extract_boxed_content(response)
        answers.append(answer)
    consistency_rewards = []
    K = len(responses)
    for i, answer_i in enumerate(answers):
        different_count = sum(1 for j, answer_j in enumerate(answers) if j != i and answer_j != answer_i)
        consistency_reward = different_count / (K - 1) if K > 1 else 0.0
        consistency_rewards.append(consistency_reward)
    return consistency_rewards


def compute_grpo_enhanced_rewards(
    token_level_rewards: torch.Tensor,
    response_texts: List[str],
    ground_truths: List[str],
    index,
    beta: float = 0.1,
    gamma: float = 0.1,
    use_ngram_diversity: bool = True,
    use_prediction_consistency: bool = True,
    ngram_size: int = 2,
    correctness_threshold: float = 0.5,
) -> torch.Tensor:
    """Compute enhanced rewards (task rewards + exploration bonuses) for GRPO training."""
    task_rewards = token_level_rewards.sum(dim=-1)
    id2responses = defaultdict(list)
    id2ground_truths = defaultdict(list)
    id2indices = defaultdict(list)
    id2task_rewards = defaultdict(list)
    bsz = task_rewards.shape[0]
    for i in range(bsz):
        problem_id = index[i].item() if hasattr(index[i], 'item') else index[i]
        id2responses[problem_id].append(response_texts[i])
        id2ground_truths[problem_id].append(ground_truths[i])
        id2indices[problem_id].append(i)
        id2task_rewards[problem_id].append(task_rewards[i].item())

    exploration_bonuses = torch.zeros_like(task_rewards)
    for problem_id in id2responses:
        responses = id2responses[problem_id]
        group_ground_truths = id2ground_truths[problem_id]
        indices = id2indices[problem_id]
        group_task_rewards = id2task_rewards[problem_id]
        if len(responses) <= 1:
            continue

        correct_indices = []
        for j, idx in enumerate(indices):
            if group_task_rewards[j] > correctness_threshold:
                correct_indices.append(idx)

        ngram_diversity_reward = 0.0
        if use_ngram_diversity and len(responses) > 1 and len(set(responses)) > 1:
            ngram_diversity_reward = compute_ngram_diversity_reward(responses, n=ngram_size, exclude_final_prediction=True)

        consistency_rewards = [0.0] * len(responses)
        if use_prediction_consistency and len(responses) > 1:
            consistency_rewards = compute_prediction_consistency_reward(responses, group_ground_truths)

        for correct_idx in correct_indices:
            exploration_bonus = 0.0
            if use_ngram_diversity and len(responses) > 1:
                exploration_bonus += beta * ngram_diversity_reward
            if use_prediction_consistency and len(responses) > 1:
                original_position = indices.index(correct_idx)
                exploration_bonus += gamma * consistency_rewards[original_position]
            exploration_bonuses[correct_idx] = exploration_bonus

    enhanced_rewards = task_rewards + exploration_bonuses
    return enhanced_rewards


