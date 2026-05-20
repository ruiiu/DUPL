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
PPO config
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple

from ..utils.py_functional import get_abs_path
from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    image_dir: Optional[str] = None
    video_fps: float = 2.0
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16

    def post_init(self):
        self.image_dir = get_abs_path(self.image_dir, prompt="Image directory")
        self.format_prompt = get_abs_path(self.format_prompt, prompt="Format prompt file")
        self.override_chat_template = get_abs_path(self.override_chat_template, prompt="Chat template file")


@dataclass
class ExplorationConfig:
    """Configuration for exploration bonuses in GRPO training"""
    use_exploration: bool = False
    """whether to enable exploration bonuses (automatically sets adv_estimator to grpo_exploration)"""
    beta: float = 0.1
    """scaling factor for n-gram diversity exploration reward"""
    gamma: float = 0.1
    """scaling factor for prediction consistency exploration reward"""
    use_ngram_diversity: bool = True
    """whether to include n-gram diversity reward"""
    use_prediction_consistency: bool = True
    """whether to include prediction consistency reward"""
    ngram_size: int = 2
    """size of n-grams for diversity calculation (2=bigrams, 3=trigrams, etc.)"""


@dataclass
class DuplConfig:
    """Configuration for DUPL (dual-path uncertainty learning)"""
    enabled: bool = False
    """whether to enable DUPL"""

    augmentation_strength: float = 0.1
    """strength of image augmentations (0.0 to 1.0)"""
    gaussian_noise_std: float = 0.1
    """standard deviation for Gaussian noise"""
    uncertainty_alpha: float = 0.3
    """scaling factor for uncertainty bonus in advantage shaping"""
    uncertainty_kappa: float = 2.5
    """denominator for advantage magnitude normalization"""
    sampling_strategy: str = "fixed"
    """sampling strategy: 'fixed' (constant probability) or 'adaptive' (curriculum learning)"""
    fixed_prob: float = 0.5
    """fixed sampling probability"""
    initial_aug_prob: float = 1.0
    """initial probability of sampling from augmented branch"""
    final_aug_prob: float = 0.0
    """final probability of sampling from augmented branch"""

    kl_penalty_weight: float = 1.0
    """weight for KL divergence penalty"""
    top_k_for_kl: int = 100
    """number of top-k probabilities for KL computation"""
    exploration_ratio: float = 0.45
    """ratio of total training steps to use for exploration phase (add KL penalty)"""
    transition_ratio: float = 0.25
    """ratio of total training steps to use for gradual transition"""
    enable_kl_transition: bool = False
    """if True, KL multiplier transitions from +1 to -1; if False, stays at +1"""
    use_kl_penalty: bool = True
    """if True, compute and apply KL divergence penalty"""
    use_forward_kl_only: bool = False
    """if True, use only forward KL for exploration"""


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    """discount factor for ppo gae advantage estimator"""
    lam: float = 1.0
    """lambda value for ppo gae advantage estimator"""
    adv_estimator: str = "grpo"
    """advantage estimator, support `gae`, `grpo`, `grpo_exploration`, `grpo_dupl`, `reinforce_plus_plus`, `remax`, `rloo`"""
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    """configuration for exploration bonuses (used with grpo_exploration)"""
    dupl: DuplConfig = field(default_factory=DuplConfig)
    """configuration for DUPL dual-path uncertainty learning (used with grpo_dupl)"""
    disable_kl: bool = False
    """disable reference model"""
    use_kl_loss: bool = False
    """use kl loss instead of kl in reward"""
    kl_penalty: str = "kl"
    """kl penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
    kl_coef: float = 1e-3
    """kl coefficient"""
    kl_type: str = "fixed"
    """kl controller type, support `fixed`, `adaptive`"""
    kl_horizon: float = 10000.0
    """kl horizon for adaptive kl controller"""
    kl_target: float = 0.1
    """target kl for adaptive kl controller"""
    online_filtering: bool = False
    """use online filtering"""
    filter_key: str = "overall"
    """reward key for filtering samples"""
    filter_low: float = 0.01
    """filter out low reward samples if online filtering"""
    filter_high: float = 0.99
    """filter out high reward samples if online filtering"""
    use_entropy_shaping: bool = False
    """enable entropy-based advantage shaping for GRPO"""
    entropy_alpha: float = 0.4
    """scaling factor for entropy term in advantage shaping"""
    entropy_kappa: float = 2.0
    """denominator for advantage magnitude term in advantage shaping"""

    def post_init(self):
        """Post-initialization to automatically configure advantage estimator based on exploration and DUPL settings"""
        if self.exploration.use_exploration:
            if self.adv_estimator == "grpo":
                self.adv_estimator = "grpo_exploration"
                print(f"INFO: Automatically set adv_estimator to 'grpo_exploration' because exploration is enabled")
            elif self.adv_estimator not in ["grpo_exploration"]:
                print(f"WARNING: Exploration bonuses are enabled but adv_estimator is '{self.adv_estimator}'.")
        else:
            if self.adv_estimator == "grpo_exploration":
                self.adv_estimator = "grpo"
                print(f"INFO: Automatically set adv_estimator to 'grpo' because exploration is disabled")

        if self.dupl.enabled:
            if self.adv_estimator == "grpo":
                self.adv_estimator = "grpo_dupl"
                print(f"INFO: Automatically set adv_estimator to 'grpo_dupl' because DUPL is enabled")
        else:
            if self.adv_estimator == "grpo_dupl":
                self.adv_estimator = "grpo"
                print(f"INFO: Automatically set adv_estimator to 'grpo' because DUPL is disabled")


@dataclass
class TrainerConfig:
    total_epochs: Optional[int] = None
    """total epochs for training"""
    max_steps: int = 200
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "DUPL"
    """project name for logger"""
    experiment_name: str = "demo"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "wandb")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    nnodes: int = 1
    """number of nodes for training"""
    n_gpus_per_node: int = 8
    """number of gpus per node for training"""
    max_try_make_batch: int = 20
    """max number of generations for online filtering, -1 means no limit"""
    critic_warmup: int = 0
    """critic warmup steps"""
    val_freq: int = -1
    """validation frequency, -1 means no validation"""
    val_before_train: bool = True
    """validate before training"""
    val_only: bool = False
    """validate only, skip training"""
    val_generations_to_log: int = 0
    """number of generations to log for validation"""
    save_freq: int = -1
    """save frequency, -1 means no saving"""
    save_limit: int = -1
    """max number of checkpoints to save, -1 means no limit"""
    save_model_only: bool = False
    """save model only, no optimizer state dict"""
    save_checkpoint_path: Optional[str] = None
    """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
    load_checkpoint_path: Optional[str] = None
    """load checkpoint path"""
    ray_timeline: Optional[str] = None
    """file to save ray timeline"""
    find_last_checkpoint: bool = True
    """automatically find the last checkpoint in the save checkpoint path to resume training"""

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # may be not exist
        self.load_checkpoint_path = get_abs_path(self.load_checkpoint_path, prompt="Model checkpoint")


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
