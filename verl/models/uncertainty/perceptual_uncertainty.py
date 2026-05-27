#!/usr/bin/env python3
"""
Perceptual uncertainty estimation approaches for GRPO training.

This module provides two approaches for perceptual uncertainty estimation:
1. PixelLevelUncertainty: Image augmentation-based uncertainty through variance (based on image_augmentation.py)
2. DualPathUncertainty: Dual-path processing with KL divergence penalty

Both approaches share common image augmentation functionality and are designed
to work with the existing GRPO training pipeline without architectural changes.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import math


class BasePerceptualUncertainty:
    """
    Base class for perceptual uncertainty estimation with shared augmentation functionality.
    """
    
    def __init__(
        self,
        augmentation_strength: float = 0.1,
        gaussian_noise_std: float = 0.1,
        sampling_strategy: str = "fixed",
        fixed_prob: float = 0.5,
        total_training_steps: int = 200,
        initial_aug_prob: float = 1.0,
        final_aug_prob: float = 0.0,
    ):
        """
        Initialize base perceptual uncertainty estimator.
        
        Args:
            augmentation_strength: Strength of image augmentations (0.0 to 1.0)
            gaussian_noise_std: Standard deviation for Gaussian noise
        """
        self.augmentation_strength = augmentation_strength
        self.gaussian_noise_std = gaussian_noise_std
        self.sampling_strategy = sampling_strategy
        self.fixed_prob = fixed_prob
        self.total_training_steps = total_training_steps
        self.initial_aug_prob = initial_aug_prob
        self.final_aug_prob = final_aug_prob
        
        # Define augmentation transforms optimized for mathematical content
        # For math problems with geometry lines, we use gentler augmentations
        self.augmentation_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=10),
            # Very light brightness/contrast changes (preserve text readability)
            transforms.ColorJitter(
                brightness=augmentation_strength * 0.15,  # Reduced from 0.2
                contrast=augmentation_strength * 0.15,    # Reduced from 0.2
                saturation=0.0,  # No saturation changes (preserve line colors)
                hue=0.0          # No hue changes (preserve diagram colors)
            ),
        ])
    
    def add_gaussian_noise(self, image: Image.Image, noise_std: float = None) -> Image.Image:
        """
        Add Gaussian noise to an image.
        
        Args:
            image: PIL Image
            noise_std: Standard deviation of Gaussian noise (uses self.gaussian_noise_std if None)
            
        Returns:
            PIL Image with added noise
        """
        if noise_std is None:
            noise_std = self.gaussian_noise_std * self.augmentation_strength
            
        # Convert PIL to numpy array
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_std, img_array.shape)
        noisy_img_array = img_array + noise
        
        # Clip values to valid range [0, 1]
        noisy_img_array = np.clip(noisy_img_array, 0, 1)
        
        # Convert back to PIL Image
        noisy_img_array = (noisy_img_array * 255).astype(np.uint8)
        return Image.fromarray(noisy_img_array)
    
    def create_augmented_image(self, image: Image.Image) -> Image.Image:
        """
        Create a consistently augmented version of an image.
        Always applies the same transformation (augmentation + Gaussian noise).
        
        Args:
            image: PIL Image
            
        Returns:
            PIL Image (always augmented with same transformation)
        """
        # Always apply augmentation
        augmented_image = self.augmentation_transforms(image)
        # Add Gaussian noise
        augmented_image = self.add_gaussian_noise(augmented_image)
        return augmented_image


class PixelLevelUncertainty(BasePerceptualUncertainty):
    """
    Pixel-level perceptual uncertainty estimator using image augmentation.

    This class estimates perceptual uncertainty by:
    1. Creating multiple augmented versions of input images
    2. Computing vision encoder embeddings for each version
    3. Calculating variance across embeddings as uncertainty measure
    4. Providing controlled noise injection based on uncertainty
    """

    def __init__(
        self,
        n_samples: int = 5,
        augmentation_strength: float = 0.1,
        noise_scaling_factor: float = 0.05,
        min_uncertainty: float = 0.001,
        max_uncertainty: float = 0.5,
        gaussian_noise_std: float = 0.1,
        total_training_steps: int = 200,
        sampling_strategy: str = "fixed",
        fixed_prob: float = 0.5,
        initial_aug_prob: float = 1.0,
        final_aug_prob: float = 0.0,
    ):
        """
        Initialize the pixel-level uncertainty estimator.

        Args:
            n_samples: Number of augmented samples to generate
            augmentation_strength: Strength of image augmentations (0.0 to 1.0)
            noise_scaling_factor: Factor to convert uncertainty variance to noise std
            min_uncertainty: Minimum uncertainty value for meaningful training impact
            max_uncertainty: Maximum uncertainty value
            gaussian_noise_std: Standard deviation for Gaussian noise
            fixed_prob: Fixed probability for sampling strategy
            initial_aug_prob: Initial augmentation probability for adaptive sampling
            final_aug_prob: Final augmentation probability for adaptive sampling
            sampling_strategy: Sampling strategy ('fixed' or 'adaptive')
        """
        super().__init__(augmentation_strength, gaussian_noise_std)
        self.n_samples = n_samples
        self.noise_scaling_factor = noise_scaling_factor
        self.min_uncertainty = min_uncertainty
        self.max_uncertainty = max_uncertainty
        self.total_training_steps = total_training_steps
        self.sampling_strategy = sampling_strategy
        self.fixed_prob = fixed_prob
        self.initial_aug_prob = initial_aug_prob
        self.final_aug_prob = final_aug_prob
    
    def sample_image_for_generation(self, augmented_images: List[Image.Image], current_step: int = 0) -> Image.Image:
        """
        Sample an image from the augmented set for downstream token generation.

        Args:
            augmented_images: List of images [raw_image, augmented_1, augmented_2, ...]
            current_step: Current training step (used for adaptive sampling)

        Returns:
            PIL Image (either raw or one of the augmented images based on sampling strategy)
        """
        if len(augmented_images) == 0:
            raise ValueError("augmented_images list cannot be empty")

        # Determine sampling probability based on strategy
        if self.sampling_strategy == "fixed":
            # Fixed probability sampling
            sample_prob = self.fixed_prob
        elif self.sampling_strategy == "adaptive":
            # Adaptive sampling: start with high augmentation, decay over training
            if self.total_training_steps > 0:
                progress = min(current_step / self.total_training_steps, 1.0)
                # Linear decay from initial_aug_prob to final_aug_prob
                sample_prob = self.initial_aug_prob + progress * (self.final_aug_prob - self.initial_aug_prob)
            else:
                sample_prob = self.fixed_prob
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

        # Decide whether to use raw image (index 0) or augmented image
        if np.random.random() < sample_prob:
            # Use raw image (first in the list)
            return augmented_images[0]
        else:
            # Use one of the augmented images (randomly select from indices 1 to n_samples) 
            if len(augmented_images) > 1:
                augmented_idx = np.random.randint(1, len(augmented_images))
                return augmented_images[augmented_idx]
            else:
                # Fallback to raw image if no augmented images available
                return augmented_images[0]
    
    def augment_images(self, images: List[Image.Image]) -> List[List[Image.Image]]:
        """
        Create augmented versions of input images for uncertainty estimation.
        Creates 1 raw image + n_samples augmented images for each input image.
        
        Args:
            images: List of PIL Images
            
        Returns:
            List of lists, where each inner list contains [raw_image, augmented_1, augmented_2, ...]
        """
        augmented_batches = []
        
        for image in images:
            # Start with original image (raw)
            augmented_samples = [image]
            
            # Add n_samples augmented versions for uncertainty estimation
            for _ in range(self.n_samples):
                augmented_image = self.create_augmented_image(image)
                augmented_samples.append(augmented_image)
            
            augmented_batches.append(augmented_samples)
        
        return augmented_batches
    
    def estimate_perceptual_uncertainty(
        self,
        vision_encoder: torch.nn.Module,
        processor,
        images: List[Image.Image],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Estimate perceptual uncertainty using image augmentation.
        
        Args:
            vision_encoder: Vision encoder model (e.g., Qwen2.5-VL vision tower)
            processor: Image processor
            images: List of PIL Images
            device: Device to run computation on
            
        Returns:
            torch.Tensor: Perceptual uncertainty values, shape [batch_size]
        """
        uncertainties = []
        
        # Process each image in the batch
        for image in images:
            # Create augmented versions (1 raw + n_samples augmented)
            augmented_samples = self.augment_images([image])[0]
            
            # Use all samples for uncertainty estimation (raw + all augmented)
            selected_samples = augmented_samples
            
            # Process all samples through vision encoder
            embeddings = []
            for sample_image in selected_samples:
                # Process single image using image_processor (not full processor)
                inputs = processor.image_processor(
                    images=sample_image,
                    return_tensors="pt"
                )
                
                # Move to device
                pixel_values = inputs["pixel_values"].to(device)
                image_grid_thw = inputs.get("image_grid_thw")
                if image_grid_thw is not None:
                    image_grid_thw = image_grid_thw.to(device)
                
                # Get vision encoder embedding
                with torch.no_grad():
                    try:
                        # Direct vision encoder call (should work with FSDP)
                        # Call the vision encoder directly with the required parameters
                        if image_grid_thw is not None:
                            embedding = vision_encoder(pixel_values, grid_thw=image_grid_thw)
                        else:
                            embedding = vision_encoder(pixel_values)
                        
                    except Exception:
                        raise
                    
                    # Pool to get single vector per image
                    if embedding.dim() > 2:
                        embedding = embedding.mean(dim=1)  # Average over sequence dimension
                    
                    embeddings.append(embedding.cpu())
            
            # Stack embeddings and compute variance
            embeddings_tensor = torch.stack(embeddings, dim=0)  # [n_samples, embed_dim]
            variance = embeddings_tensor.var(dim=0).mean()  # Average variance across dimensions
            uncertainties.append(variance.item())
        
        return torch.tensor(uncertainties, device=device)
    
    def apply_noise_to_logits(
        self,
        logits: torch.Tensor,
        perceptual_uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply controlled Gaussian noise to logits based on perceptual uncertainty.
        
        Args:
            logits: Token logits, shape [batch_size, seq_len, vocab_size]
            perceptual_uncertainty: Perceptual uncertainty values, shape [batch_size]
            
        Returns:
            torch.Tensor: Noisy logits with same shape as input
        """
        batch_size = logits.shape[0]
        
        # Convert uncertainty to noise standard deviation
        noise_std = torch.clamp(
            perceptual_uncertainty * self.noise_scaling_factor,
            min=self.min_uncertainty,
            max=self.max_uncertainty
        )
        
        # Apply noise to each sample in the batch
        noisy_logits = logits.clone()
        for i in range(batch_size):
            if noise_std[i] > 0:
                noise = torch.randn_like(logits[i]) * noise_std[i]
                noisy_logits[i] = logits[i] + noise
        
        return noisy_logits


class DualPathUncertainty(BasePerceptualUncertainty):
    """
    Dual-path perceptual uncertainty estimator using KL divergence penalty.

    This class implements a curriculum learning approach with:
    1. Dual-path processing (raw + augmented images)
    2. KL divergence penalty between branches
    3. Adaptive sampling strategy with probability decay
    4. Advantage reshaping based on KL divergence
    """

    def __init__(
        self,
        kl_penalty_weight: float = 0.01,
        top_k_for_kl: int = 100,
        augmentation_strength: float = 1.0,
        total_training_steps: int = 100,
        gaussian_noise_std: float = 0.02,
        sampling_strategy: str = "adaptive",
        fixed_prob: float = 0.5,
        initial_aug_prob: float = 1.0,
        final_aug_prob: float = 0.0,
        use_forward_kl_only: bool = False,
    ):
        """
        Initialize the dual-path uncertainty estimator.

        Args:
            initial_aug_prob: Initial probability of sampling from augmented branch
            final_aug_prob: Final probability of sampling from augmented branch
            kl_penalty_weight: Weight for KL divergence penalty
            top_k_for_kl: Number of top-k probabilities for KL computation
            augmentation_strength: Strength multiplier for augmentations
            total_training_steps: Total training steps for probability decay
            gaussian_noise_std: Standard deviation for Gaussian noise
            sampling_strategy: Sampling strategy ('fixed' or 'adaptive')
        """
        super().__init__(augmentation_strength, gaussian_noise_std)
        self.initial_aug_prob = initial_aug_prob
        self.final_aug_prob = final_aug_prob
        self.kl_penalty_weight = kl_penalty_weight
        self.top_k_for_kl = top_k_for_kl
        self.total_training_steps = total_training_steps    
        self.current_step = 0
        self.sampling_strategy = sampling_strategy
        self.fixed_prob = fixed_prob
        self.use_forward_kl_only = use_forward_kl_only

    def get_current_aug_probability(self) -> float:
        """
        Get current augmentation sampling probability based on training progress and strategy.

        Returns:
            float: Current probability of sampling from augmented branch
        """
        if self.sampling_strategy == "fixed":
            # Fixed probability (use initial_aug_prob as the fixed value for dual-path)
            return self.fixed_prob
        elif self.sampling_strategy == "adaptive":
            # Adaptive curriculum learning with linear decay
            if self.total_training_steps <= 0:
                return self.final_aug_prob

            progress = min(self.current_step / self.total_training_steps, 1.0)
            current_prob = self.initial_aug_prob + progress * (self.final_aug_prob - self.initial_aug_prob)
            return max(current_prob, 0.0)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

    def update_training_step(self, step: int):
        """Update current training step for probability decay."""
        self.current_step = step

    def compute_kl_divergence_penalty(
        self,
        raw_log_probs: torch.Tensor,
        aug_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence penalty between raw and augmented branches.

        Args:
            raw_log_probs: Log probabilities from raw image branch, shape [batch_size, seq_len]
            aug_log_probs: Log probabilities from augmented image branch, shape [batch_size, seq_len]
            response_mask: Response mask, shape [batch_size, seq_len]

        Returns:
            torch.Tensor: KL divergence penalty, shape [batch_size, seq_len]
        """
        
        # Convert log probabilities to probabilities for KL divergence computation
        raw_probs = torch.exp(raw_log_probs)
        aug_probs = torch.exp(aug_log_probs)
        
        # Forward KL: KL(raw || aug)
        forward_kl = F.kl_div(
            torch.log(aug_probs + 1e-8),  # target (augmented)
            raw_probs,  # input (raw)
            reduction='none'
        )  # (batch_size, seq_len)

        if self.use_forward_kl_only:
            # Use only forward KL for exploration
            kl_penalty = forward_kl
        else:
            # Use symmetric KL (mean of forward and inverse)
            # Inverse KL: KL(aug || raw)
            inverse_kl = F.kl_div(
                torch.log(raw_probs + 1e-8),  # target (raw)
                aug_probs,  # input (augmented)
                reduction='none'
            )  # (batch_size, seq_len)
            
            # Take mean of bidirectional KL
            kl_penalty = (forward_kl + inverse_kl) / 2.0

        # Apply response mask
        kl_penalty = kl_penalty * response_mask

        return kl_penalty

    def sample_branch_for_training(self) -> bool:
        """
        Sample which branch to use for training based on current probability.

        Returns:
            bool: True if augmented branch should be used, False for raw branch
        """
        current_prob = self.get_current_aug_probability()
        return torch.rand(1).item() < current_prob

    def compute_dual_path_advantages(
        self,
        base_advantages: torch.Tensor,
        raw_log_probs: torch.Tensor,
        aug_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        use_augmented_branch: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages for dual-path training with KL penalty.

        Args:
            base_advantages: Base advantages from reward model
            raw_log_probs: Log probabilities from raw image branch
            aug_log_probs: Log probabilities from augmented image branch
            response_mask: Response mask
            use_augmented_branch: Whether to apply KL penalty (for augmented branch)

        Returns:
            Tuple of (final_advantages, kl_penalty)
        """
        # Compute KL divergence penalty
        kl_penalty = self.compute_kl_divergence_penalty(
            raw_log_probs=raw_log_probs,
            aug_log_probs=aug_log_probs,
            response_mask=response_mask,
        )

        if use_augmented_branch:
            # For augmented branch: subtract KL penalty to prevent divergence
            final_advantages = base_advantages + self.kl_penalty_weight * kl_penalty
        else:
            # For raw branch: use base advantages without penalty
            final_advantages = base_advantages

        return final_advantages, kl_penalty


def create_pixel_level_uncertainty_estimator(config: Dict) -> PixelLevelUncertainty:
    """
    Create pixel-level uncertainty estimator from configuration.

    Args:
        config: Configuration dictionary with uncertainty parameters

    Returns:
        PixelLevelUncertainty: Configured uncertainty estimator
    """
    return PixelLevelUncertainty(
        n_samples=config.get("n_samples", 5),
        augmentation_strength=config.get("augmentation_strength", 0.1),
        noise_scaling_factor=config.get("noise_scaling_factor", 0.05),
        min_uncertainty=config.get("min_uncertainty", 0.001),
        max_uncertainty=config.get("max_uncertainty", 0.5),
        gaussian_noise_std=config.get("gaussian_noise_std", 0.1),
        total_training_steps=config.get("total_training_steps", 200),
        sampling_strategy=config.get("sampling_strategy", "fixed"),
        fixed_prob=config.get("fixed_prob", 0.5),
        initial_aug_prob=config.get("initial_aug_prob", 1.0),
        final_aug_prob=config.get("final_aug_prob", 0.0),
    )


def create_dual_path_uncertainty_estimator(config: Dict) -> DualPathUncertainty:
    """
    Create dual-path uncertainty estimator from configuration.

    Args:
        config: Configuration dictionary with dual-path parameters

    Returns:
        DualPathUncertainty: Configured dual-path estimator
    """
    return DualPathUncertainty(
        initial_aug_prob=config.get("initial_aug_prob", 1.0),
        final_aug_prob=config.get("final_aug_prob", 0.0),
        kl_penalty_weight=config.get("kl_penalty_weight", 0.01),
        top_k_for_kl=config.get("top_k_for_kl", 100),
        augmentation_strength=config.get("augmentation_strength", 1.0),
        total_training_steps=config.get("total_training_steps", 200),
        gaussian_noise_std=config.get("gaussian_noise_std", 0.02),
        sampling_strategy=config.get("sampling_strategy", "adaptive"),
        fixed_prob=config.get("fixed_prob", 0.5),
        use_forward_kl_only=config.get("use_forward_kl_only", False),
    )
