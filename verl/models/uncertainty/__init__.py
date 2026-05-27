"""
Uncertainty estimation modules for visual and other modalities.
"""

from .perceptual_uncertainty import (
    BasePerceptualUncertainty,
    PixelLevelUncertainty,
    DualPathUncertainty,
    create_pixel_level_uncertainty_estimator,
    create_dual_path_uncertainty_estimator,
    # ImageAugmentationUncertainty,  # Legacy alias
    # create_perceptual_uncertainty_estimator,  # Legacy alias
)

__all__ = [
    "BasePerceptualUncertainty",
    "PixelLevelUncertainty",
    "DualPathUncertainty",
    "create_pixel_level_uncertainty_estimator",
    "create_dual_path_uncertainty_estimator",
    # "ImageAugmentationUncertainty",  # Legacy alias
    # "create_perceptual_uncertainty_estimator",  # Legacy alias
]