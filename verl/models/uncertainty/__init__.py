"""
Uncertainty estimation modules for visual and other modalities.
"""

from .visual_uncertainty import (
    BaseVisualUncertainty,
    PixelLevelUncertainty,
    DualPathUncertainty,
    create_pixel_level_uncertainty_estimator,
    create_dual_path_uncertainty_estimator,
    # ImageAugmentationUncertainty,  # Legacy alias
    # create_visual_uncertainty_estimator,  # Legacy alias
)

__all__ = [
    "BaseVisualUncertainty",
    "PixelLevelUncertainty",
    "DualPathUncertainty",
    "create_pixel_level_uncertainty_estimator",
    "create_dual_path_uncertainty_estimator",
    # "ImageAugmentationUncertainty",  # Legacy alias
    # "create_visual_uncertainty_estimator",  # Legacy alias
]