"""BioImageViewer - Interactive image viewer widget with annotation tools."""

from .profiling import Profiler, profile_operation, timer
from .utils import (
    CHANNEL_COLORS,
    MASK_COLORS,
    array_to_base64,
    composite_channels,
    hex_to_rgb,
    labels_to_rgba,
    normalize_image,
)
from .viewer import BioImageViewer

__version__ = "0.1.0"
__all__ = [
    "BioImageViewer",
    "MASK_COLORS",
    "CHANNEL_COLORS",
    "normalize_image",
    "array_to_base64",
    "labels_to_rgba",
    "hex_to_rgb",
    "composite_channels",
    "Profiler",
    "profile_operation",
    "timer",
]
