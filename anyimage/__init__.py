"""BioImageViewer - Interactive image viewer widget with annotation tools."""

from .viewer import BioImageViewer
from .utils import (
    MASK_COLORS,
    CHANNEL_COLORS,
    normalize_image,
    array_to_base64,
    labels_to_rgba,
    hex_to_rgb,
    composite_channels,
)

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
]
