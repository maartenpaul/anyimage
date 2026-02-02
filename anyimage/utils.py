"""Utility functions and constants for anyimage."""

import base64
from io import BytesIO

import numpy as np
from PIL import Image


# Default colors for mask layers
MASK_COLORS = [
    "#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#ffeaa7",
    "#dfe6e9", "#fd79a8", "#a29bfe", "#6c5ce7", "#00b894"
]

# Default colors for channels (common microscopy LUTs)
CHANNEL_COLORS = [
    "#00ff00",  # Green (GFP)
    "#ff0000",  # Red (RFP/mCherry)
    "#0000ff",  # Blue (DAPI)
    "#ff00ff",  # Magenta
    "#00ffff",  # Cyan
    "#ffff00",  # Yellow
    "#ff8000",  # Orange
    "#ffffff",  # White/Gray
]


def _hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple (0-255).

    Internal helper function.
    """
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def normalize_image(
    data: np.ndarray,
    global_min: float | None = None,
    global_max: float | None = None,
) -> np.ndarray:
    """Normalize image data to uint8 range.

    Args:
        data: Input array to normalize
        global_min: Optional global minimum value for consistent normalization
        global_max: Optional global maximum value for consistent normalization

    Returns:
        Normalized uint8 array
    """
    if data.dtype == np.uint8 and global_min is None and global_max is None:
        return data
    data = data.astype(np.float64)
    if global_min is not None and global_max is not None:
        data_min, data_max = global_min, global_max
    else:
        data_min, data_max = data.min(), data.max()
    if data_max > data_min:
        data = (data - data_min) / (data_max - data_min) * 255
    else:
        data = np.zeros_like(data)
    return np.clip(data, 0, 255).astype(np.uint8)


def array_to_base64(data: np.ndarray) -> str:
    """Convert numpy array to base64 encoded PNG.

    Args:
        data: Input array (2D grayscale, 3D RGB, or 3D RGBA)

    Returns:
        Base64 encoded PNG string

    Raises:
        ValueError: If array shape is not supported
    """
    if data.ndim == 2:
        img = Image.fromarray(data, mode="L")
    elif data.ndim == 3 and data.shape[2] == 3:
        img = Image.fromarray(data, mode="RGB")
    elif data.ndim == 3 and data.shape[2] == 4:
        img = Image.fromarray(data, mode="RGBA")
    else:
        raise ValueError(f"Unsupported array shape: {data.shape}")

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def labels_to_rgba(
    labels: np.ndarray, contours_only: bool = False, contour_width: int = 1
) -> np.ndarray:
    """Convert label array to RGBA with unique colors per label.

    Args:
        labels: 2D array of integer labels (0 is background)
        contours_only: If True, only draw contours instead of filled regions
        contour_width: Width of contours in pixels (only used if contours_only=True)

    Returns:
        RGBA image as uint8 array (H, W, 4)
    """
    height, width = labels.shape[:2]
    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == 0:
            continue
        seed = int(label) * 2654435761
        r = (seed >> 16) & 0xFF
        g = (seed >> 8) & 0xFF
        b = seed & 0xFF
        mask = labels == label

        if contours_only:
            from scipy import ndimage
            binary_mask = mask.astype(np.uint8)
            dilated = ndimage.binary_dilation(binary_mask, iterations=contour_width)
            eroded = ndimage.binary_erosion(binary_mask, iterations=1)
            boundary = dilated & ~eroded
            rgba[boundary, 0] = r
            rgba[boundary, 1] = g
            rgba[boundary, 2] = b
            rgba[boundary, 3] = 255
        else:
            rgba[mask, 0] = r
            rgba[mask, 1] = g
            rgba[mask, 2] = b
            rgba[mask, 3] = 255

    return rgba


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple (0-255).

    Args:
        hex_color: Hex color string (e.g., "#ff0000" or "ff0000")

    Returns:
        Tuple of (R, G, B) values in range 0-255
    """
    return _hex_to_rgb(hex_color)


def composite_channels(
    channels: list[np.ndarray],
    colors: list[str],
    mins: list[float],
    maxs: list[float],
    data_mins: list[float] | None = None,
    data_maxs: list[float] | None = None,
) -> np.ndarray:
    """Composite multiple channels into an RGB image.

    Args:
        channels: List of 2D arrays (one per channel)
        colors: List of hex colors for each channel
        mins: List of min values (0-1) for contrast
        maxs: List of max values (0-1) for contrast
        data_mins: List of global min values for each channel (for consistent normalization)
        data_maxs: List of global max values for each channel (for consistent normalization)

    Returns:
        RGB image as uint8 array (H, W, 3)
    """
    if not channels:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    height, width = channels[0].shape
    if height == 0 or width == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    composite = np.zeros((height, width, 3), dtype=np.float64)

    for i, (ch_data, color, vmin, vmax) in enumerate(zip(channels, colors, mins, maxs)):
        if ch_data.size == 0:
            continue
        # Normalize channel data to 0-1 using global min/max if provided
        ch_float = ch_data.astype(np.float64)
        if data_mins is not None and data_maxs is not None and i < len(data_mins) and i < len(data_maxs):
            ch_min, ch_max = data_mins[i], data_maxs[i]
        else:
            ch_min, ch_max = ch_float.min(), ch_float.max()
        if ch_max > ch_min:
            ch_norm = (ch_float - ch_min) / (ch_max - ch_min)
        else:
            ch_norm = np.zeros_like(ch_float)

        # Apply contrast limits
        ch_norm = np.clip((ch_norm - vmin) / (vmax - vmin + 1e-10), 0, 1)

        # Get RGB color
        r, g, b = _hex_to_rgb(color)

        # Additive blending
        composite[:, :, 0] += ch_norm * r
        composite[:, :, 1] += ch_norm * g
        composite[:, :, 2] += ch_norm * b

    # Clip and convert to uint8
    composite = np.clip(composite, 0, 255).astype(np.uint8)
    return composite
