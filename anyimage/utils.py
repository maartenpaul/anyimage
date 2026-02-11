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

# Pre-computed color lookup table for hex_to_rgb (avoids repeated parsing)
_hex_color_cache: dict[str, tuple[int, int, int]] = {}


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to RGB tuple (0-255).

    Args:
        hex_color: Hex color string (e.g., "#ff0000" or "ff0000")

    Returns:
        Tuple of (R, G, B) values in range 0-255
    """
    cached = _hex_color_cache.get(hex_color)
    if cached is not None:
        return cached
    stripped = hex_color.lstrip("#")
    result = (int(stripped[0:2], 16), int(stripped[2:4], 16), int(stripped[4:6], 16))
    _hex_color_cache[hex_color] = result
    return result


def normalize_image(
    data: np.ndarray,
    global_min: float | None = None,
    global_max: float | None = None,
) -> np.ndarray:
    """Normalize image data to uint8 range.

    Uses float32 for ~2x faster computation compared to float64 on large images.

    Args:
        data: Input array to normalize
        global_min: Optional global minimum value for consistent normalization
        global_max: Optional global maximum value for consistent normalization

    Returns:
        Normalized uint8 array
    """
    if data.dtype == np.uint8 and global_min is None and global_max is None:
        return data

    if global_min is not None and global_max is not None:
        data_min, data_max = global_min, global_max
    else:
        data_min, data_max = float(data.min()), float(data.max())

    if data_max > data_min:
        # Use float32 for speed; precision is sufficient for uint8 output
        result = data.astype(np.float32)
        scale = np.float32(255.0 / (data_max - data_min))
        result -= np.float32(data_min)
        result *= scale
        np.clip(result, 0, 255, out=result)
        return result.astype(np.uint8)
    else:
        return np.zeros(data.shape, dtype=np.uint8)


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
    img.save(buffer, format="PNG", compress_level=1)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def labels_to_rgba(
    labels: np.ndarray, contours_only: bool = False, contour_width: int = 1
) -> np.ndarray:
    """Convert label array to RGBA with unique colors per label.

    Uses vectorized operations to avoid per-label Python loops, providing
    significant speedup for masks with many labels.

    Args:
        labels: 2D array of integer labels (0 is background)
        contours_only: If True, only draw contours instead of filled regions
        contour_width: Width of contours in pixels (only used if contours_only=True)

    Returns:
        RGBA image as uint8 array (H, W, 4)
    """
    height, width = labels.shape[:2]
    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    # Create foreground mask (non-zero labels)
    foreground = labels != 0
    if not foreground.any():
        return rgba

    if contours_only:
        from scipy import ndimage

        # Compute boundary for ALL labels at once, then colorize
        # First, detect boundary pixels: where the label changes between neighbors
        # Dilate the foreground and erode it to find boundaries
        fg_uint8 = foreground.astype(np.uint8)
        dilated = ndimage.binary_dilation(fg_uint8, iterations=contour_width)
        eroded = ndimage.binary_erosion(fg_uint8, iterations=1)
        boundary = dilated & ~eroded

        # Vectorized color assignment for boundary pixels
        boundary_labels = labels[boundary]
        seeds = boundary_labels.astype(np.int64) * np.int64(2654435761)
        rgba[boundary, 0] = ((seeds >> 16) & 0xFF).astype(np.uint8)
        rgba[boundary, 1] = ((seeds >> 8) & 0xFF).astype(np.uint8)
        rgba[boundary, 2] = (seeds & 0xFF).astype(np.uint8)
        rgba[boundary, 3] = 255
    else:
        # Vectorized color assignment: compute colors from label values directly
        fg_labels = labels[foreground]
        seeds = fg_labels.astype(np.int64) * np.int64(2654435761)
        rgba[foreground, 0] = ((seeds >> 16) & 0xFF).astype(np.uint8)
        rgba[foreground, 1] = ((seeds >> 8) & 0xFF).astype(np.uint8)
        rgba[foreground, 2] = (seeds & 0xFF).astype(np.uint8)
        rgba[foreground, 3] = 255

    return rgba


def composite_channels(
    channels: list[np.ndarray],
    colors: list[str],
    mins: list[float],
    maxs: list[float],
    data_mins: list[float] | None = None,
    data_maxs: list[float] | None = None,
) -> np.ndarray:
    """Composite multiple channels into an RGB image.

    Uses float32 arithmetic for ~2x faster computation on large images.
    Pre-computes color arrays for efficient vectorized blending.

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

    composite = np.zeros((height, width, 3), dtype=np.float32)

    for i, (ch_data, color, vmin, vmax) in enumerate(zip(channels, colors, mins, maxs)):
        if ch_data.size == 0:
            continue
        # Normalize channel data to 0-1 using global min/max if provided
        ch_float = ch_data.astype(np.float32)
        if data_mins is not None and data_maxs is not None and i < len(data_mins) and i < len(data_maxs):
            ch_min, ch_max = data_mins[i], data_maxs[i]
        else:
            ch_min, ch_max = float(ch_float.min()), float(ch_float.max())
        if ch_max > ch_min:
            inv_range = np.float32(1.0 / (ch_max - ch_min))
            ch_float -= np.float32(ch_min)
            ch_float *= inv_range
        else:
            ch_float = np.zeros_like(ch_float)

        # Apply contrast limits
        contrast_range = vmax - vmin + 1e-10
        inv_contrast = np.float32(1.0 / contrast_range)
        ch_float -= np.float32(vmin)
        ch_float *= inv_contrast
        np.clip(ch_float, 0, 1, out=ch_float)

        # Get RGB color and blend in one step
        r, g, b = hex_to_rgb(color)
        composite[:, :, 0] += ch_float * np.float32(r)
        composite[:, :, 1] += ch_float * np.float32(g)
        composite[:, :, 2] += ch_float * np.float32(b)

    # Clip and convert to uint8
    np.clip(composite, 0, 255, out=composite)
    return composite.astype(np.uint8)
