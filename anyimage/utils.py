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


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to RGB tuple (0-255).

    Args:
        hex_color: Hex color string (e.g., "#ff0000" or "ff0000")

    Returns:
        Tuple of (R, G, B) values in range 0-255
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
        Base64 encoded PNG string (no data URI prefix).

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


def array_to_fast_png_base64(data: np.ndarray) -> str:
    """Convert numpy array to base64 encoded PNG using fast (low) compression.

    Uses compress_level=1 instead of PIL's default of 6 — 5× faster with only
    ~20% larger output. Suitable for thumbnails sent frequently during navigation.

    Args:
        data: Input array (2D grayscale, 3D RGB, or 3D RGBA)

    Returns:
        Base64 encoded PNG string (no data URI prefix).
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


_lut_cache: dict = {}
_LUT_CACHE_MAX = 128


def build_channel_lut(
    color: str,
    vmin: float,
    vmax: float,
    data_min: float,
    data_max: float,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    """Build a per-channel R/G/B LUT for fast pixel mapping.

    Returns a (N, 3) uint8 array where N = number of possible input values
    (65536 for 16-bit, 256 for 8-bit). Results are cached by parameters so
    repeated calls during precompute reuse the same array.
    """
    n = 65536 if (dtype is None or np.dtype(dtype).itemsize > 1) else 256
    cache_key = (color, round(vmin, 6), round(vmax, 6), round(data_min, 6), round(data_max, 6), n)
    if cache_key in _lut_cache:
        return _lut_cache[cache_key]

    indices = np.arange(n, dtype=np.float32)

    # Map raw value → [0, 1] using global data range
    span = data_max - data_min
    if span > 0:
        normalized = (indices - data_min) / span
    else:
        normalized = np.zeros(n, dtype=np.float32)

    # Apply contrast window
    contrast_span = vmax - vmin + 1e-10
    normalized = np.clip((normalized - vmin) / contrast_span, 0, 1)

    r, g, b = hex_to_rgb(color)
    lut = np.empty((n, 3), dtype=np.uint8)
    lut[:, 0] = np.clip(normalized * r, 0, 255).astype(np.uint8)
    lut[:, 1] = np.clip(normalized * g, 0, 255).astype(np.uint8)
    lut[:, 2] = np.clip(normalized * b, 0, 255).astype(np.uint8)

    if len(_lut_cache) >= _LUT_CACHE_MAX:
        del _lut_cache[next(iter(_lut_cache))]
    _lut_cache[cache_key] = lut
    return lut


def composite_channels(
    channels: list[np.ndarray],
    colors: list[str],
    mins: list[float],
    maxs: list[float],
    data_mins: list[float] | None = None,
    data_maxs: list[float] | None = None,
) -> np.ndarray:
    """Composite multiple channels into an RGB image.

    Uses prebuilt LUTs for fast pixel mapping — avoids per-pixel float arithmetic.

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

    n_channels = len(channels)

    # Single-channel fast path: skip accumulator, write directly to uint8 output
    if n_channels == 1:
        ch_data, color, vmin, vmax = channels[0], colors[0], mins[0], maxs[0]
        ch_min = data_mins[0] if (data_mins and data_mins[0] is not None) else float(ch_data.min())
        ch_max = data_maxs[0] if (data_maxs and data_maxs[0] is not None) else float(ch_data.max())
        r, g, b = hex_to_rgb(color)
        result = np.zeros((height, width, 3), dtype=np.uint8)
        if np.issubdtype(ch_data.dtype, np.integer) and ch_data.dtype.itemsize <= 2:
            n = 65536 if ch_data.dtype.itemsize == 2 else 256
            indices = np.arange(n, dtype=np.float32)
            span = ch_max - ch_min
            norm = (indices - ch_min) / span if span > 0 else np.zeros(n, dtype=np.float32)
            norm = np.clip((norm - vmin) / (vmax - vmin + 1e-10), 0, 1)
            idx = ch_data.byteswap(inplace=False).view(ch_data.dtype.newbyteorder("=")) if not ch_data.dtype.isnative else ch_data
            if r: result[:, :, 0] = (norm * r).astype(np.uint8)[idx]
            if g: result[:, :, 1] = (norm * g).astype(np.uint8)[idx]
            if b: result[:, :, 2] = (norm * b).astype(np.uint8)[idx]
        else:
            ch_float = ch_data.astype(np.float32)
            span = ch_max - ch_min
            ch_norm = np.clip((ch_float - ch_min) / span if span > 0 else np.zeros_like(ch_float), 0, 1)
            ch_norm = np.clip((ch_norm - vmin) / (vmax - vmin + 1e-10), 0, 1)
            if r: result[:, :, 0] = np.clip(ch_norm * r, 0, 255).astype(np.uint8)
            if g: result[:, :, 1] = np.clip(ch_norm * g, 0, 255).astype(np.uint8)
            if b: result[:, :, 2] = np.clip(ch_norm * b, 0, 255).astype(np.uint8)
        return result

    # Multi-channel path: uint16 accumulator to handle additive blending without overflow
    composite = np.zeros((height, width, 3), dtype=np.uint16)

    for i, (ch_data, color, vmin, vmax) in enumerate(zip(channels, colors, mins, maxs)):
        if ch_data.size == 0:
            continue

        ch_min = data_mins[i] if (data_mins and i < len(data_mins) and data_mins[i] is not None) else float(ch_data.min())
        ch_max = data_maxs[i] if (data_maxs and i < len(data_maxs) and data_maxs[i] is not None) else float(ch_data.max())

        r, g, b = hex_to_rgb(color)

        if np.issubdtype(ch_data.dtype, np.integer) and ch_data.dtype.itemsize <= 2:
            n = 65536 if ch_data.dtype.itemsize == 2 else 256
            indices = np.arange(n, dtype=np.float32)
            span = ch_max - ch_min
            norm = (indices - ch_min) / span if span > 0 else np.zeros(n, dtype=np.float32)
            norm = np.clip((norm - vmin) / (vmax - vmin + 1e-10), 0, 1)
            idx = ch_data.byteswap(inplace=False).view(ch_data.dtype.newbyteorder("=")) if not ch_data.dtype.isnative else ch_data
            if r: composite[:, :, 0] += (norm * r).astype(np.uint8)[idx]
            if g: composite[:, :, 1] += (norm * g).astype(np.uint8)[idx]
            if b: composite[:, :, 2] += (norm * b).astype(np.uint8)[idx]
        else:
            ch_float = ch_data.astype(np.float32)
            span = ch_max - ch_min
            ch_norm = (ch_float - ch_min) / span if span > 0 else np.zeros_like(ch_float)
            ch_norm = np.clip((ch_norm - vmin) / (vmax - vmin + 1e-10), 0, 1)
            if r: composite[:, :, 0] += (ch_norm * r).astype(np.uint16)
            if g: composite[:, :, 1] += (ch_norm * g).astype(np.uint16)
            if b: composite[:, :, 2] += (ch_norm * b).astype(np.uint16)

    return np.clip(composite, 0, 255).astype(np.uint8)
