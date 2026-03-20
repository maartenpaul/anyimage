"""Image loading mixin for BioImageViewer."""

import base64
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..utils import (
    CHANNEL_COLORS,
    array_to_base64,
    composite_channels,
    normalize_image,
)


_THUMBNAIL_MAX = 512  # Max dimension for tile-mode thumbnail (used as baseImage fallback)


def _thumbnail(arr: np.ndarray, max_size: int = _THUMBNAIL_MAX) -> np.ndarray:
    """Downsample array to fit within max_size using nearest-neighbor sampling."""
    h, w = arr.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1.0:
        return arr
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    ys = (np.arange(new_h) * h / new_h).astype(np.int32)
    xs = (np.arange(new_w) * w / new_w).astype(np.int32)
    return arr[ys[:, None], xs] if arr.ndim == 2 else arr[ys[:, None], xs, :]


class ImageLoadingMixin:
    """Mixin class providing image loading functionality for BioImageViewer.

    This mixin handles loading images from numpy arrays and BioImage objects,
    including support for multi-dimensional data (5D: TCZYX), lazy loading,
    tile-based rendering, and slice caching.

    Attributes expected to be defined by the main class:
        - _bioimage: Reference to BioImage object for lazy loading
        - _slice_cache: LRU cache for slice data
        - _slice_cache_max_size: Max number of cached slices
        - _tile_cache: Cache for rendered tiles
        - _tile_cache_max_size: Max cached tiles
        - _prefetch_executor: ThreadPoolExecutor for background prefetching
        - _image_array: Current image array (for SAM integration)
        - _channel_settings: List of channel settings dicts
        - dim_t, dim_c, dim_z: Dimension sizes
        - current_t, current_c, current_z: Current positions
        - current_resolution: Current resolution level
        - width, height: Image dimensions
        - image_data: Base64 encoded image data (traitlet)
        - _preview_mode: Preview mode flag (traitlet)
        - _tile_size: Tile size (traitlet)
        - _tiles_data: Tiles data dict (traitlet)
        - resolution_levels: List of resolution levels (traitlet)
        - scenes: List of scenes (traitlet)
        - current_scene: Current scene name (traitlet)
    """

    def set_image(self, data):
        """Set the base image from a numpy array or BioImage object.

        Args:
            data: Either a numpy array or a BioImage object.
                  If BioImage, enables lazy loading for 5D data.
        """
        # Check if this is a BioImage object
        if hasattr(data, "dims") and hasattr(data, "dask_data"):
            self._set_bioimage(data)
        else:
            self._set_numpy_image(data)

    def _set_numpy_image(self, data: np.ndarray):
        """Set the base image from a numpy array.

        Args:
            data: Numpy array to display. Will be squeezed if multi-dimensional.
        """
        if data.ndim > 2:
            data = data.squeeze()
            if data.ndim > 2:
                data = data[0] if data.ndim == 3 else data[0, 0]

        self.height, self.width = data.shape[:2]
        normalized = normalize_image(data)

        # Store for SAM integration
        self._image_array = normalized

        # Reset dimension info for simple arrays
        self.dim_t = 1
        self.dim_c = 1
        self.dim_z = 1
        self.current_t = 0
        self.current_c = 0
        self.current_z = 0
        self.resolution_levels = []
        self.scenes = []
        self._channel_settings = []  # No channel controls for simple arrays
        self._bioimage = None

        self.image_data = array_to_base64(normalized)

    def _set_bioimage(self, img):
        """Set the base image from a BioImage object with lazy loading support.

        Args:
            img: A BioImage object from bioio
        """
        self._bioimage = img

        # Extract dimension sizes (BioImage uses TCZYX order)
        self.dim_t = img.dims.T
        self.dim_c = img.dims.C
        self.dim_z = img.dims.Z
        self.height = img.dims.Y
        self.width = img.dims.X

        # Reset current positions
        self.current_t = 0
        self.current_c = 0
        self.current_z = 0

        # Compute global min/max for each channel by sampling slices
        # This ensures consistent normalization across all tiles, timeframes, and z-stacks
        channel_ranges = self._compute_channel_ranges(img)

        # Initialize channel settings with default colors and global ranges
        channel_settings = []
        for i in range(self.dim_c):
            data_min, data_max = channel_ranges.get(i, (0.0, 1.0))
            channel_settings.append({
                "name": f"Channel {i}",
                "color": CHANNEL_COLORS[i % len(CHANNEL_COLORS)],
                "visible": True,
                "min": 0.0,
                "max": 1.0,
                "data_min": float(data_min),
                "data_max": float(data_max),
            })
        self._channel_settings = channel_settings

        # Extract resolution levels if available
        if hasattr(img, "resolution_levels") and img.resolution_levels:
            self.resolution_levels = list(img.resolution_levels)
            self.current_resolution = 0
        else:
            self.resolution_levels = []

        # Extract scenes if available
        if hasattr(img, "scenes") and img.scenes:
            self.scenes = list(img.scenes)
            self.current_scene = img.current_scene if hasattr(img, "current_scene") else ""
        else:
            self.scenes = []

        self._update_slice()

    def _compute_channel_ranges(self, img) -> dict[int, tuple[float, float]]:
        """Compute global min/max ranges for each channel by sampling slices.

        Samples slices across T and Z dimensions to estimate the global data range
        for consistent normalization across all tiles, timeframes, and z-stacks.

        Args:
            img: A BioImage object

        Returns:
            Dictionary mapping channel index to (min, max) tuple
        """
        channel_ranges = {}

        # Sample strategy: sample first, middle, and last T/Z positions
        t_samples = list(set([0, self.dim_t // 2, self.dim_t - 1]))
        z_samples = list(set([0, self.dim_z // 2, self.dim_z - 1]))

        for c in range(self.dim_c):
            global_min = float("inf")
            global_max = float("-inf")

            for t in t_samples:
                for z in z_samples:
                    try:
                        slice_data = img.get_image_dask_data("YX", T=t, C=c, Z=z).compute()
                        slice_min = float(slice_data.min())
                        slice_max = float(slice_data.max())
                        global_min = min(global_min, slice_min)
                        global_max = max(global_max, slice_max)
                    except Exception:
                        continue

            # Fallback if no valid samples
            if global_min == float("inf") or global_max == float("-inf"):
                global_min, global_max = 0.0, 1.0

            channel_ranges[c] = (global_min, global_max)

        return channel_ranges

    def _get_slice_cached(self, t: int, c: int, z: int) -> np.ndarray:
        """Get slice data from cache or load from disk.

        Args:
            t: Time index
            c: Channel index
            z: Z-slice index

        Returns:
            2D numpy array of slice data
        """
        cache_key = (t, c, z)
        if cache_key in self._slice_cache:
            return self._slice_cache[cache_key]

        ch_data = self._bioimage.get_image_dask_data("YX", T=t, C=c, Z=z).compute()

        if len(self._slice_cache) >= self._slice_cache_max_size:
            del self._slice_cache[next(iter(self._slice_cache))]

        self._slice_cache[cache_key] = ch_data
        return ch_data

    def _get_tile(self, t: int, z: int, tile_x: int, tile_y: int):
        """Get a single tile as raw RGBA bytes (much faster than PNG).

        Args:
            t: Time index
            z: Z-slice index
            tile_x: Tile X coordinate
            tile_y: Tile Y coordinate

        Returns:
            Dictionary with tile data {w, h, data} or None if tile is empty
        """
        cache_key = (t, z, tile_x, tile_y, self.current_resolution)
        if cache_key in self._tile_cache:
            return self._tile_cache[cache_key]

        y_start = tile_y * self._tile_size
        x_start = tile_x * self._tile_size
        if y_start >= self.height or x_start >= self.width:
            return None

        y_end = min(y_start + self._tile_size, self.height)
        x_end = min(x_start + self._tile_size, self.width)

        visible_channels, colors, mins, maxs, data_mins, data_maxs = [], [], [], [], [], []
        for i, ch_settings in enumerate(self._channel_settings):
            if ch_settings.get("visible", True):
                ch_data = self._get_slice_cached(t, i, z)
                if ch_data is None:
                    continue
                tile_slice = ch_data[y_start:y_end, x_start:x_end]
                if tile_slice.size == 0:
                    continue
                visible_channels.append(tile_slice)
                colors.append(ch_settings.get("color", "#ffffff"))
                mins.append(ch_settings.get("min", 0.0))
                maxs.append(ch_settings.get("max", 1.0))
                data_mins.append(ch_settings.get("data_min"))
                data_maxs.append(ch_settings.get("data_max"))

        if not visible_channels:
            return None

        composite = composite_channels(visible_channels, colors, mins, maxs, data_mins, data_maxs)
        h, w = composite.shape[:2]

        # Convert RGB to RGBA (add alpha channel)
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = composite
        rgba[:, :, 3] = 255

        # Raw bytes are ~10x faster than PNG encoding
        tile_data = {
            "w": w, "h": h,
            "data": base64.b64encode(rgba.tobytes()).decode("utf-8")
        }

        if len(self._tile_cache) >= self._tile_cache_max_size:
            del self._tile_cache[next(iter(self._tile_cache))]
        self._tile_cache[cache_key] = tile_data

        return tile_data

    def _on_tile_request(self, change):
        """Handle tile request from JavaScript.

        Args:
            change: Traitlet change dict with 'new' containing the tile request
        """
        start_total = time.perf_counter()

        request = change.get("new")
        if not request or self._bioimage is None:
            return

        t = request.get("t", self.current_t)
        z = request.get("z", self.current_z)
        tiles_data = {}
        num_cached, num_generated = 0, 0

        for tile_info in request.get("tiles", []):
            tx, ty = tile_info.get("tx"), tile_info.get("ty")
            if tx is not None and ty is not None:
                cache_key = (t, z, tx, ty, self.current_resolution)
                was_cached = cache_key in self._tile_cache
                tile_data = self._get_tile(t, z, tx, ty)
                if tile_data:
                    tiles_data[f"{t}_{z}_{tx}_{ty}"] = tile_data
                    if was_cached:
                        num_cached += 1
                    else:
                        num_generated += 1

        elapsed = (time.perf_counter() - start_total) * 1000
        print(f"[Py] {num_generated} new + {num_cached} cached tiles in {elapsed:.0f}ms")
        self._tiles_data = tiles_data

    def _prefetch_slice(self, t: int, c: int, z: int) -> None:
        """Prefetch a slice into the cache (background thread).

        Args:
            t: Time index
            c: Channel index
            z: Z-slice index
        """
        if self._bioimage is None:
            return
        cache_key = (t, c, z)
        if cache_key in self._slice_cache:
            return
        try:
            ch_data = self._bioimage.get_image_dask_data("YX", T=t, C=c, Z=z).compute()
            if len(self._slice_cache) < self._slice_cache_max_size:
                self._slice_cache[cache_key] = ch_data
        except Exception:
            pass

    def _prefetch_tiles_for_slice(self, t: int, z: int) -> None:
        """Pre-render all tiles for a (t, z) slice into the tile cache."""
        if self._bioimage is None:
            return
        n_tiles_x = (self.width + self._tile_size - 1) // self._tile_size
        n_tiles_y = (self.height + self._tile_size - 1) // self._tile_size
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                cache_key = (t, z, tx, ty, self.current_resolution)
                if cache_key not in self._tile_cache:
                    try:
                        self._get_tile(t, z, tx, ty)
                    except Exception:
                        pass

    def _prefetch_adjacent_slices(self) -> None:
        """Pre-fetch adjacent Z and T slices in background for smoother scrubbing."""
        if self._bioimage is None:
            return

        t, z = self.current_t, self.current_z
        visible_channels = [i for i, ch in enumerate(self._channel_settings) if ch.get("visible", True)]

        for delta in [-1, 1]:
            if 0 <= z + delta < self.dim_z:
                for c in visible_channels:
                    self._prefetch_executor.submit(self._prefetch_slice, t, c, z + delta)
                self._prefetch_executor.submit(self._prefetch_tiles_for_slice, t, z + delta)
            if 0 <= t + delta < self.dim_t:
                for c in visible_channels:
                    self._prefetch_executor.submit(self._prefetch_slice, t + delta, c, z)
                self._prefetch_executor.submit(self._prefetch_tiles_for_slice, t + delta, z)

    def _update_slice(self):
        """Update the displayed slice based on current T, Z positions and channel settings."""
        if self._bioimage is None:
            return

        try:
            # Get visible channels
            visible_channels = []
            colors = []
            mins = []
            maxs = []
            data_mins = []
            data_maxs = []

            # In preview mode, only use cached data (no disk I/O)
            preview_mode = self._preview_mode

            for i, ch_settings in enumerate(self._channel_settings):
                if ch_settings.get("visible", True):
                    cache_key = (self.current_t, i, self.current_z)
                    if cache_key in self._slice_cache:
                        # Use cached data
                        ch_data = self._slice_cache[cache_key]
                    elif preview_mode:
                        # In preview mode, skip uncached channels (don't block on I/O)
                        # Schedule background load instead
                        self._prefetch_executor.submit(
                            self._prefetch_slice, self.current_t, i, self.current_z
                        )
                        continue
                    else:
                        # Not in preview mode, load from disk
                        ch_data = self._get_slice_cached(self.current_t, i, self.current_z)

                    visible_channels.append(ch_data)
                    colors.append(ch_settings.get("color", "#ffffff"))
                    mins.append(ch_settings.get("min", 0.0))
                    maxs.append(ch_settings.get("max", 1.0))
                    data_mins.append(ch_settings.get("data_min"))
                    data_maxs.append(ch_settings.get("data_max"))

            use_tile_mode = self._use_tile_mode

            if not visible_channels:
                # No visible channels (or all uncached in preview mode)
                if not preview_mode:
                    if use_tile_mode:
                        self.image_data = ""
                    else:
                        normalized = np.zeros((self.height, self.width), dtype=np.uint8)
                        self._image_array = normalized
                        self.image_data = array_to_base64(normalized)
                return

            if len(visible_channels) == 1 and colors[0] == "#ffffff":
                # Single grayscale channel - no compositing needed
                global_min = data_mins[0] if data_mins and data_mins[0] is not None else None
                global_max = data_maxs[0] if data_maxs and data_maxs[0] is not None else None
                normalized = normalize_image(visible_channels[0], global_min, global_max)
                # Apply contrast
                vmin, vmax = mins[0], maxs[0]
                if vmin > 0 or vmax < 1:
                    normalized = normalized.astype(np.float64) / 255.0
                    normalized = np.clip((normalized - vmin) / (vmax - vmin + 1e-10), 0, 1)
                    normalized = (normalized * 255).astype(np.uint8)
                self._image_array = normalized
                if use_tile_mode:
                    self.image_data = array_to_base64(_thumbnail(normalized))
                else:
                    self.image_data = array_to_base64(normalized)
            else:
                # Composite multiple channels
                composite = composite_channels(visible_channels, colors, mins, maxs, data_mins, data_maxs)
                # Store grayscale version for SAM
                self._image_array = np.mean(composite, axis=2).astype(np.uint8)
                if use_tile_mode:
                    self.image_data = array_to_base64(_thumbnail(composite))
                else:
                    self.image_data = array_to_base64(composite)

            # Pre-fetch adjacent slices for smoother navigation (not in preview mode)
            if not preview_mode:
                self._prefetch_adjacent_slices()

        except Exception as e:
            print(f"Error updating slice: {e}")

    def _on_dimension_change(self, change):
        """Observer callback when T or Z dimension changes."""
        self._update_slice()

    def _on_channel_settings_change(self, change):
        """Observer callback when channel settings change."""
        self._tile_cache.clear()
        self._update_slice()

    def _clear_caches(self):
        """Clear both slice and tile caches."""
        self._slice_cache.clear()
        self._tile_cache.clear()

    def _on_resolution_change(self, change):
        """Observer callback when resolution level changes."""
        if self._bioimage is None or not hasattr(self._bioimage, "set_resolution_level"):
            return

        try:
            self._bioimage.set_resolution_level(change.get("new", 0))
            self._clear_caches()
            self.height = self._bioimage.dims.Y
            self.width = self._bioimage.dims.X
            self._update_slice()
        except Exception as e:
            print(f"Error changing resolution level: {e}")

    def _on_scene_change(self, change):
        """Observer callback when scene changes."""
        new_scene = change.get("new", "")
        if self._bioimage is None or not new_scene or not hasattr(self._bioimage, "set_scene"):
            return

        try:
            self._bioimage.set_scene(new_scene)
            self._clear_caches()
            self.dim_t = self._bioimage.dims.T
            self.dim_c = self._bioimage.dims.C
            self.dim_z = self._bioimage.dims.Z
            self.height = self._bioimage.dims.Y
            self.width = self._bioimage.dims.X
            self.current_t = 0
            self.current_c = 0
            self.current_z = 0
            self._update_slice()
        except Exception as e:
            print(f"Error changing scene: {e}")
