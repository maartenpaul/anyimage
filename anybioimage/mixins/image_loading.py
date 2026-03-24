"""Image loading mixin for BioImageViewer."""

import base64
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# If the full image array fits within this threshold, load it all into RAM eagerly.
_EAGER_LOAD_BYTES = 2 * 1024 ** 3  # 2 GB

from ..utils import (
    CHANNEL_COLORS,
    array_to_base64,
    array_to_fast_png_base64,
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
        # Store raw data for re-rendering on LUT/contrast changes
        self._raw_numpy_array = data

        # Always create channel settings so UI controls are available
        data_min = float(data.min()) if data.size > 0 else 0.0
        data_max = float(data.max()) if data.size > 0 else 1.0
        self._channel_settings = [{
            "name": "Channel 0",
            "color": "#ffffff",
            "visible": True,
            "min": 0.0,
            "max": 1.0,
            "data_min": data_min,
            "data_max": data_max,
        }]
        self._bioimage = None

        self.image_data = array_to_base64(normalized)

    def _set_bioimage(self, img):
        """Set the base image from a BioImage object with lazy loading support.

        Args:
            img: A BioImage object from bioio
        """
        self._bioimage = img
        self._full_array = None

        # Pre-load full array into RAM if it fits (avoids per-slice zarr I/O during navigation)
        try:
            itemsize = np.dtype(img.dtype).itemsize
        except Exception:
            itemsize = 2  # conservative fallback (uint16)
        nbytes = img.dims.T * img.dims.C * img.dims.Z * img.dims.Y * img.dims.X * itemsize
        eager_array = None
        if nbytes <= _EAGER_LOAD_BYTES:
            try:
                arr = img.get_image_dask_data("TCZYX").compute()
                if not arr.dtype.isnative:
                    arr = np.ascontiguousarray(arr.astype(arr.dtype.newbyteorder("=")))
                eager_array = arr
            except Exception as e:
                print(f"[anybioimage] Eager load failed, falling back to lazy loading: {e}")

        if eager_array is not None:
            channel_ranges = self._compute_channel_ranges_from_array(eager_array)
        else:
            channel_ranges = self._compute_channel_ranges(img)

        # Batch all traitlet assignments to suppress intermediate observer callbacks
        with self.hold_trait_notifications():
            # Extract dimension sizes (BioImage uses TCZYX order)
            self.dim_t = img.dims.T
            self.dim_c = img.dims.C
            self.dim_z = img.dims.Z
            self.height = img.dims.Y
            self.width = img.dims.X

            # Pre-set tile mode for large images so _update_slice skips full PNG encoding
            if img.dims.Y * img.dims.X >= 1024 * 1024:
                self._use_tile_mode = True

            # Reset current positions
            self.current_t = 0
            self.current_c = 0
            self.current_z = 0

            # Read channel names from OME metadata if available
            try:
                raw_names = img.channel_names if hasattr(img, "channel_names") else None
                channel_names = list(raw_names) if raw_names else []
            except Exception:
                channel_names = []

            # Initialize channel settings with default colors and global ranges
            channel_settings = []
            for i in range(self.dim_c):
                range_data = channel_ranges.get(i, (0.0, 1.0, 0.0, 1.0))
                abs_min, abs_max, display_lo, display_hi = range_data
                span = abs_max - abs_min
                vmin = max(0.0, (display_lo - abs_min) / span) if span > 0 else 0.0
                vmax = min(1.0, (display_hi - abs_min) / span) if span > 0 else 1.0
                name = channel_names[i] if i < len(channel_names) else f"Channel {i}"
                channel_settings.append({
                    "name": name,
                    "color": "#ffffff" if self.dim_c == 1 else CHANNEL_COLORS[i % len(CHANNEL_COLORS)],
                    "visible": True,
                    "min": vmin,
                    "max": vmax,
                    "data_min": abs_min,
                    "data_max": abs_max,
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

            # Set full array inside hold so _clear_caches from scene/resolution observers can't wipe it
            self._full_array = eager_array

            # Build synthetic pyramid for flat files (no native resolution levels)
            has_native = (
                hasattr(img, "resolution_levels")
                and isinstance(img.resolution_levels, (list, tuple))
                and len(img.resolution_levels) > 1
            )
            if eager_array is not None and not has_native:
                levels = [eager_array]
                for step in [2, 4]:
                    h, w = eager_array.shape[3] // step, eager_array.shape[4] // step
                    if h < 256 or w < 256:
                        break
                    levels.append(None)  # materialised on demand
                self._pyramid = levels
                self._pyramid_has_native = False
                self.resolution_levels = list(range(len(levels)))
            else:
                self._pyramid = None
                self._pyramid_has_native = has_native

        # Dynamic composite cache sizing: fit as many T×Z composites as memory allows
        bytes_per_composite = self.height * self.width * 3
        mem_limit = 512 * 1024 * 1024  # 512 MB budget
        max_by_memory = mem_limit // max(bytes_per_composite, 1)
        self._composite_cache_max_size = max(64, min(self.dim_t * self.dim_z, max_by_memory, 1024))

        self._update_slice()
        self._start_precompute()

    def _compute_channel_ranges(self, img) -> dict[int, tuple[float, float, float, float]]:
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
            samples = []

            for t in t_samples:
                for z in z_samples:
                    try:
                        slice_data = img.get_image_dask_data("YX", T=t, C=c, Z=z).compute()
                        samples.append(slice_data.ravel())
                    except Exception:
                        continue

            if not samples:
                channel_ranges[c] = (0.0, 1.0, 0.0, 1.0)
                continue

            combined = np.concatenate(samples)
            abs_min = float(combined.min())
            abs_max = float(combined.max())
            display_lo = float(np.percentile(combined, 1))
            display_hi = float(np.percentile(combined, 99))
            if display_hi <= display_lo:
                display_lo, display_hi = abs_min, abs_max

            channel_ranges[c] = (abs_min, abs_max, display_lo, display_hi)

        return channel_ranges

    def _compute_channel_ranges_from_array(self, arr: np.ndarray) -> dict[int, tuple[float, float, float, float]]:
        """Compute absolute range and display window per channel from a TCZYX array.

        Returns (abs_min, abs_max, display_lo, display_hi) per channel where
        abs_min/abs_max span the full data range (no clipping) and display_lo/hi
        are p0.5/p99.5 percentiles used to set the initial contrast window.
        """
        channel_ranges = {}
        for c in range(arr.shape[1]):
            ch = arr[:, c, :, :, :].ravel()
            abs_min = float(ch.min())
            abs_max = float(ch.max())
            lo = float(np.percentile(ch, 0.5))
            hi = float(np.percentile(ch, 99.5))
            if hi <= lo:
                lo, hi = abs_min, abs_max
            channel_ranges[c] = (abs_min, abs_max, lo, hi)
        return channel_ranges

    def _get_pyramid_level(self, level: int) -> np.ndarray:
        """Return (and materialise if needed) the TCZYX array for a synthetic pyramid level."""
        if level == 0 or self._pyramid is None:
            return self._full_array
        if self._pyramid[level] is None:
            step = 2 ** level
            self._pyramid[level] = np.ascontiguousarray(
                self._full_array[:, :, :, ::step, ::step]
            )
        return self._pyramid[level]

    def _get_slice_cached(self, t: int, c: int, z: int) -> np.ndarray:
        """Get slice data from cache or load from disk.

        Args:
            t: Time index
            c: Channel index
            z: Z-slice index

        Returns:
            2D numpy array of slice data
        """
        if self._full_array is not None:
            pyramid = getattr(self, "_pyramid", None)
            pyramid_has_native = getattr(self, "_pyramid_has_native", False)
            if pyramid is not None and not pyramid_has_native:
                return self._get_pyramid_level(self.current_resolution)[t, c, z]
            return self._full_array[t, c, z]

        cache_key = (t, c, z)
        if cache_key in self._slice_cache:
            return self._slice_cache[cache_key]

        ch_data = self._bioimage.get_image_dask_data("YX", T=t, C=c, Z=z).compute()

        if len(self._slice_cache) >= self._slice_cache_max_size:
            del self._slice_cache[next(iter(self._slice_cache))]

        self._slice_cache[cache_key] = ch_data
        return ch_data

    def _get_composite_slice(self, t: int, z: int) -> np.ndarray | None:
        """Return the full composited RGB slice for (t, z), using a cache.

        Compositing once per slice is far faster than compositing per tile,
        since normalize/blend only runs once regardless of how many tiles are requested.
        """
        composite_cache: dict = getattr(self, "_composite_cache", {})
        composite_cache_lock = getattr(self, "_composite_cache_lock", None)
        cache_key = (t, z, self.current_resolution)

        if composite_cache_lock is not None:
            with composite_cache_lock:
                if cache_key in composite_cache:
                    return composite_cache[cache_key]
        elif cache_key in composite_cache:
            return composite_cache[cache_key]

        visible_indices = [
            i for i, ch in enumerate(self._channel_settings)
            if ch.get("visible", True)
        ]
        if not visible_indices:
            return None

        # Fetch all channels in parallel — each is an independent S3 GET for remote zarr
        n_workers = min(len(visible_indices), 4)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(self._get_slice_cached, t, i, z): i for i in visible_indices}
            channel_data = {futures[f]: f.result() for f in futures}

        visible_channels, colors, mins, maxs, data_mins, data_maxs = [], [], [], [], [], []
        for i in visible_indices:
            ch_data = channel_data[i]
            if ch_data is None or ch_data.size == 0:
                continue
            ch_settings = self._channel_settings[i]
            visible_channels.append(ch_data)
            colors.append(ch_settings.get("color", "#ffffff"))
            mins.append(ch_settings.get("min", 0.0))
            maxs.append(ch_settings.get("max", 1.0))
            data_mins.append(ch_settings.get("data_min"))
            data_maxs.append(ch_settings.get("data_max"))

        if not visible_channels:
            return None

        result = composite_channels(visible_channels, colors, mins, maxs, data_mins, data_maxs)

        composite_cache_max: int = getattr(self, "_composite_cache_max_size", 64)
        if composite_cache_lock is not None:
            with composite_cache_lock:
                if cache_key not in composite_cache:  # re-check inside lock
                    if len(composite_cache) >= composite_cache_max:
                        del composite_cache[next(iter(composite_cache))]
                    composite_cache[cache_key] = result
        else:
            if len(composite_cache) >= composite_cache_max:
                del composite_cache[next(iter(composite_cache))]
            composite_cache[cache_key] = result
        return result

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
        tile_cache_lock = getattr(self, "_tile_cache_lock", None)

        if tile_cache_lock is not None:
            with tile_cache_lock:
                if cache_key in self._tile_cache:
                    return self._tile_cache[cache_key]
        elif cache_key in self._tile_cache:
            return self._tile_cache[cache_key]

        y_start = tile_y * self._tile_size
        x_start = tile_x * self._tile_size
        if y_start >= self.height or x_start >= self.width:
            return None

        y_end = min(y_start + self._tile_size, self.height)
        x_end = min(x_start + self._tile_size, self.width)

        composite = self._get_composite_slice(t, z)
        if composite is None:
            return None
        region = composite[y_start:y_end, x_start:x_end]
        if region.size == 0:
            return None
        h, w = region.shape[:2]

        use_jpeg = getattr(self, "use_jpeg_tiles", False)
        if use_jpeg:
            from io import BytesIO
            from PIL import Image as PILImage
            buf = BytesIO()
            PILImage.fromarray(region, mode="RGB").save(buf, format="JPEG", quality=85, optimize=False)
            tile_data = {"w": w, "h": h, "data": base64.b64encode(buf.getvalue()).decode("utf-8"), "fmt": "jpeg"}
        else:
            # Convert RGB to RGBA (add alpha channel)
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = region
            rgba[:, :, 3] = 255
            # Raw bytes are ~10x faster than PNG encoding
            tile_data = {"w": w, "h": h, "data": base64.b64encode(rgba.tobytes()).decode("utf-8")}

        if tile_cache_lock is not None:
            with tile_cache_lock:
                if cache_key not in self._tile_cache:  # re-check inside lock
                    if len(self._tile_cache) >= self._tile_cache_max_size:
                        del self._tile_cache[next(iter(self._tile_cache))]
                    self._tile_cache[cache_key] = tile_data
        else:
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

    def _get_viewport_tile_ranges(self) -> tuple[int, int, int, int] | None:
        """Return (tx_start, ty_start, tx_end, ty_end) from the last JS viewport report, or None."""
        vp = getattr(self, "_viewport_tiles", {})
        if not vp:
            return None
        try:
            return (int(vp["tx0"]), int(vp["ty0"]), int(vp["tx1"]), int(vp["ty1"]))
        except (KeyError, TypeError, ValueError):
            return None

    def _precompute_all_composites(self, cancel_event) -> None:
        """Background task: composite all (t,z) slices and pre-render their tiles.

        Strategy:
          Pass 1 — viewport tiles only, all T/Z slices, sorted by distance from current.
                    This makes Z/T scrubbing instant for whatever the user is looking at.
          Pass 2 — remaining (non-viewport) tiles for all slices.
                    Fills the full cache in the background.

        Only runs when _full_array is in RAM. Cancelled via cancel_event.
        """
        if self._full_array is None:
            return

        t_center, z_center = self.current_t, self.current_z
        res = self.current_resolution

        n_tiles_x = (self.width + self._tile_size - 1) // self._tile_size
        n_tiles_y = (self.height + self._tile_size - 1) // self._tile_size

        slices = sorted(
            [(t, z) for t in range(self.dim_t) for z in range(self.dim_z)],
            key=lambda tz: abs(tz[0] - t_center) + abs(tz[1] - z_center),
        )
        total = len(slices)
        report_every = max(1, total // 10)

        vp = self._get_viewport_tile_ranges()
        if vp:
            vp_tx0, vp_ty0, vp_tx1, vp_ty1 = vp
            viewport_tiles = [
                (tx, ty)
                for ty in range(max(0, vp_ty0), min(n_tiles_y, vp_ty1))
                for tx in range(max(0, vp_tx0), min(n_tiles_x, vp_tx1))
            ]
            offscreen_tiles = [
                (tx, ty)
                for ty in range(n_tiles_y)
                for tx in range(n_tiles_x)
                if (tx, ty) not in set(viewport_tiles)
            ]
        else:
            # No viewport info yet — treat everything as viewport
            viewport_tiles = [(tx, ty) for ty in range(n_tiles_y) for tx in range(n_tiles_x)]
            offscreen_tiles = []

        # Pass 1: viewport tiles across all slices — makes Z/T scrubbing instant
        def _process_pass1(tz):
            t, z = tz
            if cancel_event.is_set():
                return
            if (t, z, res) not in self._composite_cache:
                try:
                    self._get_composite_slice(t, z)
                except Exception:
                    pass
            for tx, ty in viewport_tiles:
                if cancel_event.is_set():
                    return
                if (t, z, tx, ty, res) not in self._tile_cache:
                    try:
                        self._get_tile(t, z, tx, ty)
                    except Exception:
                        pass

        with ThreadPoolExecutor(max_workers=4) as pool:
            for i, _ in enumerate(pool.map(_process_pass1, slices)):
                if cancel_event.is_set():
                    pool.shutdown(wait=False, cancel_futures=True)
                    return
                if (i + 1) % report_every == 0:
                    self._cache_progress = (i + 1) / total / 2  # Pass 1 = first half

        # Pass 2: off-screen tiles for all slices (background fill)
        def _process_pass2(tz):
            t, z = tz
            if cancel_event.is_set():
                return
            for tx, ty in offscreen_tiles:
                if cancel_event.is_set():
                    return
                if (t, z, tx, ty, res) not in self._tile_cache:
                    try:
                        self._get_tile(t, z, tx, ty)
                    except Exception:
                        pass

        with ThreadPoolExecutor(max_workers=4) as pool:
            for i, _ in enumerate(pool.map(_process_pass2, slices)):
                if cancel_event.is_set():
                    pool.shutdown(wait=False, cancel_futures=True)
                    return
                if (i + 1) % report_every == 0:
                    self._cache_progress = 0.5 + (i + 1) / total / 2  # Pass 2 = second half

        self._cache_progress = 1.0

    def _precompute_composites_remote(self, cancel_event, axis: str) -> None:
        """Precompute composites along one axis for remote zarr (no full-array in RAM).

        axis='t': all T at current_z — time-series playback mode
        axis='z': all Z at current_t — z-scan mode

        Fetches up to _REMOTE_PRECOMPUTE_WORKERS (t,z) pairs concurrently. Each pair
        itself uses parallel channel fetching, so total concurrent S3 GETs =
        workers × channels (e.g. 4 × 2 = 8).

        Slices are sorted nearest-first so the pool prioritises the most useful frames.
        """
        _REMOTE_PRECOMPUTE_WORKERS = 4

        if self._bioimage is None:
            return
        res = self.current_resolution

        if axis == "t":
            fixed_z = self.current_z
            candidates = sorted(range(self.dim_t), key=lambda t: abs(t - self.current_t))
            pairs = [(t, fixed_z) for t in candidates]
        else:
            fixed_t = self.current_t
            candidates = sorted(range(self.dim_z), key=lambda z: abs(z - self.current_z))
            pairs = [(fixed_t, z) for z in candidates]

        def _fetch_one(tz):
            if cancel_event.is_set():
                return
            t, z = tz
            if (t, z, res) not in self._composite_cache:
                try:
                    self._get_composite_slice(t, z)
                except Exception:
                    pass

        with ThreadPoolExecutor(max_workers=_REMOTE_PRECOMPUTE_WORKERS) as pool:
            for _ in pool.map(_fetch_one, pairs):
                if cancel_event.is_set():
                    pool.shutdown(wait=False, cancel_futures=True)
                    return

    def _start_precompute(self, axis: str = "t") -> None:
        """Cancel any running precompute task and start a fresh one."""
        if getattr(self, "_precompute_event", None) is not None:
            self._precompute_event.set()

        self._cache_progress = 0.0
        event = threading.Event()
        self._precompute_event = event

        if self._full_array is not None:
            # In-RAM path: full T×Z tile precompute
            self._precompute_future = self._prefetch_executor.submit(
                self._precompute_all_composites, event
            )
        elif self._bioimage is not None:
            # Remote path: composite-only, one axis at a time
            self._precompute_future = self._prefetch_executor.submit(
                self._precompute_composites_remote, event, axis
            )

    def _prefetch_adjacent_slices(self) -> None:
        """Pre-fetch adjacent Z and T slices in background for smoother scrubbing.

        For remote zarr, extends look-ahead to ±3 along the active axis so that
        navigation can outpace the background precompute thread.
        For in-RAM datasets, ±1 on both axes is sufficient (precompute covers the rest).
        """
        if self._bioimage is None:
            return

        t, z = self.current_t, self.current_z
        visible_channels = [i for i, ch in enumerate(self._channel_settings) if ch.get("visible", True)]

        # Wider look-ahead for remote zarr; ±1 suffices when data is in RAM
        is_remote = self._full_array is None
        axis = getattr(self, "_last_precompute_axis", "both")
        lookahead = 3 if is_remote else 1
        t_deltas = list(range(-lookahead, lookahead + 1)) if axis in ("t", "both") else [-1, 1]
        z_deltas = list(range(-lookahead, lookahead + 1)) if axis in ("z", "both") else [-1, 1]
        t_deltas = [d for d in t_deltas if d != 0]
        z_deltas = [d for d in z_deltas if d != 0]

        for delta in z_deltas:
            if 0 <= z + delta < self.dim_z:
                for c in visible_channels:
                    self._prefetch_executor.submit(self._prefetch_slice, t, c, z + delta)
                if not is_remote:
                    self._prefetch_executor.submit(self._prefetch_tiles_for_slice, t, z + delta)
        for delta in t_deltas:
            if 0 <= t + delta < self.dim_t:
                for c in visible_channels:
                    self._prefetch_executor.submit(self._prefetch_slice, t + delta, c, z)
                if not is_remote:
                    self._prefetch_executor.submit(self._prefetch_tiles_for_slice, t + delta, z)

    def _viewport_tiles_all_cached(self, t: int, z: int) -> bool:
        """Return True if every viewport tile for (t, z) is already in the Python tile cache."""
        vp = self._get_viewport_tile_ranges()
        if not vp:
            return False
        tx0, ty0, tx1, ty1 = vp
        res = self.current_resolution
        n_tiles_x = (self.width + self._tile_size - 1) // self._tile_size
        n_tiles_y = (self.height + self._tile_size - 1) // self._tile_size
        for ty in range(max(0, ty0), min(n_tiles_y, ty1)):
            for tx in range(max(0, tx0), min(n_tiles_x, tx1)):
                if (t, z, tx, ty, res) not in self._tile_cache:
                    return False
        return True

    def _update_slice(self):
        """Update the displayed slice based on current T, Z positions and channel settings."""
        if self._bioimage is None:
            return

        try:
            t, z = self.current_t, self.current_z
            use_tile_mode = self._use_tile_mode
            preview_mode = self._preview_mode

            if use_tile_mode:
                # Skip thumbnail update if all viewport tiles are already in Python cache.
                # JS tileCache was pre-populated by precompute, so renderCanvas() will draw
                # them immediately with no round-trip — no thumbnail flash needed.
                if self._viewport_tiles_all_cached(t, z):
                    return

                # Composite is cached; thumbnail is the only new data needed
                composite = self._get_composite_slice(t, z)
                if composite is not None:
                    self._image_array = np.mean(composite, axis=2).astype(np.uint8)
                    self.image_data = array_to_fast_png_base64(_thumbnail(composite))
                elif not preview_mode:
                    self.image_data = ""
            else:
                # Non-tile mode: compose channels manually for non-tile rendering path
                visible_channels, colors, mins, maxs, data_mins, data_maxs = [], [], [], [], [], []

                for i, ch_settings in enumerate(self._channel_settings):
                    if ch_settings.get("visible", True):
                        cache_key = (t, i, z)
                        if cache_key in self._slice_cache:
                            ch_data = self._slice_cache[cache_key]
                        elif preview_mode:
                            self._prefetch_executor.submit(self._prefetch_slice, t, i, z)
                            continue
                        else:
                            ch_data = self._get_slice_cached(t, i, z)

                        visible_channels.append(ch_data)
                        colors.append(ch_settings.get("color", "#ffffff"))
                        mins.append(ch_settings.get("min", 0.0))
                        maxs.append(ch_settings.get("max", 1.0))
                        data_mins.append(ch_settings.get("data_min"))
                        data_maxs.append(ch_settings.get("data_max"))

                if not visible_channels:
                    if not preview_mode:
                        normalized = np.zeros((self.height, self.width), dtype=np.uint8)
                        self._image_array = normalized
                        self.image_data = array_to_base64(normalized)
                    return

                if len(visible_channels) == 1 and colors[0] == "#ffffff":
                    global_min = data_mins[0] if data_mins and data_mins[0] is not None else None
                    global_max = data_maxs[0] if data_maxs and data_maxs[0] is not None else None
                    normalized = normalize_image(visible_channels[0], global_min, global_max)
                    vmin, vmax = mins[0], maxs[0]
                    if vmin > 0 or vmax < 1:
                        normalized = normalized.astype(np.float64) / 255.0
                        normalized = np.clip((normalized - vmin) / (vmax - vmin + 1e-10), 0, 1)
                        normalized = (normalized * 255).astype(np.uint8)
                    self._image_array = normalized
                    self.image_data = array_to_base64(normalized)
                else:
                    composite = composite_channels(visible_channels, colors, mins, maxs, data_mins, data_maxs)
                    self._image_array = np.mean(composite, axis=2).astype(np.uint8)
                    self.image_data = array_to_base64(composite)

            # Pre-fetch adjacent slices only if background precompute hasn't covered them yet
            if not preview_mode and getattr(self, "_cache_progress", 0.0) < 1.0:
                self._prefetch_adjacent_slices()

        except Exception as e:
            print(f"Error updating slice: {e}")

    def _on_dimension_change(self, change):
        """Observer callback when T or Z dimension changes."""
        self._update_slice()
        # For remote zarr: restart precompute when the active axis changes
        if self._full_array is None and self._bioimage is not None:
            axis = "t" if change.get("name") == "current_t" else "z"
            if getattr(self, "_last_precompute_axis", None) != axis:
                self._last_precompute_axis = axis
                self._start_precompute(axis=axis)

    def _update_numpy_image(self):
        """Re-render a numpy array image using current channel settings (LUT/contrast)."""
        raw = getattr(self, "_raw_numpy_array", None)
        if raw is None:
            return
        settings = self._channel_settings
        if not settings:
            return
        ch = settings[0]
        color = ch.get("color", "#ffffff")
        vmin = ch.get("min", 0.0)
        vmax = ch.get("max", 1.0)
        data_min = ch.get("data_min")
        data_max = ch.get("data_max")

        composite = composite_channels(
            [raw], [color], [vmin], [vmax],
            [data_min], [data_max],
        )
        self._image_array = np.mean(composite, axis=2).astype(np.uint8)
        self.image_data = array_to_base64(composite)

    def _on_auto_contrast_request(self, change):
        """Compute 2nd-98th percentile contrast for requested channel(s)."""
        request = change.get("new")
        if not request:
            return

        t = request.get("t", self.current_t)
        z = request.get("z", self.current_z)
        channel = request.get("channel", -1)
        timestamp = request.get("timestamp")

        def compute_range(ch_idx):
            """Compute normalized percentile range for a single channel."""
            if self._bioimage is not None:
                ch_data = self._get_slice_cached(t, ch_idx, z)
            elif hasattr(self, "_raw_numpy_array") and self._raw_numpy_array is not None:
                ch_data = self._raw_numpy_array
            else:
                return {"min": 0.0, "max": 1.0}

            p2, p98 = np.percentile(ch_data, [2, 98])
            ch_settings = self._channel_settings[ch_idx]
            data_min = ch_settings.get("data_min", float(ch_data.min()))
            data_max = ch_settings.get("data_max", float(ch_data.max()))
            span = data_max - data_min
            if span > 0:
                norm_min = (float(p2) - data_min) / span
                norm_max = (float(p98) - data_min) / span
            else:
                norm_min, norm_max = 0.0, 1.0
            return {"min": max(0.0, norm_min), "max": min(1.0, norm_max)}

        if channel == -1:
            ranges = {}
            for i in range(len(self._channel_settings)):
                ranges[str(i)] = compute_range(i)
            self._auto_contrast_result = {"channel": -1, "ranges": ranges, "timestamp": timestamp}
        else:
            result = compute_range(channel)
            self._auto_contrast_result = {
                "channel": channel,
                "min": result["min"],
                "max": result["max"],
                "timestamp": timestamp,
            }

    def _on_histogram_request(self, change):
        """Compute intensity histogram for requested channel(s)."""
        request = change.get("new")
        if not request:
            return

        t = request.get("t", self.current_t)
        z = request.get("z", self.current_z)
        channel = request.get("channel", -1)
        timestamp = request.get("timestamp")

        def compute_histogram(ch_idx):
            if self._bioimage is not None:
                ch_data = self._get_slice_cached(t, ch_idx, z)
            elif hasattr(self, "_raw_numpy_array") and self._raw_numpy_array is not None:
                ch_data = self._raw_numpy_array
            else:
                return [0] * 256

            ch_settings = self._channel_settings[ch_idx]
            data_min = ch_settings.get("data_min", float(ch_data.min()))
            data_max = ch_settings.get("data_max", float(ch_data.max()))
            counts, _ = np.histogram(ch_data.ravel(), bins=256, range=(data_min, data_max))
            return counts.tolist()

        if channel == -1:
            histograms = {}
            for i in range(len(self._channel_settings)):
                histograms[str(i)] = compute_histogram(i)
            self._histogram_data = {"channel": -1, "histograms": histograms, "timestamp": timestamp}
        else:
            counts = compute_histogram(channel)
            self._histogram_data = {"channel": channel, "counts": counts, "timestamp": timestamp}

    def _on_channel_settings_change(self, change):
        """Observer callback when channel settings change."""
        # For numpy arrays (no BioImage), re-render directly
        if self._bioimage is None:
            self._update_numpy_image()
            return

        if getattr(self, "_precompute_event", None) is not None:
            self._precompute_event.set()
        self._tile_cache.clear()
        composite_cache = getattr(self, "_composite_cache", None)
        if isinstance(composite_cache, dict):
            composite_cache.clear()
        self._update_slice()
        self._start_precompute()

    def _clear_caches(self, clear_full_array: bool = True):
        """Clear all caches. Pass clear_full_array=False to preserve the eager-loaded array."""
        self._slice_cache.clear()
        self._tile_cache.clear()
        composite_cache = getattr(self, "_composite_cache", None)
        if isinstance(composite_cache, dict):
            composite_cache.clear()
        if clear_full_array:
            self._full_array = None
        # Reset materialised synthetic pyramid levels (but keep level 0 = _full_array)
        pyramid = getattr(self, "_pyramid", None)
        if pyramid is not None and len(pyramid) > 1:
            pyramid[1:] = [None] * (len(pyramid) - 1)

    def _on_viewport_change(self, change):
        """Restart precompute when the user pans or zooms to a new viewport."""
        self._start_precompute()

    def _on_jpeg_toggle(self, change):
        """Clear tile cache and restart precompute when JPEG mode is toggled."""
        self._tile_cache.clear()
        self._update_slice()
        self._start_precompute()

    def _on_resolution_change(self, change):
        """Observer callback when resolution level changes."""
        level = change.get("new", 0)

        # Synthetic pyramid branch (flat TIFF/CZI/etc that have no native resolution levels)
        pyramid = getattr(self, "_pyramid", None)
        pyramid_has_native = getattr(self, "_pyramid_has_native", False)
        if pyramid is not None and not pyramid_has_native:
            try:
                arr = self._get_pyramid_level(level)
                self._clear_caches(clear_full_array=False)  # keep _full_array = level 0
                self.height = arr.shape[3]
                self.width = arr.shape[4]
                bytes_per_composite = self.height * self.width * 3
                max_by_memory = (512 * 1024 * 1024) // max(bytes_per_composite, 1)
                self._composite_cache_max_size = max(64, min(self.dim_t * self.dim_z, max_by_memory, 1024))
                self._update_slice()
                self._start_precompute()
            except Exception as e:
                print(f"Error changing synthetic resolution level: {e}")
            return

        if self._bioimage is None or not hasattr(self._bioimage, "set_resolution_level"):
            return

        try:
            self._bioimage.set_resolution_level(level)
            self._clear_caches()
            self.height = self._bioimage.dims.Y
            self.width = self._bioimage.dims.X
            bytes_per_composite = self.height * self.width * 3
            max_by_memory = (512 * 1024 * 1024) // max(bytes_per_composite, 1)
            self._composite_cache_max_size = max(64, min(self.dim_t * self.dim_z, max_by_memory, 1024))
            self._update_slice()
        except Exception as e:
            print(f"Error changing resolution level: {e}")

    def _on_scene_change(self, change):
        """Observer callback when scene changes."""
        new_scene = change.get("new", "")
        old_scene = change.get("old", "")
        # Skip if no actual scene change or during initial setup (old was empty)
        if self._bioimage is None or not new_scene or new_scene == old_scene or not old_scene:
            return
        if not hasattr(self._bioimage, "set_scene"):
            return

        try:
            self._bioimage.set_scene(new_scene)
            self._clear_caches()
            self.dim_t = self._bioimage.dims.T
            self.dim_c = self._bioimage.dims.C
            self.dim_z = self._bioimage.dims.Z
            self.height = self._bioimage.dims.Y
            self.width = self._bioimage.dims.X
            bytes_per_composite = self.height * self.width * 3
            max_by_memory = (512 * 1024 * 1024) // max(bytes_per_composite, 1)
            self._composite_cache_max_size = max(64, min(self.dim_t * self.dim_z, max_by_memory, 1024))
            self.current_t = 0
            self.current_c = 0
            self.current_z = 0
            self._update_slice()
        except Exception as e:
            print(f"Error changing scene: {e}")
