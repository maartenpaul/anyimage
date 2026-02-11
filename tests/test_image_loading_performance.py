"""Performance tests for image loading, caching, and tile generation.

Uses MockBioImage to test performance without real file I/O,
isolating Python-side computation bottlenecks.
"""

import time

import numpy as np

from anyimage.mixins.image_loading import LRUCache
from anyimage.profiling import Profiler

from .conftest import MockBioImage

# ---- LRUCache ----

class TestLRUCache:
    def test_basic_operations(self):
        cache = LRUCache(max_size=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        assert len(cache) == 3
        assert "a" in cache
        assert cache["a"] == 1

    def test_eviction(self):
        """Oldest item should be evicted when capacity exceeded."""
        cache = LRUCache(max_size=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        cache["d"] = 4  # Should evict "a"
        assert "a" not in cache
        assert "d" in cache
        assert len(cache) == 3

    def test_lru_order(self):
        """Accessing an item should make it most recently used."""
        cache = LRUCache(max_size=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        # Access "a" to make it recently used
        _ = cache["a"]
        cache["d"] = 4  # Should evict "b" (least recently used), not "a"
        assert "a" in cache
        assert "b" not in cache
        assert "d" in cache

    def test_update_existing(self):
        """Updating existing key should not increase size."""
        cache = LRUCache(max_size=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        cache["a"] = 10  # Update, not insert
        assert len(cache) == 3
        assert cache["a"] == 10

    def test_clear(self):
        cache = LRUCache(max_size=3)
        cache["a"] = 1
        cache["b"] = 2
        cache.clear()
        assert len(cache) == 0
        assert "a" not in cache

    def test_performance_insert_lookup(self):
        """Insert and lookup performance for cache-like workload."""
        cache = LRUCache(max_size=2048)
        # Insert 4096 items (forces 2048 evictions)
        start = time.perf_counter()
        for i in range(4096):
            cache[i] = f"value_{i}"
        elapsed_insert = (time.perf_counter() - start) * 1000

        # Lookup 2048 existing items
        start = time.perf_counter()
        for i in range(2048, 4096):
            _ = cache[i]
        elapsed_lookup = (time.perf_counter() - start) * 1000

        assert elapsed_insert < 50, f"4096 inserts: {elapsed_insert:.1f}ms"
        assert elapsed_lookup < 20, f"2048 lookups: {elapsed_lookup:.1f}ms"
        print(f"LRUCache: 4096 inserts in {elapsed_insert:.1f}ms, 2048 lookups in {elapsed_lookup:.1f}ms")


# ---- Helper to create a viewer-like object for mixin testing ----

def _make_viewer_state(bioimage):
    """Create a dict-like namespace mimicking BioImageViewer state for mixin methods.

    Returns a simple object with all attributes needed by ImageLoadingMixin.
    """
    from concurrent.futures import ThreadPoolExecutor

    from anyimage.mixins.image_loading import ImageLoadingMixin, LRUCache

    class TestViewer(ImageLoadingMixin):
        def __init__(self):
            self._bioimage = None
            self._slice_cache = LRUCache(max_size=128)
            self._tile_cache = LRUCache(max_size=2048)
            self._prefetch_executor = ThreadPoolExecutor(max_workers=4)
            self._image_array = None
            self._channel_settings = []
            self.dim_t = 1
            self.dim_c = 1
            self.dim_z = 1
            self.current_t = 0
            self.current_c = 0
            self.current_z = 0
            self.current_resolution = 0
            self.width = 0
            self.height = 0
            self.image_data = ""
            self._preview_mode = False
            self._tile_size = 256
            self._tiles_data = {}
            self.resolution_levels = []
            self.scenes = []
            self.current_scene = ""

    viewer = TestViewer()
    return viewer


# ---- Channel Range Computation ----

class TestChannelRangeComputation:
    def test_correctness_single_channel(self):
        """Channel range should cover actual data range."""
        bio = MockBioImage(t=1, c=1, z=1, y=100, x=100, dtype=np.uint16)
        viewer = _make_viewer_state(bio)
        viewer._bioimage = bio
        viewer.dim_t = bio.dims.T
        viewer.dim_c = bio.dims.C
        viewer.dim_z = bio.dims.Z
        viewer.height = bio.dims.Y
        viewer.width = bio.dims.X

        ranges = viewer._compute_channel_ranges(bio)
        assert 0 in ranges
        data_min, data_max = ranges[0]
        assert data_min < data_max
        # For uint16 random data, min should be near 0, max near 65535
        assert data_min < 1000
        assert data_max > 60000

    def test_multi_channel_ranges(self):
        """Each channel should have its own range."""
        bio = MockBioImage(t=1, c=3, z=1, y=100, x=100)
        viewer = _make_viewer_state(bio)
        viewer._bioimage = bio
        viewer.dim_t = bio.dims.T
        viewer.dim_c = bio.dims.C
        viewer.dim_z = bio.dims.Z
        viewer.height = bio.dims.Y
        viewer.width = bio.dims.X

        ranges = viewer._compute_channel_ranges(bio)
        assert len(ranges) == 3

    def test_caches_slices_during_range_computation(self):
        """Slices loaded during range computation should be cached."""
        bio = MockBioImage(t=3, c=2, z=5, y=256, x=256)
        viewer = _make_viewer_state(bio)
        viewer._bioimage = bio
        viewer.dim_t = bio.dims.T
        viewer.dim_c = bio.dims.C
        viewer.dim_z = bio.dims.Z
        viewer.height = bio.dims.Y
        viewer.width = bio.dims.X

        viewer._compute_channel_ranges(bio)
        # Should have cached slices for sampled T/Z positions
        assert len(viewer._slice_cache) > 0

    def test_performance_small(self, mock_bioimage_small):
        """Small BioImage range computation."""
        viewer = _make_viewer_state(mock_bioimage_small)
        viewer._bioimage = mock_bioimage_small
        viewer.dim_t = mock_bioimage_small.dims.T
        viewer.dim_c = mock_bioimage_small.dims.C
        viewer.dim_z = mock_bioimage_small.dims.Z
        viewer.height = mock_bioimage_small.dims.Y
        viewer.width = mock_bioimage_small.dims.X

        start = time.perf_counter()
        viewer._compute_channel_ranges(mock_bioimage_small)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 100, f"range computation small: {elapsed_ms:.1f}ms"
        print(f"channel_ranges 1T x 2C x 1Z x 512x512: {elapsed_ms:.1f}ms")

    def test_performance_5d(self, mock_bioimage_5d):
        """5D BioImage range computation (10T x 3C x 20Z)."""
        viewer = _make_viewer_state(mock_bioimage_5d)
        viewer._bioimage = mock_bioimage_5d
        viewer.dim_t = mock_bioimage_5d.dims.T
        viewer.dim_c = mock_bioimage_5d.dims.C
        viewer.dim_z = mock_bioimage_5d.dims.Z
        viewer.height = mock_bioimage_5d.dims.Y
        viewer.width = mock_bioimage_5d.dims.X

        start = time.perf_counter()
        viewer._compute_channel_ranges(mock_bioimage_5d)
        elapsed_ms = (time.perf_counter() - start) * 1000
        # 3 T samples x 3 Z samples x 3 channels = 27 slice loads
        assert elapsed_ms < 2000, f"range computation 5D: {elapsed_ms:.1f}ms"
        print(f"channel_ranges 10T x 3C x 20Z x 1024x1024: {elapsed_ms:.1f}ms")

    def test_performance_large_subsampling(self, mock_bioimage_large):
        """Large image should use subsampling for faster range computation."""
        viewer = _make_viewer_state(mock_bioimage_large)
        viewer._bioimage = mock_bioimage_large
        viewer.dim_t = mock_bioimage_large.dims.T
        viewer.dim_c = mock_bioimage_large.dims.C
        viewer.dim_z = mock_bioimage_large.dims.Z
        viewer.height = mock_bioimage_large.dims.Y
        viewer.width = mock_bioimage_large.dims.X

        start = time.perf_counter()
        viewer._compute_channel_ranges(mock_bioimage_large)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 2000, f"range computation large: {elapsed_ms:.1f}ms"
        print(f"channel_ranges 1T x 2C x 1Z x 4096x4096: {elapsed_ms:.1f}ms")


# ---- Slice Caching ----

class TestSliceCaching:
    def test_cache_hit(self):
        """Second access to same slice should be a cache hit."""
        bio = MockBioImage(t=1, c=1, z=1, y=256, x=256)
        viewer = _make_viewer_state(bio)
        viewer._bioimage = bio

        # First access: cache miss
        data1 = viewer._get_slice_cached(0, 0, 0)
        # Second access: cache hit
        data2 = viewer._get_slice_cached(0, 0, 0)
        np.testing.assert_array_equal(data1, data2)

    def test_cache_eviction_maintains_lru(self):
        """LRU eviction should remove least recently used slices."""
        bio = MockBioImage(t=10, c=1, z=1, y=64, x=64)
        viewer = _make_viewer_state(bio)
        viewer._bioimage = bio
        viewer._slice_cache = LRUCache(max_size=3)

        # Load slices t=0,1,2 (fills cache)
        viewer._get_slice_cached(0, 0, 0)
        viewer._get_slice_cached(1, 0, 0)
        viewer._get_slice_cached(2, 0, 0)

        # Access t=0 to make it recently used
        viewer._get_slice_cached(0, 0, 0)

        # Load t=3, should evict t=1 (least recently used)
        viewer._get_slice_cached(3, 0, 0)
        assert (0, 0, 0) in viewer._slice_cache
        assert (1, 0, 0) not in viewer._slice_cache
        assert (3, 0, 0) in viewer._slice_cache

    def test_performance_cache_hit(self):
        """Cache hits should be near-instant."""
        bio = MockBioImage(t=1, c=1, z=1, y=1024, x=1024)
        viewer = _make_viewer_state(bio)
        viewer._bioimage = bio

        # Warm up cache
        viewer._get_slice_cached(0, 0, 0)

        # Measure cache hit performance
        start = time.perf_counter()
        for _ in range(1000):
            viewer._get_slice_cached(0, 0, 0)
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_us = elapsed_ms * 1000 / 1000  # microseconds
        assert avg_us < 100, f"cache hit avg: {avg_us:.1f}us"
        print(f"slice_cache hit 1024x1024: {avg_us:.1f}us avg")


# ---- Tile Generation ----

class TestTileGeneration:
    def _setup_viewer_for_tiles(self, bio):
        """Set up a viewer with channel settings for tile generation."""
        from anyimage.utils import CHANNEL_COLORS

        viewer = _make_viewer_state(bio)
        viewer._bioimage = bio
        viewer.dim_t = bio.dims.T
        viewer.dim_c = bio.dims.C
        viewer.dim_z = bio.dims.Z
        viewer.height = bio.dims.Y
        viewer.width = bio.dims.X
        viewer._tile_size = 256

        # Initialize channel settings
        channel_settings = []
        for i in range(bio.dims.C):
            channel_settings.append({
                "name": f"Channel {i}",
                "color": CHANNEL_COLORS[i % len(CHANNEL_COLORS)],
                "visible": True,
                "min": 0.0,
                "max": 1.0,
                "data_min": 0.0,
                "data_max": 65535.0,
            })
        viewer._channel_settings = channel_settings
        return viewer

    def test_tile_correctness(self):
        """Generated tile should have correct dimensions and format."""
        bio = MockBioImage(t=1, c=1, z=1, y=512, x=512)
        viewer = self._setup_viewer_for_tiles(bio)

        tile = viewer._get_tile(0, 0, 0, 0)
        assert tile is not None
        assert tile["w"] == 256
        assert tile["h"] == 256
        assert isinstance(tile["data"], str)  # base64 string

    def test_tile_edge(self):
        """Edge tile should be smaller than tile_size."""
        bio = MockBioImage(t=1, c=1, z=1, y=300, x=300)
        viewer = self._setup_viewer_for_tiles(bio)

        # Tile at (1,1) should be 300-256=44 pixels
        tile = viewer._get_tile(0, 0, 1, 1)
        assert tile is not None
        assert tile["w"] == 44
        assert tile["h"] == 44

    def test_tile_out_of_bounds(self):
        """Tile outside image bounds should return None."""
        bio = MockBioImage(t=1, c=1, z=1, y=256, x=256)
        viewer = self._setup_viewer_for_tiles(bio)

        tile = viewer._get_tile(0, 0, 5, 5)  # Way out of bounds
        assert tile is None

    def test_tile_caching(self):
        """Second request for same tile should be cached."""
        bio = MockBioImage(t=1, c=1, z=1, y=512, x=512)
        viewer = self._setup_viewer_for_tiles(bio)

        tile1 = viewer._get_tile(0, 0, 0, 0)
        tile2 = viewer._get_tile(0, 0, 0, 0)
        # Should be the same object (cached)
        assert tile1 is tile2

    def test_performance_single_tile_1ch(self):
        """Single tile, 1 channel."""
        bio = MockBioImage(t=1, c=1, z=1, y=512, x=512)
        viewer = self._setup_viewer_for_tiles(bio)

        # Warm up (load slice into cache)
        viewer._get_tile(0, 0, 0, 0)
        viewer._tile_cache.clear()

        start = time.perf_counter()
        for _ in range(50):
            viewer._tile_cache.clear()
            viewer._get_tile(0, 0, 0, 0)
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / 50
        assert avg_ms < 10, f"tile gen 1ch avg: {avg_ms:.2f}ms"
        print(f"tile_gen 1ch 256x256: {avg_ms:.2f}ms avg")

    def test_performance_single_tile_4ch(self):
        """Single tile, 4 channels."""
        bio = MockBioImage(t=1, c=4, z=1, y=512, x=512)
        viewer = self._setup_viewer_for_tiles(bio)

        # Warm up slices
        for c in range(4):
            viewer._get_slice_cached(0, c, 0)
        viewer._tile_cache.clear()

        start = time.perf_counter()
        for _ in range(50):
            viewer._tile_cache.clear()
            viewer._get_tile(0, 0, 0, 0)
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / 50
        assert avg_ms < 20, f"tile gen 4ch avg: {avg_ms:.2f}ms"
        print(f"tile_gen 4ch 256x256: {avg_ms:.2f}ms avg")

    def test_performance_batch_tiles(self):
        """Batch of 16 tiles (4x4 grid), 2 channels."""
        bio = MockBioImage(t=1, c=2, z=1, y=1024, x=1024)
        viewer = self._setup_viewer_for_tiles(bio)

        # Warm up slices
        for c in range(2):
            viewer._get_slice_cached(0, c, 0)

        start = time.perf_counter()
        for _ in range(10):
            viewer._tile_cache.clear()
            for ty in range(4):
                for tx in range(4):
                    viewer._get_tile(0, 0, tx, ty)
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / 10
        assert avg_ms < 200, f"batch 16 tiles 2ch avg: {avg_ms:.2f}ms"
        print(f"tile_gen batch 16 tiles 2ch: {avg_ms:.2f}ms avg")


# ---- Profiler Integration ----

class TestProfilerIntegration:
    def test_profiler_records_operations(self):
        """Profiler should record operations when enabled."""
        profiler = Profiler()
        profiler.enable()

        bio = MockBioImage(t=1, c=2, z=1, y=256, x=256)
        viewer = _make_viewer_state(bio)
        viewer._bioimage = bio
        viewer.dim_t = bio.dims.T
        viewer.dim_c = bio.dims.C
        viewer.dim_z = bio.dims.Z
        viewer.height = bio.dims.Y
        viewer.width = bio.dims.X

        # The profiler singleton is used inside the mixin methods
        # Set it as the singleton temporarily
        old_instance = Profiler._instance
        Profiler._instance = profiler

        try:
            viewer._get_slice_cached(0, 0, 0)
            viewer._get_slice_cached(0, 0, 0)  # cache hit

            summary = profiler.summary_dict()
            caches = summary["caches"]
            assert "slice_cache" in caches
            assert caches["slice_cache"]["hits"] == 1
            assert caches["slice_cache"]["misses"] == 1
        finally:
            Profiler._instance = old_instance

    def test_profiler_report_format(self):
        """Profiler report should be a readable string."""
        profiler = Profiler()
        profiler.enable()

        with profiler.track("test_op"):
            time.sleep(0.001)

        profiler.record_cache_hit("test_cache")
        profiler.record_cache_miss("test_cache")
        profiler.update_cache_size("test_cache", 5, 100)

        report = profiler.report()
        assert "ANYIMAGE PERFORMANCE REPORT" in report
        assert "test_op" in report
        assert "test_cache" in report

    def test_profiler_disabled_no_overhead(self):
        """Disabled profiler should add minimal overhead."""
        profiler = Profiler()
        # profiler is disabled by default

        start = time.perf_counter()
        for _ in range(10_000):
            with profiler.track("noop"):
                pass
        elapsed_ms = (time.perf_counter() - start) * 1000
        # 10k context manager entries should be well under 50ms when disabled
        assert elapsed_ms < 50, f"disabled profiler overhead: {elapsed_ms:.1f}ms for 10k iterations"
        print(f"Profiler disabled overhead: {elapsed_ms:.1f}ms for 10k iterations")


# ---- Full Pipeline Benchmarks ----

class TestFullPipelineBenchmarks:
    """End-to-end benchmarks simulating real usage patterns."""

    def _setup_viewer_for_tiles(self, bio):
        from anyimage.utils import CHANNEL_COLORS

        viewer = _make_viewer_state(bio)
        viewer._bioimage = bio
        viewer.dim_t = bio.dims.T
        viewer.dim_c = bio.dims.C
        viewer.dim_z = bio.dims.Z
        viewer.height = bio.dims.Y
        viewer.width = bio.dims.X
        viewer._tile_size = 256

        channel_settings = []
        for i in range(bio.dims.C):
            channel_settings.append({
                "name": f"Channel {i}",
                "color": CHANNEL_COLORS[i % len(CHANNEL_COLORS)],
                "visible": True,
                "min": 0.0,
                "max": 1.0,
                "data_min": 0.0,
                "data_max": 65535.0,
            })
        viewer._channel_settings = channel_settings
        return viewer

    def test_timelapse_scrubbing(self, mock_bioimage_timelapse):
        """Simulate scrubbing through 50 timepoints (2ch 512x512).

        Measures: slice loading + tile generation for initial load,
        then cache performance for repeated access.
        """
        viewer = self._setup_viewer_for_tiles(mock_bioimage_timelapse)

        # First pass: cold cache (load slices from mock disk)
        start = time.perf_counter()
        for t in range(10):  # First 10 timepoints
            for c in range(2):
                viewer._get_slice_cached(t, c, 0)
        cold_ms = (time.perf_counter() - start) * 1000

        # Second pass: warm cache
        start = time.perf_counter()
        for t in range(10):
            for c in range(2):
                viewer._get_slice_cached(t, c, 0)
        warm_ms = (time.perf_counter() - start) * 1000

        assert warm_ms < cold_ms / 5, "Warm cache should be >5x faster than cold"
        print(f"Timelapse scrubbing: cold={cold_ms:.1f}ms, warm={warm_ms:.1f}ms "
              f"(speedup: {cold_ms/warm_ms:.1f}x)")

    def test_zstack_navigation(self, mock_bioimage_zstack):
        """Simulate navigating through 30 Z-slices (2ch 512x512)."""
        viewer = self._setup_viewer_for_tiles(mock_bioimage_zstack)

        # Navigate through all Z slices
        start = time.perf_counter()
        for z in range(30):
            for c in range(2):
                viewer._get_slice_cached(0, c, z)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # 60 slice loads (30 Z x 2 channels)
        avg_per_slice = elapsed_ms / 60
        assert elapsed_ms < 5000, f"Z-stack navigation: {elapsed_ms:.1f}ms"
        print(f"Z-stack 30 slices x 2ch: {elapsed_ms:.1f}ms total, {avg_per_slice:.1f}ms/slice")

    def test_viewport_tile_loading(self):
        """Simulate loading a 4x4 viewport of tiles for a large image."""
        bio = MockBioImage(t=1, c=3, z=1, y=2048, x=2048)
        viewer = self._setup_viewer_for_tiles(bio)

        # Warm up slice cache
        for c in range(3):
            viewer._get_slice_cached(0, c, 0)

        # Load 4x4 = 16 tiles (typical viewport)
        start = time.perf_counter()
        for ty in range(4):
            for tx in range(4):
                viewer._get_tile(0, 0, tx, ty)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 500, f"Viewport 4x4 tiles 3ch: {elapsed_ms:.1f}ms"
        print(f"Viewport 4x4 tiles 3ch 2048x2048: {elapsed_ms:.1f}ms")

    def test_channel_range_then_tiles(self, mock_bioimage_multichannel):
        """Full pipeline: compute ranges, then generate tiles."""
        viewer = self._setup_viewer_for_tiles(mock_bioimage_multichannel)

        # Step 1: Compute channel ranges
        start = time.perf_counter()
        ranges = viewer._compute_channel_ranges(mock_bioimage_multichannel)
        range_ms = (time.perf_counter() - start) * 1000

        # Update channel settings with computed ranges
        for i, ch in enumerate(viewer._channel_settings):
            if i in ranges:
                ch["data_min"] = ranges[i][0]
                ch["data_max"] = ranges[i][1]

        # Step 2: Generate tiles (slices may already be cached from range computation)
        start = time.perf_counter()
        for ty in range(4):
            for tx in range(4):
                viewer._get_tile(0, 0, tx, ty)
        tile_ms = (time.perf_counter() - start) * 1000

        total_ms = range_ms + tile_ms
        print(f"Full pipeline 4ch 1024x1024: ranges={range_ms:.1f}ms, "
              f"tiles={tile_ms:.1f}ms, total={total_ms:.1f}ms")
        assert total_ms < 3000, f"Full pipeline: {total_ms:.1f}ms"
