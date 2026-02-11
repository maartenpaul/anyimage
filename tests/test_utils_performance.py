"""Performance tests for anyimage utility functions.

Tests correctness and measures performance of normalize_image,
composite_channels, labels_to_rgba, and array_to_base64 across
various image sizes and configurations.
"""

import time

import numpy as np
import pytest

from anyimage.utils import (
    array_to_base64,
    composite_channels,
    hex_to_rgb,
    labels_to_rgba,
    normalize_image,
)

# ---- hex_to_rgb ----

class TestHexToRgb:
    def test_basic_colors(self):
        assert hex_to_rgb("#ff0000") == (255, 0, 0)
        assert hex_to_rgb("#00ff00") == (0, 255, 0)
        assert hex_to_rgb("#0000ff") == (0, 0, 255)
        assert hex_to_rgb("#ffffff") == (255, 255, 255)
        assert hex_to_rgb("#000000") == (0, 0, 0)

    def test_without_hash(self):
        assert hex_to_rgb("ff0000") == (255, 0, 0)

    def test_caching(self):
        """Repeated calls should be fast due to caching."""
        # Warm up
        hex_to_rgb("#abcdef")
        start = time.perf_counter()
        for _ in range(100_000):
            hex_to_rgb("#abcdef")
        elapsed_ms = (time.perf_counter() - start) * 1000
        # 100k cached lookups should complete well under 100ms
        assert elapsed_ms < 100, f"hex_to_rgb cached lookups took {elapsed_ms:.1f}ms for 100k calls"


# ---- normalize_image ----

class TestNormalizeImage:
    def test_uint8_passthrough(self):
        """uint8 data without global range should pass through unchanged."""
        data = np.array([[0, 128, 255]], dtype=np.uint8)
        result = normalize_image(data)
        np.testing.assert_array_equal(result, data)

    def test_uint16_normalization(self):
        """uint16 data should be normalized to 0-255."""
        data = np.array([[0, 32768, 65535]], dtype=np.uint16)
        result = normalize_image(data)
        assert result.dtype == np.uint8
        assert result[0, 0] == 0
        assert result[0, 2] == 255

    def test_float32_normalization(self):
        data = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        result = normalize_image(data)
        assert result.dtype == np.uint8
        assert result[0, 0] == 0
        assert result[0, 2] == 255

    def test_global_range(self):
        """Global min/max should override data range."""
        data = np.array([[100, 200]], dtype=np.uint16)
        result = normalize_image(data, global_min=0, global_max=1000)
        assert result.dtype == np.uint8
        # 100/1000 * 255 ≈ 25, 200/1000 * 255 ≈ 51
        assert 20 <= result[0, 0] <= 30
        assert 45 <= result[0, 1] <= 55

    def test_constant_image(self):
        """Constant image should produce zeros."""
        data = np.full((100, 100), 42, dtype=np.uint16)
        result = normalize_image(data)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, 0)

    def test_performance_small(self, small_2d_image):
        """256x256 normalization should be fast."""
        start = time.perf_counter()
        for _ in range(100):
            normalize_image(small_2d_image.astype(np.uint16), 0, 65535)
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / 100
        assert avg_ms < 5, f"normalize_image 256x256 avg: {avg_ms:.2f}ms"
        print(f"normalize_image 256x256: {avg_ms:.2f}ms avg")

    def test_performance_medium(self, medium_2d_image):
        """1024x1024 normalization."""
        start = time.perf_counter()
        for _ in range(20):
            normalize_image(medium_2d_image, 0, 65535)
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / 20
        assert avg_ms < 30, f"normalize_image 1024x1024 avg: {avg_ms:.2f}ms"
        print(f"normalize_image 1024x1024: {avg_ms:.2f}ms avg")

    def test_performance_large(self, large_2d_image):
        """4096x4096 normalization."""
        start = time.perf_counter()
        for _ in range(5):
            normalize_image(large_2d_image, 0, 65535)
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / 5
        assert avg_ms < 200, f"normalize_image 4096x4096 avg: {avg_ms:.2f}ms"
        print(f"normalize_image 4096x4096: {avg_ms:.2f}ms avg")


# ---- composite_channels ----

class TestCompositeChannels:
    def test_single_channel_white(self):
        """Single white channel should produce grayscale."""
        ch = np.array([[0, 128, 255]], dtype=np.uint8)
        result = composite_channels([ch], ["#ffffff"], [0.0], [1.0], [0.0], [255.0])
        assert result.dtype == np.uint8
        assert result.shape == (1, 3, 3)
        # White channel: R=G=B=value
        assert result[0, 0, 0] == 0  # R
        assert result[0, 2, 0] == 255  # R

    def test_two_channels(self):
        """Two-channel composite should blend additively."""
        green_ch = np.array([[255]], dtype=np.uint8)
        red_ch = np.array([[255]], dtype=np.uint8)
        result = composite_channels(
            [green_ch, red_ch],
            ["#00ff00", "#ff0000"],
            [0.0, 0.0], [1.0, 1.0],
            [0.0, 0.0], [255.0, 255.0],
        )
        # Green channel contributes G=255, Red channel contributes R=255
        assert result[0, 0, 0] == 255  # R from red channel
        assert result[0, 0, 1] == 255  # G from green channel
        assert result[0, 0, 2] == 0    # B = 0

    def test_contrast_limits(self):
        """Contrast limits should adjust output range."""
        ch = np.array([[128]], dtype=np.uint8)
        # With min=0, max=0.5, value 128/255 ≈ 0.502 is above max, should clip to 1.0
        result = composite_channels(
            [ch], ["#ffffff"],
            [0.0], [0.5],
            [0.0], [255.0],
        )
        assert result[0, 0, 0] == 255  # Clipped to max

    def test_empty_channels(self):
        """Empty channel list should return 1x1 black."""
        result = composite_channels([], [], [], [])
        assert result.shape == (1, 1, 3)
        np.testing.assert_array_equal(result, 0)

    def test_performance_2ch_512(self):
        """2 channels, 512x512."""
        rng = np.random.RandomState(42)
        ch1 = rng.randint(0, 65536, (512, 512), dtype=np.uint16)
        ch2 = rng.randint(0, 65536, (512, 512), dtype=np.uint16)
        start = time.perf_counter()
        for _ in range(20):
            composite_channels(
                [ch1, ch2], ["#00ff00", "#ff0000"],
                [0.0, 0.0], [1.0, 1.0],
                [0.0, 0.0], [65535.0, 65535.0],
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / 20
        assert avg_ms < 30, f"composite 2ch 512x512 avg: {avg_ms:.2f}ms"
        print(f"composite_channels 2ch 512x512: {avg_ms:.2f}ms avg")

    def test_performance_4ch_1024(self):
        """4 channels, 1024x1024."""
        rng = np.random.RandomState(42)
        channels = [rng.randint(0, 65536, (1024, 1024), dtype=np.uint16) for _ in range(4)]
        colors = ["#00ff00", "#ff0000", "#0000ff", "#ff00ff"]
        mins = [0.0] * 4
        maxs = [1.0] * 4
        data_mins = [0.0] * 4
        data_maxs = [65535.0] * 4
        start = time.perf_counter()
        for _ in range(10):
            composite_channels(channels, colors, mins, maxs, data_mins, data_maxs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / 10
        assert avg_ms < 200, f"composite 4ch 1024x1024 avg: {avg_ms:.2f}ms"
        print(f"composite_channels 4ch 1024x1024: {avg_ms:.2f}ms avg")

    def test_performance_3ch_4096(self):
        """3 channels, 4096x4096 (large image)."""
        rng = np.random.RandomState(42)
        channels = [rng.randint(0, 65536, (4096, 4096), dtype=np.uint16) for _ in range(3)]
        colors = ["#00ff00", "#ff0000", "#0000ff"]
        start = time.perf_counter()
        for _ in range(3):
            composite_channels(
                channels, colors,
                [0.0] * 3, [1.0] * 3,
                [0.0] * 3, [65535.0] * 3,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / 3
        assert avg_ms < 3000, f"composite 3ch 4096x4096 avg: {avg_ms:.2f}ms"
        print(f"composite_channels 3ch 4096x4096: {avg_ms:.2f}ms avg")


# ---- labels_to_rgba ----

class TestLabelsToRgba:
    def test_empty_labels(self):
        """All zeros should produce empty RGBA."""
        labels = np.zeros((100, 100), dtype=np.int32)
        result = labels_to_rgba(labels)
        assert result.shape == (100, 100, 4)
        np.testing.assert_array_equal(result[:, :, 3], 0)

    def test_single_label(self):
        """Single non-zero label should produce colored pixels."""
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[3:7, 3:7] = 1
        result = labels_to_rgba(labels)
        # Foreground should have alpha=255
        assert result[5, 5, 3] == 255
        # Background should have alpha=0
        assert result[0, 0, 3] == 0

    def test_multiple_labels_different_colors(self):
        """Different labels should get different colors."""
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[0:5, :] = 1
        labels[5:10, :] = 2
        result = labels_to_rgba(labels)
        # Both should have alpha=255
        assert result[2, 5, 3] == 255
        assert result[7, 5, 3] == 255
        # Colors should differ (at least one channel)
        color1 = tuple(result[2, 5, :3])
        color2 = tuple(result[7, 5, :3])
        assert color1 != color2

    def test_contours_only(self):
        """Contour mode should only color boundary pixels."""
        labels = np.zeros((20, 20), dtype=np.int32)
        labels[5:15, 5:15] = 1
        result = labels_to_rgba(labels, contours_only=True)
        # Interior pixel should be transparent
        assert result[10, 10, 3] == 0
        # Boundary pixel should be visible
        assert result[5, 5, 3] == 255 or result[5, 10, 3] == 255

    def test_performance_50_labels_256(self, label_mask_small):
        """50 labels, 256x256."""
        start = time.perf_counter()
        for _ in range(20):
            labels_to_rgba(label_mask_small)
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / 20
        assert avg_ms < 10, f"labels_to_rgba 50 labels 256x256 avg: {avg_ms:.2f}ms"
        print(f"labels_to_rgba 50 labels 256x256: {avg_ms:.2f}ms avg")

    def test_performance_500_labels_1024(self, label_mask_large):
        """500 labels, 1024x1024."""
        start = time.perf_counter()
        for _ in range(10):
            labels_to_rgba(label_mask_large)
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / 10
        assert avg_ms < 100, f"labels_to_rgba 500 labels 1024x1024 avg: {avg_ms:.2f}ms"
        print(f"labels_to_rgba 500 labels 1024x1024: {avg_ms:.2f}ms avg")

    def test_performance_5000_labels_512(self, label_mask_many_labels):
        """5000 labels, 512x512 - tests scalability with many labels."""
        start = time.perf_counter()
        for _ in range(10):
            labels_to_rgba(label_mask_many_labels)
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / 10
        assert avg_ms < 30, f"labels_to_rgba 5000 labels 512x512 avg: {avg_ms:.2f}ms"
        print(f"labels_to_rgba 5000 labels 512x512: {avg_ms:.2f}ms avg")

    def test_performance_contours_1024(self, label_mask_large):
        """Contour rendering performance 500 labels 1024x1024."""
        start = time.perf_counter()
        for _ in range(5):
            labels_to_rgba(label_mask_large, contours_only=True, contour_width=2)
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / 5
        assert avg_ms < 200, f"labels_to_rgba contours 1024x1024 avg: {avg_ms:.2f}ms"
        print(f"labels_to_rgba contours 500 labels 1024x1024: {avg_ms:.2f}ms avg")


# ---- array_to_base64 ----

class TestArrayToBase64:
    def test_grayscale(self):
        data = np.zeros((10, 10), dtype=np.uint8)
        result = array_to_base64(data)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_rgb(self):
        data = np.zeros((10, 10, 3), dtype=np.uint8)
        result = array_to_base64(data)
        assert isinstance(result, str)

    def test_rgba(self):
        data = np.zeros((10, 10, 4), dtype=np.uint8)
        result = array_to_base64(data)
        assert isinstance(result, str)

    def test_invalid_shape(self):
        data = np.zeros((10, 10, 5), dtype=np.uint8)
        with pytest.raises(ValueError):
            array_to_base64(data)

    def test_performance_512_rgb(self):
        """512x512 RGB to base64 PNG."""
        data = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        start = time.perf_counter()
        for _ in range(20):
            array_to_base64(data)
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / 20
        assert avg_ms < 50, f"array_to_base64 512x512 RGB avg: {avg_ms:.2f}ms"
        print(f"array_to_base64 512x512 RGB: {avg_ms:.2f}ms avg")

    def test_performance_1024_rgba(self):
        """1024x1024 RGBA to base64 PNG."""
        data = np.random.randint(0, 256, (1024, 1024, 4), dtype=np.uint8)
        start = time.perf_counter()
        for _ in range(5):
            array_to_base64(data)
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / 5
        assert avg_ms < 300, f"array_to_base64 1024x1024 RGBA avg: {avg_ms:.2f}ms"
        print(f"array_to_base64 1024x1024 RGBA: {avg_ms:.2f}ms avg")
