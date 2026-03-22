"""Tests for anyimage utility functions."""

import numpy as np
import pytest

from anyimage.utils import (
    array_to_base64,
    composite_channels,
    hex_to_rgb,
    normalize_image,
)


class TestHexToRgb:
    def test_red(self):
        assert hex_to_rgb("#ff0000") == (255, 0, 0)

    def test_green(self):
        assert hex_to_rgb("#00ff00") == (0, 255, 0)

    def test_blue(self):
        assert hex_to_rgb("#0000ff") == (0, 0, 255)

    def test_without_hash(self):
        assert hex_to_rgb("ffffff") == (255, 255, 255)

    def test_black(self):
        assert hex_to_rgb("#000000") == (0, 0, 0)


class TestNormalizeImage:
    def test_uint8_passthrough(self):
        arr = np.array([[0, 128, 255]], dtype=np.uint8)
        result = normalize_image(arr)
        np.testing.assert_array_equal(result, arr)

    def test_uint16_to_uint8(self):
        arr = np.array([[0, 32768, 65535]], dtype=np.uint16)
        result = normalize_image(arr)
        assert result.dtype == np.uint8
        assert result[0, 0] == 0
        assert result[0, 2] == 255

    def test_float_array(self):
        arr = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        result = normalize_image(arr)
        assert result.dtype == np.uint8
        assert result[0, 0] == 0
        assert result[0, 2] == 255

    def test_constant_array(self):
        arr = np.full((4, 4), 42, dtype=np.uint16)
        result = normalize_image(arr)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, np.zeros((4, 4), dtype=np.uint8))

    def test_global_range(self):
        arr = np.array([[0, 100]], dtype=np.float32)
        result = normalize_image(arr, global_min=0.0, global_max=200.0)
        assert result[0, 0] == 0
        assert result[0, 1] == 127  # 100/200 * 255 ≈ 127


class TestArrayToBase64:
    def test_grayscale(self):
        arr = np.zeros((8, 8), dtype=np.uint8)
        result = array_to_base64(arr)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_rgb(self):
        arr = np.zeros((8, 8, 3), dtype=np.uint8)
        result = array_to_base64(arr)
        assert isinstance(result, str)

    def test_rgba(self):
        arr = np.zeros((8, 8, 4), dtype=np.uint8)
        result = array_to_base64(arr)
        assert isinstance(result, str)

    def test_unsupported_shape_raises(self):
        arr = np.zeros((8, 8, 2), dtype=np.uint8)
        with pytest.raises(ValueError):
            array_to_base64(arr)


class TestCompositeChannels:
    def test_single_channel_red(self):
        # data range must include the pixel values for the LUT to map correctly
        ch = np.full((4, 4), 255, dtype=np.uint8)
        result = composite_channels(
            [ch], ["#ff0000"], [0.0], [1.0],
            data_mins=[0.0], data_maxs=[255.0],
        )
        assert result.shape == (4, 4, 3)
        assert result[0, 0, 0] == 255  # red
        assert result[0, 0, 1] == 0    # green
        assert result[0, 0, 2] == 0    # blue

    def test_two_channels_composite(self):
        red_ch = np.full((4, 4), 255, dtype=np.uint8)
        green_ch = np.full((4, 4), 255, dtype=np.uint8)
        result = composite_channels(
            [red_ch, green_ch], ["#ff0000", "#00ff00"], [0.0, 0.0], [1.0, 1.0],
            data_mins=[0.0, 0.0], data_maxs=[255.0, 255.0],
        )
        assert result.shape == (4, 4, 3)
        assert result[0, 0, 0] > 0   # red channel contributed
        assert result[0, 0, 1] > 0   # green channel contributed

    def test_empty_channels(self):
        result = composite_channels([], [], [], [])
        assert result.shape == (1, 1, 3)

    def test_output_dtype(self):
        ch = np.zeros((4, 4), dtype=np.uint8)
        result = composite_channels([ch], ["#ffffff"], [0.0], [1.0])
        assert result.dtype == np.uint8


class TestBioImageViewer:
    def test_import(self):
        from anyimage import BioImageViewer
        assert BioImageViewer is not None

    def test_instantiate(self):
        from anyimage import BioImageViewer
        viewer = BioImageViewer()
        assert viewer is not None

    def test_set_image_numpy(self):
        from anyimage import BioImageViewer
        viewer = BioImageViewer()
        arr = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        viewer.set_image(arr)
        assert viewer.width == 64
        assert viewer.height == 64

    def test_set_image_numpy_always_2d(self):
        from anyimage import BioImageViewer
        viewer = BioImageViewer()
        # Raw numpy arrays are squeezed to 2D — use BioImage for 5D support
        arr = np.random.randint(0, 255, (2, 3, 1, 32, 32), dtype=np.uint8)
        viewer.set_image(arr)
        # Squeezed to 2D, so dims reset to 1
        assert viewer.dim_t == 1
        assert viewer.dim_c == 1
        assert viewer.dim_z == 1

    def test_add_and_clear_mask(self):
        from anyimage import BioImageViewer
        viewer = BioImageViewer()
        viewer.set_image(np.zeros((32, 32), dtype=np.uint8))
        labels = np.zeros((32, 32), dtype=np.int32)
        labels[8:16, 8:16] = 1
        mask_id = viewer.add_mask(labels, name="Test", color="#ff0000")
        assert mask_id in viewer.get_mask_ids()
        viewer.clear_masks()
        assert len(viewer.get_mask_ids()) == 0

    def test_annotations_dataframes(self):
        from anyimage import BioImageViewer
        viewer = BioImageViewer()
        viewer.set_image(np.zeros((32, 32), dtype=np.uint8))
        assert hasattr(viewer, "rois_df")
        assert hasattr(viewer, "polygons_df")
        assert hasattr(viewer, "points_df")
