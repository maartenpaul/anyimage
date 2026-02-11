"""Test fixtures for anyimage performance tests."""

import numpy as np
import pytest


class MockDims:
    """Mock BioImage dimensions object."""

    def __init__(self, t=1, c=1, z=1, y=512, x=512):
        self.T = t
        self.C = c
        self.Z = z
        self.Y = y
        self.X = x


class MockDaskArray:
    """Mock dask array that returns a numpy array on compute()."""

    def __init__(self, data: np.ndarray):
        self._data = data

    def compute(self) -> np.ndarray:
        return self._data


class MockBioImage:
    """Mock BioImage object for testing without real file I/O.

    Generates synthetic image data on demand with configurable dimensions,
    dtype, and optional latency simulation.
    """

    def __init__(
        self,
        t=1, c=1, z=1, y=512, x=512,
        dtype=np.uint16,
        latency_ms=0,
        seed=42,
    ):
        self.dims = MockDims(t=t, c=c, z=z, y=y, x=x)
        self._dtype = dtype
        self._latency_ms = latency_ms
        self._rng = np.random.RandomState(seed)
        self._data_cache = {}
        self.dask_data = True  # Signals this is a "BioImage-like" object
        self.scenes = []
        self.current_scene = ""

    def get_image_dask_data(self, dims_order: str, T=0, C=0, Z=0) -> MockDaskArray:
        """Return a mock dask array for the requested slice.

        Generates deterministic synthetic data based on (T, C, Z) coordinates.
        """
        if self._latency_ms > 0:
            import time
            time.sleep(self._latency_ms / 1000.0)

        cache_key = (T, C, Z)
        if cache_key not in self._data_cache:
            # Generate deterministic data per (T, C, Z)
            local_rng = np.random.RandomState(hash(cache_key) % (2**31))
            if self._dtype == np.uint8:
                data = local_rng.randint(0, 256, (self.dims.Y, self.dims.X), dtype=np.uint8)
            elif self._dtype == np.uint16:
                data = local_rng.randint(0, 65536, (self.dims.Y, self.dims.X), dtype=np.uint16)
            elif self._dtype == np.float32:
                data = local_rng.rand(self.dims.Y, self.dims.X).astype(np.float32) * 1000
            else:
                data = local_rng.randint(0, 65536, (self.dims.Y, self.dims.X)).astype(self._dtype)
            self._data_cache[cache_key] = data

        return MockDaskArray(self._data_cache[cache_key])


# ---- Fixtures ----

@pytest.fixture
def small_2d_image():
    """256x256 uint8 grayscale image."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (256, 256), dtype=np.uint8)


@pytest.fixture
def medium_2d_image():
    """1024x1024 uint16 grayscale image."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 65536, (1024, 1024), dtype=np.uint16)


@pytest.fixture
def large_2d_image():
    """4096x4096 uint16 grayscale image."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 65536, (4096, 4096), dtype=np.uint16)


@pytest.fixture
def mock_bioimage_small():
    """Small BioImage: 1T x 2C x 1Z x 512x512."""
    return MockBioImage(t=1, c=2, z=1, y=512, x=512)


@pytest.fixture
def mock_bioimage_multichannel():
    """Multi-channel BioImage: 1T x 4C x 1Z x 1024x1024."""
    return MockBioImage(t=1, c=4, z=1, y=1024, x=1024)


@pytest.fixture
def mock_bioimage_timelapse():
    """Time-lapse BioImage: 50T x 2C x 1Z x 512x512."""
    return MockBioImage(t=50, c=2, z=1, y=512, x=512)


@pytest.fixture
def mock_bioimage_zstack():
    """Z-stack BioImage: 1T x 2C x 30Z x 512x512."""
    return MockBioImage(t=1, c=2, z=30, y=512, x=512)


@pytest.fixture
def mock_bioimage_5d():
    """Full 5D BioImage: 10T x 3C x 20Z x 1024x1024."""
    return MockBioImage(t=10, c=3, z=20, y=1024, x=1024)


@pytest.fixture
def mock_bioimage_large():
    """Large BioImage: 1T x 2C x 1Z x 4096x4096."""
    return MockBioImage(t=1, c=2, z=1, y=4096, x=4096)


@pytest.fixture
def label_mask_small():
    """Small label mask (50 labels, 256x256)."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 51, (256, 256), dtype=np.int32)


@pytest.fixture
def label_mask_large():
    """Large label mask (500 labels, 1024x1024)."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 501, (1024, 1024), dtype=np.int32)


@pytest.fixture
def label_mask_many_labels():
    """Label mask with many labels (5000 labels, 512x512)."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 5001, (512, 512), dtype=np.int32)
