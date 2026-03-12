import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def solid_red_image() -> Image.Image:
    """A 64x64 solid red PIL image."""
    return Image.new("RGB", (64, 64), color=(255, 0, 0))


@pytest.fixture
def solid_blue_image() -> Image.Image:
    """A 64x64 solid blue PIL image."""
    return Image.new("RGB", (64, 64), color=(0, 0, 255))


@pytest.fixture
def solid_red_array(solid_red_image) -> np.ndarray:
    return np.array(solid_red_image)


@pytest.fixture
def solid_blue_array(solid_blue_image) -> np.ndarray:
    return np.array(solid_blue_image)


@pytest.fixture
def noise_image_a() -> Image.Image:
    """A 64x64 random noise PIL image (seeded)."""
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def noise_image_b() -> Image.Image:
    """A 64x64 different random noise PIL image (seeded)."""
    rng = np.random.default_rng(99)
    arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def noise_array_a(noise_image_a) -> np.ndarray:
    return np.array(noise_image_a)


@pytest.fixture
def noise_array_b(noise_image_b) -> np.ndarray:
    return np.array(noise_image_b)
