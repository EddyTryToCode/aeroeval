"""
Unit tests for AeroEval Robustness corruptions.
"""

import numpy as np
import pytest
from aeroeval.robustness.corruptions import (
    CORRUPTIONS,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_jpeg_compression,
    apply_low_light,
    apply_motion_blur,
    apply_occlusion,
    apply_overexposure,
    apply_resolution_degradation,
)


@pytest.fixture
def sample_image():
    # 100x100 RGB image
    return np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)


def test_corruptions_registry_complete():
    expected_keys = [
        "Gaussian Blur", "Motion Blur", "Low Light",
        "Overexposure", "Gaussian Noise", "JPEG Compression",
        "Occlusion", "Resolution Degrade"
    ]
    for k in expected_keys:
        assert k in CORRUPTIONS
        assert callable(CORRUPTIONS[k])


def test_gaussian_blur(sample_image):
    out = apply_gaussian_blur(sample_image, severity=2)
    assert out.shape == sample_image.shape
    assert out.dtype == np.uint8


def test_motion_blur(sample_image):
    out = apply_motion_blur(sample_image, severity=1)
    assert out.shape == sample_image.shape


def test_low_light(sample_image):
    out = apply_low_light(sample_image, severity=2)
    assert out.shape == sample_image.shape
    assert np.mean(out) <= np.mean(sample_image)


def test_overexposure(sample_image):
    out = apply_overexposure(sample_image, severity=2)
    assert out.shape == sample_image.shape
    assert np.mean(out) >= np.mean(sample_image)


def test_gaussian_noise(sample_image):
    out = apply_gaussian_noise(sample_image, severity=2)
    assert out.shape == sample_image.shape
    assert not np.array_equal(out, sample_image)


def test_jpeg_compression(sample_image):
    out = apply_jpeg_compression(sample_image, severity=3)
    assert out.shape == sample_image.shape


def test_occlusion(sample_image):
    out = apply_occlusion(sample_image, severity=2)
    assert out.shape == sample_image.shape
    # Some pixels should be black (0)
    assert np.any(out == 0)


def test_resolution_degradation(sample_image):
    out = apply_resolution_degradation(sample_image, severity=2)
    assert out.shape == sample_image.shape
