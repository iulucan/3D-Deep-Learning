import pytest
import numpy as np
import torch

from src.data.point_cloud_loader import PointCloudNormalizer, load_point_cloud_from_array
from src.data.augmentation import (
    RandomRotationZ,
    RandomJitter,
    RandomPointDropout,
    RandomScale,
    Compose,
)


@pytest.fixture
def sample_points():
    rng = np.random.default_rng(42)
    return rng.random((512, 3)).astype(np.float32)


def test_normalizer_unit_sphere(sample_points):
    normalizer = PointCloudNormalizer()
    normalized = normalizer(sample_points)
    norms = np.linalg.norm(normalized, axis=1)
    assert norms.max() <= 1.0 + 1e-5


def test_normalizer_centering(sample_points):
    normalizer = PointCloudNormalizer()
    result = normalizer(sample_points)
    np.testing.assert_allclose(result.mean(axis=0), 0.0, atol=1e-5)


def test_load_from_array_shape(sample_points):
    tensor = load_point_cloud_from_array(sample_points, num_points=256)
    assert tensor.shape == (1, 256, 3)
    assert tensor.dtype == torch.float32


def test_rotation_z_preserves_z(sample_points):
    aug = RandomRotationZ()
    result = aug(sample_points)
    np.testing.assert_allclose(result[:, 2], sample_points[:, 2], atol=1e-5)


def test_jitter_bounded(sample_points):
    aug = RandomJitter(sigma=0.01, clip=0.05)
    result = aug(sample_points)
    assert np.abs(result - sample_points).max() <= 0.05 + 1e-5


def test_compose_pipeline_shape(sample_points):
    pipeline = Compose([
        RandomRotationZ(),
        RandomJitter(),
        RandomPointDropout(max_dropout_ratio=0.3),
        RandomScale(),
    ])
    result = pipeline(sample_points)
    assert result.shape == sample_points.shape
    assert result.dtype == np.float32
