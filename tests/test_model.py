import pytest
import torch
from src.models.pointnet import PointNetEncoder, TNet
from src.models.anomaly_detector import (
    PointNetClassifier,
    PointNetAutoEncoder,
    chamfer_distance,
    tnet_regularization_loss,
)

BATCH = 4
N_POINTS = 1024


@pytest.fixture
def point_batch():
    return torch.randn(BATCH, N_POINTS, 3)


def test_tnet_output_shape(point_batch):
    tnet = TNet(k=3)
    x = point_batch.transpose(1, 2)
    T = tnet(x)
    assert T.shape == (BATCH, 3, 3)


def test_encoder_global_feat_shape(point_batch):
    encoder = PointNetEncoder(global_feat_dim=1024)
    global_feat, point_feat, T_in, T_feat = encoder(point_batch)
    assert global_feat.shape == (BATCH, 1024)
    assert point_feat.shape == (BATCH, 64, N_POINTS)


def test_classifier_output_shape(point_batch):
    model = PointNetClassifier(num_classes=2)
    logits, T_in, T_feat = model(point_batch)
    assert logits.shape == (BATCH, 2)
    assert not torch.isnan(logits).any()


def test_autoencoder_reconstruction_shape(point_batch):
    model = PointNetAutoEncoder(num_points=N_POINTS, latent_dim=256)
    recon, z = model(point_batch)
    assert recon.shape == point_batch.shape
    assert z.shape == (BATCH, 256)


def test_chamfer_symmetry(point_batch):
    p2 = torch.randn(BATCH, N_POINTS, 3)
    d12 = chamfer_distance(point_batch, p2)
    d21 = chamfer_distance(p2, point_batch)
    torch.testing.assert_close(d12, d21, atol=1e-4, rtol=1e-4)


def test_chamfer_zero_on_identical(point_batch):
    d = chamfer_distance(point_batch, point_batch)
    assert (d < 1e-5).all()


def test_tnet_regularization_near_zero_for_identity():
    T = torch.eye(3).unsqueeze(0).repeat(BATCH, 1, 1)
    loss = tnet_regularization_loss(T)
    assert loss.item() < 1e-5
