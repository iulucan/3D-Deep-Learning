import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet import PointNetEncoder


class PointNetClassifier(nn.Module):
    """Binary classifier: normal (0) vs anomalous (1) solar panel."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.encoder = PointNetEncoder(global_feat_dim=1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        feat, _, T_in, T_feat = self.encoder(x)
        x = F.relu(self.bn1(self.fc1(feat)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        logits = self.fc3(x)
        return logits, T_in, T_feat


class PointNetAutoEncoder(nn.Module):
    """
    Autoencoder for reconstruction-based anomaly detection.

    Anomaly score = Chamfer distance between input and reconstruction.
    Higher score indicates greater deviation from the learned normal geometry.
    """

    def __init__(self, num_points: int = 1024, latent_dim: int = 256):
        super().__init__()
        self.num_points = num_points
        self.encoder = PointNetEncoder(global_feat_dim=latent_dim, use_tnet=True)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, num_points * 3),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        global_feat, _, _, _ = self.encoder(x)
        return global_feat

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        return self.decoder(z).view(B, self.num_points, 3)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        recon, _ = self.forward(x)
        return chamfer_distance(x, recon)


def chamfer_distance(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """
    Mean Chamfer distance per sample.

    p1, p2: (B, N, 3)
    Returns: (B,) distance per sample
    """
    diff = p1.unsqueeze(2) - p2.unsqueeze(1)  # (B, N, N, 3)
    dist = diff.pow(2).sum(-1)                 # (B, N, N)
    d12 = dist.min(dim=2)[0].mean(dim=1)
    d21 = dist.min(dim=1)[0].mean(dim=1)
    return (d12 + d21) / 2


def tnet_regularization_loss(T: torch.Tensor) -> torch.Tensor:
    """Orthogonality regularizer for feature transform matrices."""
    if T is None:
        return torch.tensor(0.0)
    k = T.size(1)
    I = torch.eye(k, device=T.device).unsqueeze(0)
    diff = torch.bmm(T, T.transpose(1, 2)) - I
    return diff.pow(2).mean()
