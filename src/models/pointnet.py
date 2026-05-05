import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """Spatial transformer network that predicts a k×k alignment matrix."""

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        self.fc3.bias.data = torch.eye(k).flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, k, N)
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.max(dim=-1)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        return x.view(B, self.k, self.k) + torch.eye(self.k, device=x.device).unsqueeze(0)


class PointNetEncoder(nn.Module):
    """
    PointNet encoder — extracts global and per-point features from a point cloud.

    Returns:
        global_feat:  (B, global_feat_dim)
        point_feat:   (B, 64, N)
        T_in:         (B, 3, 3) input transform, or None
        T_feat:       (B, 64, 64) feature transform, or None
    """

    def __init__(self, global_feat_dim: int = 1024, use_tnet: bool = True):
        super().__init__()
        self.use_tnet = use_tnet

        self.tnet3 = TNet(k=3) if use_tnet else None
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        self.tnet64 = TNet(k=64) if use_tnet else None
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, global_feat_dim, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(global_feat_dim)

    def forward(self, x: torch.Tensor):
        # x: (B, N, 3)
        x = x.transpose(1, 2)  # (B, 3, N)

        T_in = None
        if self.use_tnet:
            T_in = self.tnet3(x)
            x = torch.bmm(T_in, x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        T_feat = None
        if self.use_tnet:
            T_feat = self.tnet64(x)
            x = torch.bmm(T_feat, x)

        point_feat = x  # (B, 64, N)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))

        global_feat = x.max(dim=-1)[0]  # (B, global_feat_dim)
        return global_feat, point_feat, T_in, T_feat
