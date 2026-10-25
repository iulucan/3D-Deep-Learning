import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


class PointCloudNormalizer:
    """Normalize point cloud to unit sphere centered at origin."""

    def __call__(self, points: np.ndarray) -> np.ndarray:
        centroid = points.mean(axis=0)
        points = points - centroid
        scale = np.max(np.linalg.norm(points, axis=1))
        return points / (scale + 1e-8)


class SolarPanelPointCloudDataset(Dataset):
    """
    Dataset for solar panel point cloud anomaly detection.

    Expects .ply files organized as:
        root/
            normal/    *.ply   (label 0)
            anomaly/   *.ply   (label 1)

    Optionally, root/train.txt / root/val.txt can list relative file paths.
    """

    LABELS = {"normal": 0, "anomaly": 1}

    def __init__(
        self,
        root: str,
        num_points: int = 1024,
        split: str = "train",
        transform=None,
    ):
        self.root = Path(root)
        self.num_points = num_points
        self.split = split
        self.transform = transform
        self.normalizer = PointCloudNormalizer()
        self.samples = self._load_file_list()

    def _load_file_list(self) -> list:
        samples = []
        split_file = self.root / f"{self.split}.txt"
        if split_file.exists():
            with open(split_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    p = self.root / line
                    label_name = p.parent.name
                    samples.append((p, self.LABELS.get(label_name, 0)))
        else:
            for label_name, label_id in self.LABELS.items():
                folder = self.root / label_name
                if folder.exists():
                    for ply in sorted(folder.glob("*.ply")):
                        samples.append((ply, label_id))
        return samples

    def _load_ply(self, path: Path) -> np.ndarray:
        if not OPEN3D_AVAILABLE:
            raise ImportError("open3d is required to load .ply files")
        pcd = o3d.io.read_point_cloud(str(path))
        return np.asarray(pcd.points, dtype=np.float32)

    def _sample_points(self, points: np.ndarray) -> np.ndarray:
        n = len(points)
        if n >= self.num_points:
            idx = np.random.choice(n, self.num_points, replace=False)
        else:
            idx = np.random.choice(n, self.num_points, replace=True)
        return points[idx]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        points = self._load_ply(path)
        points = self._sample_points(points)
        points = self.normalizer(points)
        if self.transform:
            points = self.transform(points)
        return torch.from_numpy(points), label


def load_point_cloud_from_array(
    points: np.ndarray,
    num_points: int = 1024,
) -> torch.Tensor:
    """Prepare a raw numpy point cloud for model inference. Returns (1, N, 3)."""
    normalizer = PointCloudNormalizer()
    n = len(points)
    if n >= num_points:
        idx = np.random.choice(n, num_points, replace=False)
    else:
        idx = np.random.choice(n, num_points, replace=True)
    pts = points[idx].astype(np.float32)
    pts = normalizer(pts)
    return torch.from_numpy(pts).unsqueeze(0)
