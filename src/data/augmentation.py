import numpy as np


class RandomRotationZ:
    """Rotate point cloud around the vertical (Z) axis."""

    def __call__(self, points: np.ndarray) -> np.ndarray:
        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        return points @ R.T


class RandomJitter:
    """Add Gaussian noise to simulate sensor measurement uncertainty."""

    def __init__(self, sigma: float = 0.01, clip: float = 0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, points: np.ndarray) -> np.ndarray:
        noise = np.clip(
            np.random.normal(0, self.sigma, points.shape).astype(np.float32),
            -self.clip,
            self.clip,
        )
        return points + noise


class RandomPointDropout:
    """Randomly replace points with the first point to simulate scan gaps."""

    def __init__(self, max_dropout_ratio: float = 0.875):
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, points: np.ndarray) -> np.ndarray:
        dropout_ratio = np.random.uniform(0, self.max_dropout_ratio)
        drop_idx = np.where(np.random.rand(len(points)) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            points = points.copy()
            points[drop_idx] = points[0]
        return points


class RandomScale:
    """Uniformly scale the point cloud."""

    def __init__(self, low: float = 0.8, high: float = 1.25):
        self.low = low
        self.high = high

    def __call__(self, points: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(self.low, self.high)
        return (points * scale).astype(np.float32)


class Compose:
    """Chain multiple augmentation transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, points: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            points = t(points)
        return points


def get_train_augmentation() -> Compose:
    return Compose([
        RandomRotationZ(),
        RandomJitter(sigma=0.01),
        RandomPointDropout(max_dropout_ratio=0.5),
        RandomScale(low=0.9, high=1.1),
    ])
