import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


def plot_point_cloud(
    points: np.ndarray,
    scores: Optional[np.ndarray] = None,
    title: str = "Point Cloud",
    save_path: Optional[str] = None,
):
    """
    3D scatter plot of a point cloud, optionally colored by per-point anomaly score.

    points: (N, 3)
    scores: (N,) values in [0, 1] — red = anomalous, green = normal
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if scores is not None:
        colors = plt.cm.RdYlGn_r(scores)
        sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)
        sm = plt.cm.ScalarMappable(cmap="RdYlGn_r")
        sm.set_array(scores)
        plt.colorbar(sm, ax=ax, label="Anomaly Score")
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="steelblue", s=1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def compare_panels(
    normal: np.ndarray,
    anomalous: np.ndarray,
    save_path: Optional[str] = None,
):
    """Side-by-side view of a normal and an anomalous panel point cloud."""
    fig = plt.figure(figsize=(16, 7))

    for i, (pts, label) in enumerate([(normal, "Normal"), (anomalous, "Anomalous")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        color = "steelblue" if label == "Normal" else "crimson"
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=1)
        ax.set_title(label, fontsize=14)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    plt.suptitle("Solar Panel Point Cloud Comparison", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def visualize_open3d(points: np.ndarray, scores: Optional[np.ndarray] = None):
    """Interactive Open3D visualization with optional anomaly score coloring."""
    if not OPEN3D_AVAILABLE:
        raise ImportError("open3d is required for interactive visualization")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if scores is not None:
        colors = plt.cm.RdYlGn_r(scores)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.paint_uniform_color([0.3, 0.5, 0.8])

    o3d.visualization.draw_geometries([pcd], window_name="Solar Panel Point Cloud")
