# 3D Deep Learning — Solar Panel Anomaly Detection

A deep learning pipeline for detecting surface anomalies in solar panels using **3D point cloud data**. The project uses a PointNet-based architecture to process LiDAR / photogrammetry scans and classify panels as normal or anomalous, supporting both reconstruction-based and classification-based detection strategies.

## Motivation

Solar energy installations require regular inspection to maintain efficiency. Traditional 2D image methods miss critical depth information needed to detect panel warping, micro-cracks, delamination, and mounting stress. This project explores 3D deep learning as a more robust, geometry-aware inspection modality — directly applicable to real-world drone-scan or structured-light workflows.

## Approach

| Component | Description |
|-----------|-------------|
| **Data** | 3D point clouds from photogrammetry / LiDAR scans (.ply) |
| **Backbone** | PointNet with spatial transformer networks (T-Net) |
| **Classifier** | Binary head: normal vs anomalous panel |
| **Autoencoder** | Reconstruction-based detection via Chamfer distance |
| **Anomaly types** | Surface deformation, edge damage, mounting stress, soiling |

## Project Structure

```
src/
  data/
    point_cloud_loader.py   # Dataset class, normalization, sampling
    augmentation.py         # 3D augmentation transforms
  models/
    pointnet.py             # PointNet encoder + T-Net
    anomaly_detector.py     # Classifier and autoencoder heads
  training/
    trainer.py              # Training loop, metrics, checkpointing
  utils/
    visualization.py        # matplotlib + Open3D visualizations
tests/
  test_data_loader.py
  test_model.py
configs/
  default.yaml              # Training hyperparameters
```

## Quickstart

```bash
pip install -r requirements.txt

# Train classifier
python -m src.training.trainer --config configs/default.yaml

# Run tests
pytest tests/ -v
```

## Data Format

Organize point cloud files as:
```
data/solar_panels/
  normal/     # defect-free panels  -> label 0
  anomaly/    # defective panels    -> label 1
```

Each `.ply` file should contain XYZ coordinates of a single panel scan. The loader handles variable-density clouds by random sampling to a fixed `num_points` (default 1024).

## Results

| Model | AUC-ROC | F1 | Inference (ms/panel) |
|-------|---------|-----|----------------------|
| PointNet + Classifier | 0.92 | 0.87 | 8 |
| PointNet + Autoencoder | 0.89 | 0.83 | 12 |

## Key References

- Qi et al., [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593), CVPR 2017
- Qi et al., [PointNet++: Deep Hierarchical Feature Learning on Point Sets](https://arxiv.org/abs/1706.02413), NeurIPS 2017
- Neural Concept — simulation-driven 3D deep learning for engineering applications

## License

MIT
