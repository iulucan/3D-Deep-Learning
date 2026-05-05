import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

from ..data.point_cloud_loader import SolarPanelPointCloudDataset
from ..data.augmentation import get_train_augmentation
from ..models.anomaly_detector import PointNetClassifier, tnet_regularization_loss


class SolarPanelTrainer:
    """Training pipeline for solar panel anomaly detection."""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config["epochs"]
        )
        self.best_auc = 0.0
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(exist_ok=True)

    def _build_model(self) -> nn.Module:
        model = PointNetClassifier(
            num_classes=2,
            dropout=self.config.get("dropout", 0.3),
        )
        return model.to(self.device)

    def _build_optimizer(self) -> optim.Optimizer:
        return optim.Adam(
            self.model.parameters(),
            lr=self.config.get("lr", 1e-3),
            weight_decay=self.config.get("weight_decay", 1e-4),
        )

    def _build_dataloaders(self):
        train_ds = SolarPanelPointCloudDataset(
            root=self.config["data_root"],
            num_points=self.config.get("num_points", 1024),
            split="train",
            transform=get_train_augmentation(),
        )
        val_ds = SolarPanelPointCloudDataset(
            root=self.config["data_root"],
            num_points=self.config.get("num_points", 1024),
            split="val",
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.get("batch_size", 32),
            shuffle=True,
            num_workers=4,
            pin_memory=self.device.type == "cuda",
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.get("batch_size", 32),
            shuffle=False,
            num_workers=4,
        )
        return train_loader, val_loader

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0

        for points, labels in loader:
            points, labels = points.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            logits, T_in, T_feat = self.model(points)
            loss = criterion(logits, labels)
            loss += 0.001 * tnet_regularization_loss(T_feat)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        all_probs, all_labels = [], []

        for points, labels in loader:
            points = points.to(self.device)
            logits, _, _ = self.model(points)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

        preds = (np.array(all_probs) >= 0.5).astype(int)
        auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
        f1 = f1_score(all_labels, preds, zero_division=0)
        return {"auc": auc, "f1": f1}

    def save_checkpoint(self, epoch: int, metrics: dict):
        path = self.checkpoint_dir / f"epoch_{epoch:03d}_auc{metrics['auc']:.4f}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }, path)

    def fit(self):
        train_loader, val_loader = self._build_dataloaders()
        for epoch in range(1, self.config["epochs"] + 1):
            train_loss = self.train_epoch(train_loader)
            metrics = self.evaluate(val_loader)
            self.scheduler.step()
            print(
                f"Epoch {epoch:03d} | loss={train_loss:.4f} "
                f"| AUC={metrics['auc']:.4f} | F1={metrics['f1']:.4f}"
            )
            if metrics["auc"] > self.best_auc:
                self.best_auc = metrics["auc"]
                self.save_checkpoint(epoch, metrics)
