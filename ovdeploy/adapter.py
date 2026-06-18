"""EpisodeAdapter: per-episode class bias on frozen features."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class EpisodeAdapter(nn.Module):
    """Maps global image feature -> bias over |V| classes in episode."""

    def __init__(self, feat_dim: int = 256, hidden: int = 128, max_classes: int = 256):
        super().__init__()
        self.max_classes = max_classes
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, max_classes),
        )

    def forward(self, feat: torch.Tensor, n_classes: int) -> torch.Tensor:
        """feat: (B, D) -> bias (B, n_classes)."""
        b = self.net(feat)
        return b[:, :n_classes]


def save_adapter(model: EpisodeAdapter, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "max_classes": model.max_classes}, path)


def load_adapter(path: Path, feat_dim: int = 256) -> EpisodeAdapter:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m = EpisodeAdapter(feat_dim=feat_dim, max_classes=int(ckpt.get("max_classes", 256)))
    m.load_state_dict(ckpt["state_dict"])
    return m
