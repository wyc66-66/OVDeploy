"""CalibHead: lightweight per-episode bias on real detector features."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class CalibHead(nn.Module):
    """Maps global image feature -> bias over |V| classes in episode."""

    def __init__(self, feat_dim: int = 256, hidden: int = 128, max_classes: int = 64):
        super().__init__()
        self.max_classes = max_classes
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, max_classes),
        )

    def forward(self, feat: torch.Tensor, n_classes: int) -> torch.Tensor:
        b = self.net(feat)
        return b[:, :n_classes]


def save_calib(model: CalibHead, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "max_classes": model.max_classes},
        path,
    )


def load_calib(path: Path, feat_dim: int = 256, hidden: int = 128) -> CalibHead:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m = CalibHead(
        feat_dim=feat_dim,
        hidden=hidden,
        max_classes=int(ckpt.get("max_classes", 64)),
    )
    m.load_state_dict(ckpt["state_dict"])
    return m


def apply_calib_bias(
    preds: list[dict],
    bias: np.ndarray,
    vocab_ids: list[int],
) -> list[dict]:
    cid_to_idx = {c: i for i, c in enumerate(vocab_ids)}
    out = []
    for p in preds:
        q = dict(p)
        cid = p.get("category_id", -1)
        li = p.get("label_idx", cid_to_idx.get(cid, 0))
        idx = li if li < len(bias) else cid_to_idx.get(cid, 0)
        if idx < len(bias):
            q["score"] = float(p["score"]) + 0.05 * float(bias[idx])
        out.append(q)
    return out
