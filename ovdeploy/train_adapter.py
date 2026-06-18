"""Train EpisodeAdapter on train-split episodes (proxy features)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ovdeploy.adapter import EpisodeAdapter, load_adapter, save_adapter
from ovdeploy.episode import load_episodes_dir
from ovdeploy.generator import image_gt_cat_ids
from ovdeploy.paths_util import load_lvis_minival, load_paths


class EpisodeFeatureDataset(Dataset):
    """Proxy: multi-hot GT over episode vocab from image annotations."""

    def __init__(self, episodes_dir: Path, feat_dim: int = 256, max_classes: int = 64):
        self.samples = []
        lvis = load_lvis_minival()
        img_cats = image_gt_cat_ids(lvis)
        rng = np.random.default_rng(42)

        for ep_path in sorted(episodes_dir.rglob("*.json"))[:500]:
            from ovdeploy.episode import load_episode

            ep = load_episode(ep_path)
            vocab = ep.vocab.cat_ids[:max_classes]
            if not vocab:
                continue
            for iid in ep.image_ids[:3]:
                gt = img_cats.get(iid, set())
                label = np.zeros(max_classes, dtype=np.float32)
                for j, cid in enumerate(vocab):
                    if cid in gt:
                        label[j] = 1.0
                feat = rng.standard_normal(feat_dim).astype(np.float32)
                self.samples.append((feat, label, len(vocab)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        feat, label, n = self.samples[idx]
        return torch.from_numpy(feat), torch.from_numpy(label), n


def train(
    episodes_dir: Path,
    out_ckpt: Path,
    epochs: int = 20,
    lr: float = 1e-3,
    feat_dim: int = 256,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    ds = EpisodeFeatureDataset(episodes_dir, feat_dim=feat_dim)
    if len(ds) < 10:
        return {"status": "skipped", "reason": "too_few_samples", "n": len(ds)}

    loader = DataLoader(ds, batch_size=32, shuffle=True)
    model = EpisodeAdapter(feat_dim=feat_dim, max_classes=64)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        model.train()
        for feat, label, n_cls in loader:
            n = int(n_cls[0].item()) if n_cls.dim() else int(n_cls)
            logits = model(feat, n)
            loss = bce(logits, label[:, :n])
            opt.zero_grad()
            loss.backward()
            opt.step()

    save_adapter(model, out_ckpt)
    return {"status": "ok", "n_samples": len(ds), "epochs": epochs, "ckpt": str(out_ckpt)}


def adapter_bias_for_episode(
    ckpt_path: Path, episode_vocab_size: int, feat_dim: int = 256
) -> np.ndarray:
    if not ckpt_path.is_file():
        return np.zeros(episode_vocab_size, dtype=np.float32)
    model = load_adapter(ckpt_path, feat_dim=feat_dim)
    model.eval()
    feat = torch.randn(1, feat_dim)
    with torch.no_grad():
        b = model(feat, min(episode_vocab_size, model.max_classes))
    return b.squeeze(0).numpy()[:episode_vocab_size]
