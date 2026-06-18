"""Train CalibHead on OVDeploy train episodes with real YOLO neck features."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vocabguard.calib_head import CalibHead, save_calib
from vocabguard.paths_util import checkpoints_dir, load_config


class RealFeatureDataset(Dataset):
    def __init__(
        self,
        episodes_dir: Path,
        backend,
        lvis: dict,
        feat_dim: int = 256,
        max_classes: int = 64,
        max_episodes: int = 500,
    ):
        from ovdeploy.episode import load_episode
        from ovdeploy.generator import image_gt_cat_ids
        from ovdeploy.paths_util import load_paths
        import cv2

        self.samples: list[tuple[np.ndarray, np.ndarray, int]] = []
        lvis_paths = load_paths()
        ann = lvis_paths["_yolo"] / lvis_paths["data"]["lvis_minival_ann"]
        import json

        lvis_local = json.loads(ann.read_text(encoding="utf-8"))
        id_to_im = {im["id"]: im for im in lvis_local["images"]}
        img_cats = image_gt_cat_ids(lvis)

        ep_files = sorted(episodes_dir.rglob("*.json"))[:max_episodes]
        for ep_path in ep_files:
            ep = load_episode(ep_path)
            vocab = ep.vocab.cat_ids[:max_classes]
            if not vocab:
                continue
            for iid in ep.image_ids[:3]:
                im = id_to_im.get(iid)
                if im is None:
                    continue
                path = backend.image_path(im["file_name"])
                image = cv2.imread(str(path))
                if image is None:
                    continue
                image_rgb = image[:, :, [2, 1, 0]]
                try:
                    feat = backend.encode_image(image_rgb, iid, lvis)
                except Exception:
                    continue
                label = np.zeros(max_classes, dtype=np.float32)
                gt = img_cats.get(iid, set())
                for j, cid in enumerate(vocab):
                    if cid in gt:
                        label[j] = 1.0
                n = len(vocab)
                self.samples.append((feat.astype(np.float32), label, n))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        feat, label, n = self.samples[idx]
        return torch.from_numpy(feat), torch.from_numpy(label), n


def train_calib(
    episodes_dir: Path,
    out_ckpt: Path,
    epochs: int = 20,
    lr: float = 1e-3,
    feat_dim: int = 256,
    hidden: int = 128,
    max_classes: int = 64,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    load_config()
    from ovdeploy.paths_util import load_lvis_minival
    from vocabguard.backend import get_extended_backend

    torch.manual_seed(seed)
    lvis = load_lvis_minival()
    backend = get_extended_backend("yolo", device=device)

    ds = RealFeatureDataset(
        episodes_dir, backend, lvis, feat_dim=feat_dim, max_classes=max_classes
    )
    if len(ds) < 10:
        return {"status": "skipped", "reason": "too_few_samples", "n": len(ds)}

    loader = DataLoader(ds, batch_size=16, shuffle=True)
    model = CalibHead(feat_dim=feat_dim, hidden=hidden, max_classes=max_classes)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        model.train()
        for feat, label, n_cls in loader:
            n = int(n_cls[0].item())
            feat = feat.to(device)
            label = label.to(device)
            logits = model(feat, n)
            loss = bce(logits, label[:, :n])
            opt.zero_grad()
            loss.backward()
            opt.step()

    save_calib(model, out_ckpt)
    return {"status": "ok", "n_samples": len(ds), "epochs": epochs, "ckpt": str(out_ckpt)}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    cfg = load_config()
    train_dir = cfg["_ovdeploy"] / "data/episodes/train"
    out = Path(args.out) if args.out else checkpoints_dir() / "calib_head_s42.pt"
    dev = "cuda:0" if args.gpu else "cpu"
    try:
        import torch

        if args.gpu and not torch.cuda.is_available():
            dev = "cpu"
    except ImportError:
        dev = "cpu"

    result = train_calib(train_dir, out, epochs=args.epochs, device=dev)
    print(result)

    from datetime import datetime, timezone

    report = {
        **result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu_used": dev.startswith("cuda"),
    }
    rep_path = ROOT / "reports" / "REPORT_VG_calib_train.json"
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {rep_path}")


if __name__ == "__main__":
    main()
