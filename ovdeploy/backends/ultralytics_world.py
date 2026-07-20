"""Ultralytics YOLO-World v2 backend (yolov8{s,m,l,x}-worldv2)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ovdeploy.paths_util import load_paths
from ovdeploy.vocab import subset_class_texts

_MODEL_CACHE: dict[tuple[str, str], Any] = {}

_VARIANT_WEIGHT = {
    "s": "yolov8s-worldv2.pt",
    "m": "yolov8m-worldv2.pt",
    "l": "yolov8l-worldv2.pt",
    "x": "yolov8x-worldv2.pt",
}


class UltralyticsWorldBackend:
    name = "uyolo_s"

    def __init__(self, device: str = "cuda:0", variant: str = "s") -> None:
        self.device = device
        self.variant = (variant or "s").lower()
        if self.variant not in _VARIANT_WEIGHT:
            raise ValueError(f"Unknown uyolo variant: {variant}")
        self.name = f"uyolo_{self.variant}"
        self.cfg = load_paths()
        uy = self.cfg.get("ultralytics_world") or {}
        self.score_thresh = float(uy.get("score_thresh", 0.05))
        self.imgsz = int(uy.get("imgsz", 640))
        weight = uy.get(f"weight_{self.variant}") or _VARIANT_WEIGHT[self.variant]
        root = Path(self.cfg.get("_root", Path(__file__).resolve().parents[2]))
        p = Path(weight)
        if not p.is_absolute():
            local = root / "weights" / "ultralytics" / p.name
            if local.is_file() and local.stat().st_size > 1_000_000:
                self.weight = str(local)
            else:
                # Do NOT fall back to bare name (ultralytics would hit github.com).
                raise FileNotFoundError(
                    f"Ultralytics weight missing: {local}. "
                    f"Run: python scripts/download_ultralytics_world_weights.py"
                )
        else:
            if not p.is_file():
                raise FileNotFoundError(f"Ultralytics weight missing: {p}")
            self.weight = str(p)

    def _init_model(self):
        key = (self.weight, self.device)
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "Ultralytics YOLO-World requires ultralytics. "
                "Install: pip install ultralytics"
            ) from e
        import torch

        model = YOLO(self.weight)
        # Force CUDA when available
        if torch.cuda.is_available() and "cuda" in (self.device or ""):
            model.to(self.device)
        _MODEL_CACHE[key] = model
        return model

    def image_path(self, file_name: str) -> Path:
        from ovdeploy.paths_util import resolve_val2017_image

        return resolve_val2017_image(self.cfg, file_name)

    def predict(
        self,
        image_rgb: np.ndarray,
        texts: list[str],
        vocab_ids: list[int],
        image_id: int,
        class_names: list[str] | None = None,
        class_texts_raw: list | None = None,
        cid2idx: dict[int, int] | None = None,
    ) -> list[dict]:
        if class_texts_raw is not None and cid2idx is not None:
            texts_sub = subset_class_texts(class_texts_raw, vocab_ids, cid2idx)
            sub_names = [
                class_names[cid2idx[c]] if class_names and c in cid2idx else str(c)
                for c in vocab_ids
            ]
        else:
            texts_sub = texts
            sub_names = texts

        flat: list[str] = []
        for t in texts_sub:
            flat.append(t[0] if isinstance(t, list) and t else str(t))

        model = self._init_model()
        model.set_classes(flat)
        results = model.predict(
            source=image_rgb,
            conf=self.score_thresh,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )
        if not results:
            return []
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return []
        boxes = r0.boxes.xyxy.cpu().numpy()
        scores = r0.boxes.conf.cpu().numpy()
        labels = r0.boxes.cls.cpu().numpy().astype(int)
        preds: list[dict] = []
        for box, score, li in zip(boxes, scores, labels):
            li = int(li)
            if li < 0 or li >= len(vocab_ids):
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            preds.append(
                {
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": float(score),
                    "label_idx": li,
                    "category_id": vocab_ids[li],
                    "category_name": sub_names[li] if li < len(sub_names) else "",
                }
            )
        return preds
