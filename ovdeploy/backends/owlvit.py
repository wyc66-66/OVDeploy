"""OWL-ViT backend via HuggingFace transformers (chunked open-vocab inference)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ovdeploy.paths_util import load_paths, cat_id_to_index, load_class_texts
from ovdeploy.vocab import subset_class_texts

_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any, str]] = {}


def _box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter + 1e-6
    return inter / union


class OwlvitBackend:
    name = "owlvit"

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device
        self.cfg = load_paths()
        owlvit_cfg = self.cfg.get("owlvit") or self.cfg.get("glip", {})
        self.model_id = owlvit_cfg.get("model_id", "google/owlvit-base-patch32")
        local = owlvit_cfg.get("local_dir")
        self.local_dir = None
        if local:
            p = Path(local)
            if not p.is_absolute():
                p = Path(self.cfg.get("_root", Path(__file__).resolve().parents[2])) / local
            if p.is_dir():
                self.local_dir = str(p)
        self.chunk_size = int(owlvit_cfg.get("chunk_size", 40))
        self.score_thresh = float(owlvit_cfg.get("score_thresh", 0.05))
        self.yolo: Path = self.cfg["_yolo"]

    def _model_source(self) -> str:
        return self.local_dir or self.model_id

    def _init_model(self):
        key = (self._model_source(), self.device)
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        try:
            import torch
            from PIL import Image
            from transformers import OwlViTForObjectDetection, OwlViTProcessor
        except ImportError as e:
            raise RuntimeError(
                "OWL-ViT backend requires torch, transformers, pillow. "
                "Install: pip install transformers pillow"
            ) from e

        src = self._model_source()
        processor = OwlViTProcessor.from_pretrained(src)
        model = OwlViTForObjectDetection.from_pretrained(src)
        dev = self.device if torch.cuda.is_available() else "cpu"
        model = model.to(dev)
        model.eval()
        _MODEL_CACHE[key] = (model, processor, dev)
        return model, processor, dev

    def image_path(self, file_name: str) -> Path:
        fn = file_name.replace("\\", "/")
        cfg = self.cfg
        img_dir = self.yolo / cfg["data"]["val2017_dir"]
        coco_root = self.yolo / "data/coco"
        for candidate in (coco_root / fn, img_dir / Path(fn).name, img_dir / fn):
            if candidate.is_file():
                return candidate
        return img_dir / fn

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
        import torch
        from PIL import Image

        if class_texts_raw is not None and cid2idx is not None:
            texts_sub = subset_class_texts(class_texts_raw, vocab_ids, cid2idx)
            sub_names = [
                class_names[cid2idx[c]] if class_names and c in cid2idx else str(c)
                for c in vocab_ids
            ]
        else:
            texts_sub = texts
            sub_names = texts

        model, processor, dev = self._init_model()
        pil = Image.fromarray(image_rgb)
        h, w = image_rgb.shape[:2]
        all_preds: list[dict] = []

        for start in range(0, len(texts_sub), self.chunk_size):
            chunk_texts = texts_sub[start : start + self.chunk_size]
            chunk_ids = vocab_ids[start : start + self.chunk_size]
            chunk_names = sub_names[start : start + self.chunk_size]
            if not chunk_texts:
                continue
            flat_texts = []
            for t in chunk_texts:
                flat_texts.append(t[0] if isinstance(t, list) and t else str(t))
            inputs = processor(text=[flat_texts], images=pil, return_tensors="pt")
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            target_sizes = torch.tensor([[h, w]], device=dev)
            results = processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=self.score_thresh,
            )[0]
            boxes = results["boxes"].cpu().numpy()
            scores = results["scores"].cpu().numpy()
            labels = results["labels"].cpu().numpy()
            for box, score, label_idx in zip(boxes, scores, labels):
                li = int(label_idx)
                if li >= len(chunk_ids):
                    continue
                x1, y1, x2, y2 = [float(v) for v in box]
                bbox_xywh = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
                all_preds.append(
                    {
                        "bbox": bbox_xywh,
                        "score": float(score),
                        "label_idx": start + li,
                        "category_id": chunk_ids[li],
                        "category_name": chunk_names[li],
                    }
                )

        all_preds.sort(key=lambda x: x["score"], reverse=True)
        kept: list[dict] = []
        for p in all_preds:
            dup = False
            for k in kept:
                if k["category_id"] == p["category_id"] and _box_iou_xyxy(
                    np.array(k["bbox"]), np.array(p["bbox"])
                ) > 0.5:
                    dup = True
                    break
            if not dup:
                kept.append(p)
        return kept
