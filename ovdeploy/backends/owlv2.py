"""OWL-v2 backend via HuggingFace transformers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ovdeploy.paths_util import load_paths

_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any, str]] = {}


class Owlv2Backend:
    name = "owlv2"

    def __init__(
        self,
        device: str = "cuda:0",
        model_id: str | None = None,
        local_dir: str | None = None,
        name: str | None = None,
    ) -> None:
        self.device = device
        self.cfg = load_paths()
        owlv2_cfg = self.cfg.get("owlv2", {})
        self.model_id = model_id or owlv2_cfg.get(
            "model_id", "google/owlv2-base-patch16-ensemble"
        )
        if name:
            self.name = name
        # Prefer matching local dir (base vs large / ensemble vs non-ensemble).
        mid = str(self.model_id).lower()
        if local_dir is not None:
            local = local_dir
        elif "large" in mid:
            local = owlv2_cfg.get("local_dir_large") or owlv2_cfg.get("local_dir")
        elif "ensemble" not in mid:
            local = owlv2_cfg.get("local_dir_base_ne") or owlv2_cfg.get("local_dir")
        else:
            local = owlv2_cfg.get("local_dir")
        self.local_dir = None
        if local:
            p = Path(local)
            if not p.is_absolute():
                p = Path(self.cfg.get("_root", Path(__file__).resolve().parents[2])) / local
            if p.is_dir():
                self.local_dir = str(p)
        self.chunk_size = int(owlv2_cfg.get("chunk_size", 40))
        self.score_thresh = float(owlv2_cfg.get("score_thresh", 0.05))

    def _model_source(self) -> str:
        return self.local_dir or self.model_id

    def _init_model(self):
        src = self._model_source()
        key = (src, self.device)
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        local_only = bool(self.local_dir)
        processor = Owlv2Processor.from_pretrained(src, local_files_only=local_only)
        model = Owlv2ForObjectDetection.from_pretrained(src, local_files_only=local_only)
        dev = self.device if torch.cuda.is_available() else "cpu"
        model = model.to(dev).eval()
        _MODEL_CACHE[key] = (model, processor, dev)
        return model, processor, dev

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
        from PIL import Image
        import torch

        model, processor, dev = self._init_model()
        pil = Image.fromarray(image_rgb)
        h, w = image_rgb.shape[:2]
        labels = texts if texts else [str(t) for t in (class_names or vocab_ids)]
        all_preds: list[dict] = []
        chunk = max(1, self.chunk_size)
        for start in range(0, len(labels), chunk):
            chunk_labels = labels[start : start + chunk]
            chunk_ids = vocab_ids[start : start + chunk]
            flat_texts = []
            for t in chunk_labels:
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
            label_idxs = results["labels"].cpu().numpy()
            for box, score, label_idx in zip(boxes, scores, label_idxs):
                li = int(label_idx)
                if li >= len(chunk_ids):
                    continue
                x1, y1, x2, y2 = [float(v) for v in box]
                cid = chunk_ids[li]
                all_preds.append({
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": float(score),
                    "cat_id": int(cid),
                })
        return all_preds
