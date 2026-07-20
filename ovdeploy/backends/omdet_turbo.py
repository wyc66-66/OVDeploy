"""OmDet-Turbo backend via HuggingFace transformers (fast open-vocab)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ovdeploy.paths_util import load_paths
from ovdeploy.vocab import subset_class_texts

_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any, str]] = {}


class OmDetTurboBackend:
    name = "omdet_turbo"

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device
        self.cfg = load_paths()
        oc = self.cfg.get("omdet_turbo") or {}
        self.model_id = oc.get("model_id", "omlab/omdet-turbo-swin-tiny-hf")
        local = oc.get("local_dir", "weights/omdet-turbo-swin-tiny-hf")
        self.local_dir = None
        if local:
            p = Path(local)
            if not p.is_absolute():
                p = Path(self.cfg.get("_root", Path(__file__).resolve().parents[2])) / local
            if p.is_dir() and any(p.iterdir()):
                self.local_dir = str(p)
        self.chunk_size = int(oc.get("chunk_size", 40))
        self.score_thresh = float(oc.get("score_thresh", 0.05))
        self.task = str(oc.get("task", "Detect."))

    def _model_source(self) -> str:
        return self.local_dir or self.model_id

    def _init_model(self):
        src = self._model_source()
        key = (src, self.device)
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        import torch
        from transformers import AutoProcessor, OmDetTurboForObjectDetection

        local_only = bool(self.local_dir)
        processor = AutoProcessor.from_pretrained(src, local_files_only=local_only)
        model = OmDetTurboForObjectDetection.from_pretrained(src, local_files_only=local_only)
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

        model, processor, dev = self._init_model()
        pil = Image.fromarray(image_rgb)
        h, w = image_rgb.shape[:2]
        all_preds: list[dict] = []
        chunk = max(1, self.chunk_size)

        for start in range(0, len(flat), chunk):
            chunk_labels = flat[start : start + chunk]
            chunk_ids = vocab_ids[start : start + chunk]
            chunk_names = sub_names[start : start + chunk]
            # HF OmDet-Turbo: processor(images, text=classes) — no task kw
            inputs = processor(
                images=pil,
                text=chunk_labels,
                return_tensors="pt",
            )
            inputs = {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            # OmDet-Turbo HF: score_threshold (not threshold); classes required
            results = None
            if hasattr(processor, "post_process_grounded_object_detection"):
                results = processor.post_process_grounded_object_detection(
                    outputs,
                    classes=chunk_labels,
                    score_threshold=self.score_thresh,
                    target_sizes=[(h, w)],
                )[0]
            elif hasattr(processor, "post_process_object_detection"):
                results = processor.post_process_object_detection(
                    outputs,
                    target_sizes=torch.tensor([[h, w]], device=dev),
                    threshold=self.score_thresh,
                )[0]

            if results is None:
                continue

            boxes = results.get("boxes")
            scores = results.get("scores")
            # HF OmDet uses "classes" (str names); other processors may use "labels"
            labels = results.get("classes", results.get("labels"))
            if boxes is None:
                continue
            if hasattr(boxes, "cpu"):
                boxes = boxes.cpu().numpy()
            if hasattr(scores, "cpu"):
                scores = scores.cpu().numpy()
            # labels may be class name strings or indices
            for i, box in enumerate(boxes):
                score = float(scores[i]) if scores is not None else 0.0
                if score < self.score_thresh:
                    continue
                li = None
                lab = labels[i] if labels is not None else None
                if hasattr(lab, "item"):
                    try:
                        lab = lab.item()
                    except Exception:
                        lab = str(lab)
                if isinstance(lab, (int, np.integer)):
                    li = int(lab)
                elif lab is not None:
                    lab_s = str(lab).lower().strip()
                    for j, name in enumerate(chunk_labels):
                        if name.lower().strip() == lab_s:
                            li = j
                            break
                if li is None or li < 0 or li >= len(chunk_ids):
                    continue
                x1, y1, x2, y2 = [float(v) for v in box]
                all_preds.append(
                    {
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": score,
                        "label_idx": start + li,
                        "category_id": chunk_ids[li],
                        "category_name": chunk_names[li] if li < len(chunk_names) else "",
                    }
                )
        return all_preds
