"""GLIP-T backend: native Microsoft GLIP (.pth) or Grounding-DINO transformers fallback."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ovdeploy.paths_util import load_paths, load_class_texts, cat_id_to_index
from ovdeploy.vocab import subset_class_texts

_MODEL_CACHE: dict[tuple[str, str], tuple[Any, ...]] = {}


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


def _flat_text(t: Any) -> str:
    if isinstance(t, list) and t:
        return str(t[0])
    return str(t)


def _grounding_caption(texts: list) -> str:
    parts = [_flat_text(t).strip().rstrip(".") for t in texts if _flat_text(t).strip()]
    if not parts:
        return "object."
    return ". ".join(parts) + "."


def _label_to_index(label: Any, chunk_texts: list, chunk_names: list) -> int | None:
    if isinstance(label, (int, np.integer)):
        li = int(label)
        return li if 0 <= li < len(chunk_texts) else None
    text = str(label).strip().lower()
    if not text:
        return None
    flat = [_flat_text(t).strip().lower() for t in chunk_texts]
    names = [str(n).strip().lower() for n in chunk_names]
    for i, (t, n) in enumerate(zip(flat, names)):
        for cand in (t, n):
            if not cand:
                continue
            if cand == text or cand in text or text in cand:
                return i
    return None


def _result_arrays(results: dict) -> tuple[np.ndarray, np.ndarray, list]:
    boxes = results["boxes"]
    scores = results["scores"]
    labels = results["labels"]
    if hasattr(boxes, "cpu"):
        boxes = boxes.cpu().numpy()
    else:
        boxes = np.asarray(boxes)
    if hasattr(scores, "cpu"):
        scores = scores.cpu().numpy()
    else:
        scores = np.asarray(scores)
    if hasattr(labels, "cpu"):
        labels = labels.cpu().numpy().tolist()
    elif not isinstance(labels, list):
        labels = list(labels)
    return boxes, scores, labels


class GlipBackend:
    name = "glip"

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device
        self.cfg = load_paths()
        glip_cfg = self.cfg.get("glip", {})
        self.backend = glip_cfg.get("backend", "transformers")
        self.model_id = glip_cfg.get(
            "glip_tiny_model_id", "IDEA-Research/grounding-dino-tiny"
        )
        local = glip_cfg.get("glip_tiny_local_dir")
        self.local_dir = None
        if local:
            p = Path(local)
            if not p.is_absolute():
                p = Path(self.cfg.get("_root", Path(__file__).resolve().parents[2])) / local
            if p.is_dir() and (p / "config.json").is_file():
                self.local_dir = str(p)
        native_w = glip_cfg.get("native_weight")
        self.native_weight = None
        if native_w:
            wp = Path(native_w)
            if not wp.is_absolute():
                wp = Path(self.cfg.get("_root", Path(__file__).resolve().parents[2])) / native_w
            if wp.is_file() and wp.stat().st_size > 1_000_000:
                self.native_weight = str(wp)
        self.chunk_size = int(glip_cfg.get("chunk_size", 40))
        self.score_thresh = float(glip_cfg.get("score_thresh", 0.05))
        self.yolo: Path = self.cfg["_yolo"]
        self._use_grounding_dino = False

    def _model_source(self) -> str:
        return self.local_dir or self.model_id

    def _init_transformers(self):
        src = self._model_source()
        key = ("transformers", src, self.device)
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]

        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        processor = AutoProcessor.from_pretrained(src)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(src)
        proc_name = type(processor).__name__
        if not hasattr(processor, "post_process_grounded_object_detection"):
            raise RuntimeError(
                f"GLIP backend needs grounded post-process; got {proc_name} from {src}. "
                "Install transformers>=4.45 and download IDEA-Research/grounding-dino-tiny."
            )
        self._use_grounding_dino = "GroundingDino" in proc_name
        dev = self.device if torch.cuda.is_available() else "cpu"
        model = model.to(dev)
        model.eval()
        _MODEL_CACHE[key] = (model, processor, dev, self._use_grounding_dino)
        return model, processor, dev, self._use_grounding_dino

    def _init_model(self):
        if self.backend == "native" and self.native_weight:
            raise RuntimeError(
                "Native GLIP backend not wired yet; set glip.backend=transformers "
                f"(weight present: {self.native_weight})"
            )
        return self._init_transformers()

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

        model, processor, dev, use_gdino = self._init_model()
        pil = Image.fromarray(image_rgb)
        h, w = image_rgb.shape[:2]
        all_preds: list[dict] = []

        for start in range(0, len(texts_sub), self.chunk_size):
            chunk_texts = texts_sub[start : start + self.chunk_size]
            chunk_ids = vocab_ids[start : start + self.chunk_size]
            chunk_names = sub_names[start : start + self.chunk_size]
            if not chunk_texts:
                continue

            if use_gdino:
                caption = _grounding_caption(chunk_texts)
                inputs = processor(images=pil, text=caption, return_tensors="pt")
            else:
                flat_texts = [_flat_text(t) for t in chunk_texts]
                inputs = processor(text=[flat_texts], images=pil, return_tensors="pt")

            inputs = {k: v.to(dev) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)

            if use_gdino:
                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    box_threshold=self.score_thresh,
                    text_threshold=self.score_thresh,
                    target_sizes=[pil.size[::-1]],
                )[0]
            else:
                target_sizes = torch.tensor([[h, w]], device=dev)
                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    box_threshold=self.score_thresh,
                    text_threshold=self.score_thresh,
                    target_sizes=target_sizes,
                )[0]

            boxes, scores, labels = _result_arrays(results)
            for box, score, label_idx in zip(boxes, scores, labels):
                li = _label_to_index(label_idx, chunk_texts, chunk_names)
                if li is None:
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
