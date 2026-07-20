"""Florence-2 open-vocabulary detection backend (phrase grounding)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ovdeploy.paths_util import load_paths

_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any, str]] = {}


class Florence2Backend:
    name = "florence"

    def __init__(
        self,
        device: str = "cuda:0",
        model_id: str = "microsoft/Florence-2-base",
        variant: str = "base",
    ) -> None:
        self.device = device
        self.variant = variant
        self.cfg = load_paths()
        fl_cfg = self.cfg.get("florence2", {})
        if variant in ("large", "l", "florence_l"):
            self.model_id = fl_cfg.get("model_id_large", "microsoft/Florence-2-large")
            local_key = "local_dir_large"
        else:
            self.model_id = model_id or fl_cfg.get("model_id_base", "microsoft/Florence-2-base")
            local_key = "local_dir_base"
        self.local_dir = None
        local = fl_cfg.get(local_key)
        if local:
            p = Path(local)
            if not p.is_absolute():
                p = Path(self.cfg.get("_root", Path(__file__).resolve().parents[2])) / local
            if p.is_dir():
                self.local_dir = str(p)
        self.score_thresh = float(fl_cfg.get("score_thresh", 0.05))

    def _model_source(self) -> str:
        return self.local_dir or self.model_id

    def _init_model(self):
        src = self._model_source()
        key = (src, self.device)
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(src, trust_remote_code=True, local_files_only=bool(self.local_dir))
        model = AutoModelForCausalLM.from_pretrained(src, trust_remote_code=True, local_files_only=bool(self.local_dir))
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

        model, processor, dev = self._init_model()
        pil = Image.fromarray(image_rgb)
        raw = texts if texts else list(class_names or vocab_ids)
        labels: list[str] = []
        for t in raw:
            if isinstance(t, list):
                labels.append(str(t[0]) if t else "")
            else:
                labels.append(str(t))
        labels = [x for x in labels if x]
        if not labels or not vocab_ids:
            return []
        # Keep prompts and category ids aligned (zip by shortest length).
        n = min(len(labels), len(vocab_ids))
        labels = labels[:n]
        vocab_ids = list(vocab_ids[:n])
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        phrase = ". ".join(labels[: min(50, len(labels))])
        try:
            inputs = processor(
                text=task_prompt + phrase,
                images=pil,
                return_tensors="pt",
            ).to(dev)
            generated = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=1,
                early_stopping=False,
            )
            generated_text = processor.batch_decode(generated, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(
                generated_text, task=task_prompt, image_size=pil.size
            )
        except Exception:
            return []
        preds: list[dict] = []
        det = parsed.get(task_prompt, {}) if isinstance(parsed, dict) else {}
        bboxes = det.get("bboxes", []) if isinstance(det, dict) else []
        labels_out = det.get("labels", []) if isinstance(det, dict) else []
        for i, box in enumerate(bboxes):
            if len(box) < 4:
                continue
            lbl = labels_out[i] if i < len(labels_out) else (labels[0] if labels else "")
            cid = int(vocab_ids[0])
            for t, vid in zip(labels, vocab_ids):
                if t.lower() in str(lbl).lower():
                    cid = int(vid)
                    break
            x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            # Match YOLO/OWL: xywh + category_id (metrics.py).
            preds.append({
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": 0.5,
                "category_id": cid,
                "cat_id": cid,  # compat
            })
        return [p for p in preds if p.get("score", 0) >= self.score_thresh]
