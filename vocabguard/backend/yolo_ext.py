"""Extended YOLO-World backend: detector-native class scores + neck features."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ovdeploy.paths_util import cat_id_to_index, load_class_texts, load_paths
from ovdeploy.vocab import subset_class_texts

_MODEL_CACHE: dict[tuple[str, str, str], tuple[Any, Any]] = {}
_FEAT_HOOK: dict[str, list] = {"feat": []}


class YoloWorldExtended:
    """Wraps frozen YOLO-World with class scoring and image embedding hooks."""

    name = "yolo_ext"

    def __init__(self, device: str = "cuda:0", variant: str = "s") -> None:
        self.device = device
        self.variant = variant.lower()
        self.cfg = load_paths()
        self.yolo: Path = self.cfg["_yolo"]
        self._hook_handle = None

    def _variant_cfg(self) -> tuple[str, str]:
        if self.variant in ("m", "yolo_m", "v2_m"):
            return (
                self.cfg["config"].get(
                    "eval_8gb_m", "configs/pretrain/yolo_world_v2_m_lvis_minival_8gb.py"
                ),
                self.cfg["weights"]["yolo_world_v2_m"],
            )
        return (
            self.cfg["config"]["eval_8gb"],
            self.cfg["weights"]["yolo_world_v2_s"],
        )

    def _init_model(self):
        cfg_rel, ckpt_rel = self._variant_cfg()
        key = (str(self.yolo), self.device, self.variant)
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]

        os.chdir(self.yolo)
        sys.path.insert(0, str(self.yolo))
        from mmengine.config import Config
        from mmengine.dataset import Compose
        from mmdet.apis import init_detector
        from mmdet.utils import get_test_pipeline_cfg

        ckpt = self.yolo / ckpt_rel
        model_cfg = Config.fromfile(str(self.yolo / cfg_rel))
        model_cfg.load_from = str(ckpt)
        model = init_detector(model_cfg, checkpoint=str(ckpt), device=self.device)
        test_pipeline_cfg = get_test_pipeline_cfg(cfg=model_cfg)
        test_pipeline_cfg[0].type = "mmdet.LoadImageFromNDArray"
        test_pipeline = Compose(test_pipeline_cfg)
        _MODEL_CACHE[key] = (model, test_pipeline)
        return model, test_pipeline

    def _register_feat_hook(self, model) -> None:
        if self._hook_handle is not None:
            return
        target = None
        if hasattr(model, "neck"):
            target = model.neck
        elif hasattr(model, "backbone"):
            target = model.backbone

        def _hook(_module, _inp, out):
            t = out[-1] if isinstance(out, (list, tuple)) else out
            if hasattr(t, "detach"):
                pooled = t.mean(dim=(2, 3)) if t.dim() == 4 else t
                _FEAT_HOOK["feat"].append(pooled.detach().cpu())

        if target is not None:
            self._hook_handle = target.register_forward_hook(_hook)

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
        vocab_ids: list[int],
        image_id: int,
        lvis: dict,
        class_names: list[str] | None = None,
        class_texts_raw: list | None = None,
        cid2idx: dict[int, int] | None = None,
    ) -> list[dict]:
        import torch

        if class_names is None or class_texts_raw is None or cid2idx is None:
            class_names, class_texts_raw = load_class_texts(self.cfg)
            cid2idx = cat_id_to_index(lvis)

        texts_sub = subset_class_texts(class_texts_raw, vocab_ids, cid2idx)
        sub_names = [
            class_names[cid2idx[c]] if c in cid2idx else str(c) for c in vocab_ids
        ]

        model, test_pipeline = self._init_model()
        _FEAT_HOOK["feat"].clear()
        self._register_feat_hook(model)

        data_info = dict(img=image_rgb, img_id=image_id, texts=texts_sub)
        data_info = test_pipeline(data_info)
        data_batch = dict(
            inputs=data_info["inputs"].unsqueeze(0),
            data_samples=[data_info["data_samples"]],
        )
        with torch.no_grad():
            out = model.test_step(data_batch)[0]
        pred = out.pred_instances

        preds = []
        for b, s, lb in zip(
            pred.bboxes.cpu().numpy(), pred.scores.cpu().numpy(), pred.labels.cpu().numpy()
        ):
            li = int(lb)
            cid = vocab_ids[li] if li < len(vocab_ids) else -1
            x1, y1, x2, y2 = [float(v) for v in b]
            bbox_xywh = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
            preds.append(
                {
                    "bbox": bbox_xywh,
                    "score": float(s),
                    "label_idx": li,
                    "category_id": cid,
                    "category_name": sub_names[li] if li < len(sub_names) else "",
                }
            )
        return preds

    def score_classes(
        self,
        image_rgb: np.ndarray,
        cat_ids: list[int],
        image_id: int,
        lvis: dict,
    ) -> dict[int, float]:
        """Detector-native alignment: max detection score per class in probe vocab."""
        if not cat_ids:
            return {}
        preds = self.predict(image_rgb, cat_ids, image_id, lvis)
        scores: dict[int, float] = {c: 0.0 for c in cat_ids}
        for p in preds:
            cid = p.get("category_id", -1)
            if cid in scores:
                scores[cid] = max(scores[cid], float(p["score"]))
        return scores

    def encode_image(self, image_rgb: np.ndarray, image_id: int, lvis: dict) -> np.ndarray:
        """Global image embedding from neck GAP (256-d fallback if hook fails)."""
        import torch

        _FEAT_HOOK["feat"].clear()
        # Minimal vocab forward to trigger neck hook
        dummy_vocab = [lvis["categories"][0]["id"]]
        self.predict(image_rgb, dummy_vocab, image_id, lvis)
        if _FEAT_HOOK["feat"]:
            t = _FEAT_HOOK["feat"][-1]
            if t.dim() > 1:
                feat = t.mean(dim=0).numpy().astype(np.float32)
            else:
                feat = t.numpy().astype(np.float32)
            if feat.size >= 256:
                return feat[:256]
            out = np.zeros(256, dtype=np.float32)
            out[: feat.size] = feat
            return out
        rng = np.random.default_rng(image_id)
        return rng.standard_normal(256).astype(np.float32)
