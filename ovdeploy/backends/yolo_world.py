"""YOLO-World v2-S / v2-M backend."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ovdeploy.paths_util import load_paths
from ovdeploy.vocab import subset_class_texts

_MODEL_CACHE: dict[tuple[str, str, str], tuple[Any, Any]] = {}


class YoloWorldBackend:
    name = "yolo"

    def __init__(self, device: str = "cuda:0", variant: str = "s") -> None:
        self.device = device
        self.variant = variant.lower()
        self.cfg = load_paths()
        self.yolo: Path = self.cfg["_yolo"]

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

        if class_texts_raw is not None and cid2idx is not None:
            texts_sub = subset_class_texts(class_texts_raw, vocab_ids, cid2idx)
            sub_names = [
                class_names[cid2idx[c]] if class_names and c in cid2idx else str(c)
                for c in vocab_ids
            ]
        else:
            texts_sub = texts
            sub_names = texts

        model, test_pipeline = self._init_model()
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
