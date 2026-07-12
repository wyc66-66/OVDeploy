"""DetCLIPv2-T backend (MMDet ATSS + text encoder; requires third_party config + checkpoint)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ovdeploy.paths_util import load_paths
from ovdeploy.vocab import subset_class_texts

_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


class DetclipCheckpointError(FileNotFoundError):
    """Raised when DetCLIPv2 weights or config are missing."""


def _flat_text(t: Any) -> str:
    if isinstance(t, list) and t:
        return str(t[0])
    return str(t)


def _resolve_paths(cfg: dict) -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[2]
    dc = cfg.get("detclip_v2", {})
    ckpt_rel = dc.get("checkpoint", "weights/detclipv2_swin_t/detclipv2_swin_t.pth")
    cfg_rel = dc.get("config", "third_party/DetCLIPv2/configs/detclipv2_swin_t_lvis.py")
    env_ckpt = os.environ.get("DETCLIP_V2_CHECKPOINT", "").strip()
    env_cfg = os.environ.get("DETCLIP_V2_CONFIG", "").strip()
    ckpt = Path(env_ckpt) if env_ckpt else root / ckpt_rel
    mcfg = Path(env_cfg) if env_cfg else root / cfg_rel
    third = root / "third_party" / "DetCLIPv2"
    return ckpt, mcfg, third


def checkpoint_ready(cfg: dict | None = None) -> tuple[bool, str]:
    cfg = cfg or load_paths()
    ckpt, mcfg, _ = _resolve_paths(cfg)
    if not mcfg.is_file():
        return False, f"Missing config: {mcfg}"
    if not ckpt.is_file():
        return False, f"Missing checkpoint: {ckpt}"
    return True, str(ckpt)


class DetclipV2Backend:
    name = "detclip_v2"

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device
        self.cfg = load_paths()
        self.dc = self.cfg.get("detclip_v2", {})
        self.chunk_size = int(self.dc.get("chunk_size", 120))
        self.score_thresh = float(self.dc.get("score_thresh", 0.05))
        self._text_chunks: list[tuple[list[str], list[int]]] | None = None
        self._warm_vocab_key: tuple[int, ...] | None = None

    def warm_full_vocab(
        self,
        class_names: list[str],
        class_texts_raw: list,
        all_cat_ids: list[int],
        cid2idx: dict[int, int],
    ) -> None:
        """Preload model and chunk plan for B0 full-vocab cache loop."""
        texts_sub, _ = subset_class_texts(class_names, class_texts_raw, all_cat_ids, cid2idx)
        key = tuple(all_cat_ids)
        if self._warm_vocab_key == key:
            return
        self._init_model()
        self._text_chunks = self._chunk_vocab(
            texts_sub, all_cat_ids, class_texts_raw, cid2idx
        )
        self._warm_vocab_key = key
        print(
            f"DetCLIPv2 B0 warm: {len(self._text_chunks)} chunks "
            f"(chunk_size={self.chunk_size}, n_vocab={len(all_cat_ids)})",
            flush=True,
        )

    def _init_model(self):
        ok, msg = checkpoint_ready(self.cfg)
        if not ok:
            raise DetclipCheckpointError(
                f"{msg}. See docs/DETCLIP_V2_CHECKPOINT_HUNT.md"
            )
        ckpt, mcfg, third = _resolve_paths(self.cfg)
        key = (str(mcfg), str(ckpt), self.device)
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]

        yolo_root = self.cfg.get("_yolo")
        if yolo_root and yolo_root.is_dir():
            os.chdir(yolo_root)
            if str(yolo_root) not in sys.path:
                sys.path.insert(0, str(yolo_root))
        if third.is_dir() and str(third) not in sys.path:
            sys.path.insert(0, str(third))

        from mmengine.config import Config
        from mmengine.dataset import Compose
        from mmdet.apis import init_detector
        from mmdet.utils import get_test_pipeline_cfg

        model_cfg = Config.fromfile(str(mcfg))
        model_cfg.load_from = str(ckpt)
        model = init_detector(model_cfg, checkpoint=str(ckpt), device=self.device)
        test_pipeline_cfg = get_test_pipeline_cfg(cfg=model_cfg)
        for step in test_pipeline_cfg:
            if step.get("type") == "mmdet.LoadImageFromFile":
                step["type"] = "mmdet.LoadImageFromNDArray"
        test_pipeline = Compose(test_pipeline_cfg)
        _MODEL_CACHE[key] = (model, test_pipeline)
        return model, test_pipeline

    def image_path(self, file_name: str) -> Path:
        from ovdeploy.paths_util import resolve_val2017_image

        return resolve_val2017_image(self.cfg, file_name)

    def _chunk_vocab(
        self,
        texts: list[str],
        vocab_ids: list[int],
        class_texts_raw: list | None,
        cid2idx: dict[int, int] | None,
    ) -> list[tuple[list[str], list[int]]]:
        if class_texts_raw is not None and cid2idx is not None:
            flat = [_flat_text(class_texts_raw[cid2idx[c]]) for c in vocab_ids]
            names = flat
        else:
            flat = [_flat_text(t) for t in texts]
            names = flat
        chunks: list[tuple[list[str], list[int]]] = []
        for i in range(0, len(flat), self.chunk_size):
            chunk_texts = flat[i : i + self.chunk_size]
            chunk_ids = vocab_ids[i : i + self.chunk_size]
            chunks.append((chunk_texts, chunk_ids))
        return chunks

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

        model, test_pipeline = self._init_model()
        warm_key = tuple(vocab_ids)
        if (
            self._text_chunks is not None
            and self._warm_vocab_key == warm_key
            and len(vocab_ids) == len(self._warm_vocab_key or ())
        ):
            chunks = self._text_chunks
        else:
            chunks = self._chunk_vocab(texts, vocab_ids, class_texts_raw, cid2idx)
        all_preds: list[dict] = []

        for chunk_texts, chunk_ids in chunks:
            if class_texts_raw is not None and cid2idx is not None:
                sub_names = [_flat_text(class_texts_raw[cid2idx[c]]) for c in chunk_ids]
            else:
                sub_names = chunk_texts

            data_info = dict(img=image_rgb, img_id=image_id, texts=chunk_texts)
            data_info = test_pipeline(data_info)
            data_batch = dict(
                inputs=data_info["inputs"].unsqueeze(0),
                data_samples=[data_info["data_samples"]],
            )
            with torch.no_grad():
                out = model.test_step(data_batch)[0]
            pred = out.pred_instances

            for b, s, lb in zip(
                pred.bboxes.cpu().numpy(),
                pred.scores.cpu().numpy(),
                pred.labels.cpu().numpy(),
            ):
                score = float(s)
                if score < self.score_thresh:
                    continue
                li = int(lb)
                cid = chunk_ids[li] if li < len(chunk_ids) else -1
                x1, y1, x2, y2 = [float(v) for v in b]
                bbox_xywh = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
                all_preds.append(
                    {
                        "bbox": bbox_xywh,
                        "score": score,
                        "label_idx": li,
                        "category_id": cid,
                        "category_name": sub_names[li] if li < len(sub_names) else "",
                    }
                )

        return self._nms_preds(all_preds)

    @staticmethod
    def _nms_preds(preds: list[dict], iou_thresh: float = 0.5) -> list[dict]:
        if not preds:
            return preds
        boxes = []
        for p in preds:
            x, y, w, h = p["bbox"]
            boxes.append([x, y, x + w, y + h, p["score"], p["category_id"], p])
        boxes.sort(key=lambda x: -x[4])
        kept: list[dict] = []
        while boxes:
            best = boxes.pop(0)
            kept.append(best[6])
            boxes = [
                b
                for b in boxes
                if b[5] != best[5]
                or _box_iou(best[:4], b[:4]) < iou_thresh
            ]
        return kept


def _box_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter + 1e-6
    return inter / union
