"""Native Microsoft GLIP-T (.pth) via maskrcnn_benchmark GLIPDemo."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from ovdeploy.paths_util import load_paths
from ovdeploy.vocab import subset_class_texts

_DEMO_CACHE: dict[tuple[str, str], Any] = {}


def _flat_text(t: Any) -> str:
    if isinstance(t, list) and t:
        return str(t[0])
    return str(t)


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


def _glip_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    glip = root / "third_party" / "GLIP"
    if glip.is_dir() and (glip / "setup.py").is_file():
        if str(glip) not in sys.path:
            sys.path.insert(0, str(glip))
    return glip


def _init_demo(device: str, weight: str, config_file: str):
    key = (device, weight)
    if key in _DEMO_CACHE:
        return _DEMO_CACHE[key]

    _glip_root()
    import torch
    from maskrcnn_benchmark.config import cfg as glip_defaults
    from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
    from maskrcnn_benchmark.utils.env import setup_environment  # noqa: F401

    cfg = glip_defaults.clone()
    cfg.merge_from_file(config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHT = weight
    dev = device if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = dev
    cfg.freeze()

    demo = GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.25,
        show_mask_heatmaps=False,
    )
    _DEMO_CACHE[key] = demo
    return demo


class NativeGlipBackend:
    name = "glip_native"

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device
        self.cfg = load_paths()
        glip_cfg = self.cfg.get("glip", {})
        root = Path(self.cfg.get("_root", Path(__file__).resolve().parents[2]))
        native_w = glip_cfg.get("native_weight", "weights/glip-native/glip_tiny_model_o365_goldg.pth")
        wp = Path(native_w)
        if not wp.is_absolute():
            wp = root / wp
        if not wp.is_file():
            raise FileNotFoundError(f"Native GLIP weight missing: {wp}")
        self.native_weight = str(wp)

        native_cfg = glip_cfg.get("native_config")
        cp = root / "third_party/GLIP/configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
        if native_cfg:
            alt = Path(native_cfg)
            if not alt.is_absolute():
                alt = root / alt
            if alt.is_file() and "third_party" in str(alt):
                cp = alt
        if not cp.is_file():
            raise FileNotFoundError(f"Native GLIP config missing: {cp}")
        self.config_file = str(cp)

        self.chunk_size = int(glip_cfg.get("chunk_size", 40))
        self.score_thresh = float(glip_cfg.get("score_thresh", 0.05))
        self.yolo: Path = self.cfg["_yolo"]
        self._demo = None

    def _demo_instance(self):
        if self._demo is None:
            self._demo = _init_demo(self.device, self.native_weight, self.config_file)
        return self._demo

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
        if class_texts_raw is not None and cid2idx is not None:
            texts_sub = subset_class_texts(class_texts_raw, vocab_ids, cid2idx)
            sub_names = [
                class_names[cid2idx[c]] if class_names and c in cid2idx else str(c)
                for c in vocab_ids
            ]
        else:
            texts_sub = texts
            sub_names = [str(t) for t in texts]

        bgr = image_rgb[:, :, ::-1].copy()
        demo = self._demo_instance()
        all_preds: list[dict] = []

        for start in range(0, len(texts_sub), self.chunk_size):
            chunk_texts = texts_sub[start : start + self.chunk_size]
            chunk_ids = vocab_ids[start : start + self.chunk_size]
            chunk_names = sub_names[start : start + self.chunk_size]
            if not chunk_texts:
                continue

            phrases = [_flat_text(t).strip() for t in chunk_texts]
            phrases = [p if p else "object" for p in phrases]
            top = demo.compute_prediction(bgr, phrases)
            top = demo._post_process(top, self.score_thresh)
            demo.entities = phrases
            if top is None or len(top.bbox) == 0:
                continue

            boxes = top.bbox.cpu().numpy() if hasattr(top.bbox, "cpu") else np.asarray(top.bbox)
            scores = top.get_field("scores")
            scores = scores.cpu().numpy() if hasattr(scores, "cpu") else np.asarray(scores)
            labels = top.get_field("labels").tolist()

            plus = 1 if demo.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD" else 0
            entities = getattr(demo, "entities", None) or phrases

            for box, score, lb in zip(boxes, scores, labels):
                li = int(lb) - plus
                if 0 <= li < len(chunk_ids):
                    idx = li
                else:
                    ent = entities[li] if 0 <= li < len(entities) else ""
                    idx = next(
                        (
                            i
                            for i, p in enumerate(phrases)
                            if p.lower() == str(ent).lower()
                            or p.lower() in str(ent).lower()
                            or str(ent).lower() in p.lower()
                        ),
                        None,
                    )
                    if idx is None:
                        continue
                x1, y1, x2, y2 = [float(v) for v in box]
                bbox_xywh = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
                all_preds.append(
                    {
                        "bbox": bbox_xywh,
                        "score": float(score),
                        "label_idx": start + idx,
                        "category_id": chunk_ids[idx],
                        "category_name": chunk_names[idx],
                    }
                )

        all_preds.sort(key=lambda x: x["score"], reverse=True)
        kept: list[dict] = []
        for p in all_preds:
            dup = False
            bx = np.array([p["bbox"][0], p["bbox"][1], p["bbox"][0] + p["bbox"][2], p["bbox"][1] + p["bbox"][3]])
            for k in kept:
                kb = k["bbox"]
                kxy = np.array([kb[0], kb[1], kb[0] + kb[2], kb[1] + kb[3]])
                if k["category_id"] == p["category_id"] and _box_iou_xyxy(kxy, bx) > 0.5:
                    dup = True
                    break
            if not dup:
                kept.append(p)
        return kept
