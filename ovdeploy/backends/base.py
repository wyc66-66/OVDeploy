"""Detector backend registry."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from ovdeploy.paths_util import load_paths


class DetectorBackend(Protocol):
    name: str

    def predict(
        self,
        image_rgb: np.ndarray,
        texts: list[str],
        vocab_ids: list[int],
        image_id: int,
    ) -> list[dict]:
        ...


def get_backend(name: str = "yolo", device: str = "cuda:0") -> DetectorBackend:
    key = (name or "yolo").lower()
    if key in ("yolo", "yoloworld", "yolo_world", "yolo_s", "yolov2s"):
        from ovdeploy.backends.yolo_world import YoloWorldBackend

        return YoloWorldBackend(device=device, variant="s")
    if key in ("yolo_m", "yoloworld_m", "yolov2m"):
        from ovdeploy.backends.yolo_world import YoloWorldBackend

        return YoloWorldBackend(device=device, variant="m")
    if key in ("owlvit", "owl_vit", "owl-vit"):
        from ovdeploy.backends.owlvit import OwlvitBackend

        return OwlvitBackend(device=device)
    if key in ("glip", "glip_t", "gdino_tiny", "gdino-tiny"):
        from ovdeploy.backends.glip import GlipBackend

        return GlipBackend(device=device)
    if key in ("gdino_base", "grounding_dino_base", "gdino-base", "glip_base"):
        from ovdeploy.backends.glip import GlipBackend

        root = Path(__file__).resolve().parents[2]
        local = root / "weights" / "grounding-dino-base"
        cfg = load_paths()
        gdino_cfg = cfg.get("gdino_base", {})
        chunk_size = gdino_cfg.get("chunk_size")
        score_thresh = gdino_cfg.get("score_thresh")
        max_text_tokens = gdino_cfg.get("max_text_tokens")
        b0_short = gdino_cfg.get("b0_short_captions")
        b0_edge = gdino_cfg.get("b0_image_short_edge")
        return GlipBackend(
            device=device,
            model_id="IDEA-Research/grounding-dino-base",
            local_dir=str(local) if local.is_dir() else None,
            chunk_size=int(chunk_size) if chunk_size is not None else None,
            score_thresh=float(score_thresh) if score_thresh is not None else None,
            max_text_tokens=int(max_text_tokens) if max_text_tokens is not None else None,
            b0_short_captions=bool(b0_short) if b0_short is not None else None,
            b0_image_short_edge=int(b0_edge) if b0_edge is not None else None,
        )
    if key in ("glip_native", "native_glip", "microsoft_glip", "glip_ms"):
        from ovdeploy.backends.glip_native import NativeGlipBackend

        return NativeGlipBackend(device=device)
    raise ValueError(f"Unknown backbone: {name}")
