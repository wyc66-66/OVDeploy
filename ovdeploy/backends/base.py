"""Detector backend registry."""
from __future__ import annotations

from typing import Protocol

import numpy as np


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
    if key in ("glip", "glip_t"):
        from ovdeploy.backends.glip import GlipBackend

        return GlipBackend(device=device)
    if key in ("glip_native", "native_glip", "microsoft_glip", "glip_ms"):
        from ovdeploy.backends.glip_native import NativeGlipBackend

        return NativeGlipBackend(device=device)
    raise ValueError(f"Unknown backbone: {name}")
