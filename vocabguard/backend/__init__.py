"""Backend registry for VocabGuard."""
from vocabguard.backend.yolo_ext import YoloWorldExtended

__all__ = ["YoloWorldExtended"]


def get_extended_backend(name: str = "yolo", device: str = "cuda:0"):
    key = (name or "yolo").lower()
    if key in ("yolo", "yoloworld", "yolo_s", "yolo_ext"):
        return YoloWorldExtended(device=device, variant="s")
    if key in ("yolo_m",):
        return YoloWorldExtended(device=device, variant="m")
    if key in ("owlvit", "owl"):
        from ovdeploy.backends.base import get_backend

        return get_backend("owlvit", device=device)
    raise ValueError(f"Unknown backbone: {name}")
