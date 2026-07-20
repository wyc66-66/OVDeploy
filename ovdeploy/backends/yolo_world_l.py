"""YOLO-World v2-L backend (scale ablation)."""
from __future__ import annotations

from ovdeploy.backends.yolo_world import YoloWorldBackend


class YoloWorldLBackend(YoloWorldBackend):
    name = "yolo_l"

    def __init__(self, device: str = "cuda:0") -> None:
        super().__init__(device=device, variant="l")
