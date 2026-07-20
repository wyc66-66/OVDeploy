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
    if key in ("yolo_l", "yoloworld_l", "yolov2l"):
        from ovdeploy.backends.yolo_world import YoloWorldBackend

        return YoloWorldBackend(device=device, variant="l")
    if key in ("yolo_x", "yoloworld_x", "yolov2x"):
        from ovdeploy.backends.yolo_world import YoloWorldBackend

        return YoloWorldBackend(device=device, variant="x")
    if key in ("uyolo_s", "ultralytics_s", "yolov8s_worldv2"):
        from ovdeploy.backends.ultralytics_world import UltralyticsWorldBackend

        return UltralyticsWorldBackend(device=device, variant="s")
    if key in ("uyolo_m", "ultralytics_m", "yolov8m_worldv2"):
        from ovdeploy.backends.ultralytics_world import UltralyticsWorldBackend

        return UltralyticsWorldBackend(device=device, variant="m")
    if key in ("uyolo_l", "ultralytics_l", "yolov8l_worldv2"):
        from ovdeploy.backends.ultralytics_world import UltralyticsWorldBackend

        return UltralyticsWorldBackend(device=device, variant="l")
    if key in ("uyolo_x", "ultralytics_x", "yolov8x_worldv2"):
        from ovdeploy.backends.ultralytics_world import UltralyticsWorldBackend

        return UltralyticsWorldBackend(device=device, variant="x")
    if key in ("owlvit", "owl_vit", "owl-vit"):
        from ovdeploy.backends.owlvit import OwlvitBackend

        return OwlvitBackend(device=device)
    if key in ("owlvit_b16", "owlvit_base_patch16", "owl-vit-b16"):
        from ovdeploy.backends.owlvit import OwlvitBackend

        cfg = load_paths()
        oc = cfg.get("owlvit_b16") or {}
        return OwlvitBackend(
            device=device,
            model_id=oc.get("model_id", "google/owlvit-base-patch16"),
            local_dir=oc.get("local_dir", "weights/owlvit-base-patch16"),
            name="owlvit_b16",
        )
    if key in ("owlvit_l", "owlvit_large", "owl-vit-large"):
        from ovdeploy.backends.owlvit import OwlvitBackend

        cfg = load_paths()
        oc = cfg.get("owlvit_l") or {}
        return OwlvitBackend(
            device=device,
            model_id=oc.get("model_id", "google/owlvit-large-patch14"),
            local_dir=oc.get("local_dir", "weights/owlvit-large-patch14"),
            name="owlvit_l",
        )
    if key in ("owlv2", "owl_v2", "owl-v2", "owlv2_ensemble"):
        from ovdeploy.backends.owlv2 import Owlv2Backend

        return Owlv2Backend(device=device)
    if key in ("owlv2_base", "owlv2_base_ne", "owlv2-base-patch16"):
        from ovdeploy.backends.owlv2 import Owlv2Backend

        cfg = load_paths()
        oc = cfg.get("owlv2") or {}
        return Owlv2Backend(
            device=device,
            model_id=oc.get("model_id_base_ne", "google/owlv2-base-patch16"),
            local_dir=oc.get("local_dir_base_ne", "weights/owlv2-base-patch16"),
            name="owlv2_base",
        )
    if key in ("owlv2_large", "owl_v2_large"):
        from ovdeploy.backends.owlv2 import Owlv2Backend

        cfg = load_paths()
        mid = (cfg.get("owlv2") or {}).get("model_id_large", "google/owlv2-large-patch14-ensemble")
        return Owlv2Backend(device=device, model_id=mid)
    if key in ("florence_b", "florence_base", "florence-2-base"):
        from ovdeploy.backends.florence import Florence2Backend

        return Florence2Backend(device=device, variant="base")
    if key in ("florence_l", "florence_large", "florence-2-large"):
        from ovdeploy.backends.florence import Florence2Backend

        return Florence2Backend(device=device, variant="large")
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
    if key in ("glip_l", "glip_large", "native_glip_l"):
        from ovdeploy.backends.glip_native import NativeGlipBackend

        return NativeGlipBackend(device=device, variant="large")
    if key in ("detclip_v2", "detclipv2", "detclip", "detclip_v2_t"):
        from ovdeploy.backends.detclip import DetclipV2Backend

        return DetclipV2Backend(device=device)
    if key in ("omdet_turbo", "omdet", "omdet-turbo", "omdetturbo"):
        from ovdeploy.backends.omdet_turbo import OmDetTurboBackend

        return OmDetTurboBackend(device=device)
    if key in ("detic",):
        from ovdeploy.backends.detic_stub import DeticBackend

        return DeticBackend(device=device)
    if key in ("openseed", "open_seed"):
        from ovdeploy.backends.openseed_stub import OpenSeedBackend

        return OpenSeedBackend(device=device)
    raise ValueError(f"Unknown backbone: {name}")
