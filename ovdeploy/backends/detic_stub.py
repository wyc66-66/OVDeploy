"""Detic backend stub: fail-closed until Detectron2 + weights are wired."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ovdeploy.paths_util import load_paths


class DeticBackend:
    name = "detic"

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device
        self.cfg = load_paths()
        root = Path(self.cfg.get("_root", Path(__file__).resolve().parents[2]))
        weight_dir = root / "weights" / "detic"
        # Timed integration: require non-empty weights + detectron2
        try:
            import detectron2  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "Detic blocked: detectron2 not installed. "
                "Seat: timed attempt → checkpoint_blocked."
            ) from e
        if not weight_dir.is_dir() or not any(weight_dir.iterdir()):
            raise RuntimeError(
                f"Detic blocked: empty weights dir {weight_dir}. "
                "Seat: timed attempt → checkpoint_blocked."
            )
        raise RuntimeError(
            "Detic blocked: Detectron2 present but OVDeploy Detic predict() "
            "not fully wired (timed fail-fast). See OVD_FAMILY_SEAT_TABLE_zh.md"
        )

    def predict(
        self,
        image_rgb: np.ndarray,
        texts: list[str],
        vocab_ids: list[int],
        image_id: int,
        **kwargs,
    ) -> list[dict]:
        raise RuntimeError("DeticBackend not runnable")
