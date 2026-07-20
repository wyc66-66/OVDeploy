"""OpenSeeD backend stub: fail-closed until official inference is wired."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ovdeploy.paths_util import load_paths


class OpenSeedBackend:
    name = "openseed"

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device
        self.cfg = load_paths()
        root = Path(self.cfg.get("_root", Path(__file__).resolve().parents[2]))
        weight_dir = root / "weights" / "openseed"
        if not weight_dir.is_dir() or not any(weight_dir.iterdir()):
            raise RuntimeError(
                f"OpenSeeD blocked: empty weights dir {weight_dir}. "
                "Seat: timed attempt → checkpoint_blocked."
            )
        raise RuntimeError(
            "OpenSeeD blocked: weights folder exists but OVDeploy predict() "
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
        raise RuntimeError("OpenSeedBackend not runnable")
