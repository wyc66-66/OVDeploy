"""VocabGuard: deployment-constrained OVD inference (Paper 4)."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_CFG = _ROOT / "config" / "paths.yaml"


def _bootstrap_ovdeploy() -> None:
    import yaml

    if not _CFG.is_file():
        return
    with open(_CFG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("ovdeploy_root", "ovdeploy_root_wsl"):
        p = Path(cfg[key])
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
            return


_bootstrap_ovdeploy()

from vocabguard.calib_head import CalibHead
from vocabguard.oov_guard import OOVGuard
from vocabguard.router import VocabRouter

__all__ = ["VocabRouter", "OOVGuard", "CalibHead"]
