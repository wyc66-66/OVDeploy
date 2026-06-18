"""Load RobustVocab config; bridge to OVDeploy read-only assets."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _resolve(p: str | Path) -> Path:
    path = Path(p)
    if path.exists():
        return path
    alt = Path(str(p).replace("d:/ccfa", "/mnt/d/ccfa").replace("\\", "/"))
    if alt.exists():
        return alt
    return path


def load_config() -> dict[str, Any]:
    with open(ROOT / "config" / "paths.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ov_root = _resolve(cfg["ovdeploy_root"])
    if not ov_root.exists():
        ov_root = _resolve(cfg["ovdeploy_root_wsl"])
    cfg["_ovdeploy"] = ov_root

    yolo = _resolve(cfg["yolo_root"])
    if not yolo.exists():
        yolo = _resolve(cfg["yolo_root_wsl"])
    cfg["_yolo"] = yolo

    cfg["_root"] = ROOT
    cfg["_vocabguard"] = ROOT

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if ov_root.is_dir() and str(ov_root) not in sys.path:
        sys.path.insert(0, str(ov_root))

    return cfg


def episodes_dir(split: str = "dev") -> Path:
    cfg = load_config()
    return cfg["_ovdeploy"] / "data" / "episodes" / split


def reports_dir() -> Path:
    cfg = load_config()
    d = ROOT / cfg["outputs"]["reports"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoints_dir() -> Path:
    cfg = load_config()
    d = ROOT / cfg["outputs"]["checkpoints"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def cooccur_cache_path() -> Path:
    cfg = load_config()
    return ROOT / cfg["data"]["cooccur_cache"]
