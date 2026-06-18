"""Load Paper 4 config and bridge to OVDeploy (Paper 2) read-only assets."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
OVROOT = Path(__file__).resolve().parents[1]  # placeholder, resolved in load_config


def _bootstrap_ovdeploy(cfg: dict[str, Any]) -> None:
    ov_root = _resolve(cfg["ovdeploy_root"])
    if not ov_root.exists():
        ov_root = _resolve(cfg["ovdeploy_root_wsl"])
    if ov_root.is_dir() and str(ov_root) not in sys.path:
        sys.path.insert(0, str(ov_root))


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
    cfg["_root"] = ROOT

    yolo = _resolve(cfg["yolo_root"])
    if not yolo.exists():
        yolo = _resolve(cfg["yolo_root_wsl"])
    cfg["_yolo"] = yolo

    _bootstrap_ovdeploy(cfg)
    return cfg


def ovdeploy_paths() -> dict[str, Any]:
    """OVDeploy paths.yaml via Paper 2 root."""
    from ovdeploy.paths_util import load_paths

    return load_paths()


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
