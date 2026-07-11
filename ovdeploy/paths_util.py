"""Load project + YOLO-World paths."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_paths() -> dict[str, Any]:
    with open(ROOT / "config" / "paths.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    import sys

    if sys.platform != "win32" and cfg.get("yolo_root_wsl"):
        yolo = Path(cfg["yolo_root_wsl"])
    else:
        yolo = Path(cfg["yolo_root"])
        if not yolo.is_dir() and cfg.get("yolo_root_wsl"):
            yolo = Path(cfg["yolo_root_wsl"])
    cfg["_yolo"] = yolo
    cfg["_root"] = ROOT
    return cfg


def load_episodes_cfg() -> dict[str, Any]:
    with open(ROOT / "config" / "episodes.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def yolo_path(cfg: dict, key: str) -> Path:
    return cfg["_yolo"] / cfg["data"][key]


def resolve_val2017_image(cfg: dict, file_name: str) -> Path:
    """Resolve COCO val2017 path; prefer WSL native cache when present."""
    fn = file_name.replace("\\", "/")
    import sys

    cache_key = "val2017_cache_wsl"
    if sys.platform != "win32" and cfg.get(cache_key):
        cache_dir = Path(cfg[cache_key]).expanduser()
        cached = cache_dir / Path(fn).name
        if cached.is_file():
            return cached

    img_dir = cfg["_yolo"] / cfg["data"]["val2017_dir"]
    coco_root = cfg["_yolo"] / "data/coco"
    for candidate in (coco_root / fn, img_dir / Path(fn).name, img_dir / fn):
        if candidate.is_file():
            return candidate
    return img_dir / fn


def report_path(cfg: dict, key: str) -> Path:
    return ROOT / cfg["reports"][key]


def load_lvis_minival(cfg: dict | None = None) -> dict:
    cfg = cfg or load_paths()
    ann = yolo_path(cfg, "lvis_minival_ann")
    return json.loads(ann.read_text(encoding="utf-8"))


def load_class_texts(cfg: dict | None = None) -> tuple[list[str], list]:
    cfg = cfg or load_paths()
    raw = json.loads(yolo_path(cfg, "class_texts").read_text(encoding="utf-8"))
    names = [t[0] if isinstance(t, list) else str(t) for t in raw]
    return names, raw


def cat_id_to_index(lvis: dict) -> dict[int, int]:
    cats = sorted(lvis["categories"], key=lambda c: c["id"])
    return {c["id"]: i for i, c in enumerate(cats)}


def index_to_cat_id(lvis: dict) -> dict[int, int]:
    cats = sorted(lvis["categories"], key=lambda c: c["id"])
    return {i: c["id"] for i, c in enumerate(cats)}
