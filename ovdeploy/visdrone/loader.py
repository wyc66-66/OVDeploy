"""Load VisDrone2019-DET as COCO (native categories, no LVIS)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

VISDRONE_NATIVE_CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def data_dir() -> Path:
    return ROOT / "data" / "visdrone"


def domain_dir() -> Path:
    return data_dir()


def is_ready() -> bool:
    return (domain_dir() / "annotations.json").is_file()


def load_domain_meta() -> dict[str, Any]:
    p = domain_dir() / "domain.json"
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {
        "domain": "VisDrone",
        "classes": VISDRONE_NATIVE_CLASSES,
        "max_images": 100,
        "vocab_source": "native",
    }


def load_visdrone_coco() -> dict[str, Any]:
    ann_path = domain_dir() / "annotations.json"
    if not ann_path.is_file():
        raise FileNotFoundError(
            f"Missing {ann_path}; run: python scripts/setup_visdrone_coco.py"
        )
    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    coco["_slug"] = "visdrone"
    coco["_images_dir"] = domain_dir() / "images"
    return coco


def image_path(file_name: str) -> Path:
    fn = file_name.replace("\\", "/")
    base = domain_dir() / "images"
    for candidate in (base / fn, base / Path(fn).name):
        if candidate.is_file():
            return candidate
    return base / fn
