"""nuScenes 23-class taxonomy and category name → cat_id mapping."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# nuScenes sample_annotation category.name → pilot cat_id (1..23)
NUSCENES_CATEGORY_TO_CAT_ID: dict[str, int] = {
    "vehicle.car": 1,
    "vehicle.truck": 2,
    "vehicle.bus.bendy": 3,
    "vehicle.bus.rigid": 3,
    "vehicle.trailer": 4,
    "vehicle.construction": 5,
    "human.pedestrian.adult": 6,
    "human.pedestrian.child": 6,
    "human.pedestrian.construction_worker": 6,
    "human.pedestrian.police_officer": 6,
    "human.pedestrian.personal_mobility": 6,
    "human.pedestrian.stroller": 6,
    "human.pedestrian.wheelchair": 6,
    "vehicle.motorcycle": 7,
    "vehicle.bicycle": 8,
    "movable_object.trafficcone": 9,
    "movable_object.barrier": 10,
    "vehicle.emergency.ambulance": 18,
    "vehicle.emergency.police": 19,
    "animal": 23,
    "static_object.bicycle_rack": 22,
    "movable_object.debris": 21,
    "movable_object.pushable_pullable": 20,
}


@dataclass
class NuScenesTaxonomy:
    num_classes: int
    all_cat_ids: list[int]
    cat_id_to_name: dict[int, str]
    cat_id_to_prompt: dict[int, str]
    class_names: list[str]
    class_texts_raw: list[list[str]]
    cid2idx: dict[int, int]

    def map_category_name(self, name: str) -> int | None:
        if name in NUSCENES_CATEGORY_TO_CAT_ID:
            return NUSCENES_CATEGORY_TO_CAT_ID[name]
        prefix = name.split(".")[0]
        if prefix == "vehicle" and "bicycle" in name:
            return 8
        if prefix == "human":
            return 6
        return None

    def prompts_for_vocab(self, vocab_ids: list[int]) -> dict[int, list[str]]:
        return {cid: [self.cat_id_to_prompt[cid]] for cid in vocab_ids}


def load_taxonomy(class_map_path: Path | None = None) -> NuScenesTaxonomy:
    if class_map_path is None:
        class_map_path = (
            Path(__file__).resolve().parents[2].parent
            / "outreach-mars"
            / "pilot"
            / "config"
            / "nuscenes_class_map.yaml"
        )
    with open(class_map_path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    cat_id_to_name: dict[int, str] = {}
    cat_id_to_prompt: dict[int, str] = {}
    for c in cfg["classes"]:
        cid = int(c["id"]) + 1
        cat_id_to_name[cid] = c["nuscenes_name"]
        cat_id_to_prompt[cid] = c["prompt"]

    num = int(cfg.get("num_classes", len(cat_id_to_name)))
    all_cat_ids = sorted(cat_id_to_name.keys())
    class_names = [cat_id_to_name[cid] for cid in all_cat_ids]
    class_texts_raw = [[cat_id_to_prompt[cid]] for cid in all_cat_ids]
    cid2idx = {cid: i for i, cid in enumerate(all_cat_ids)}

    return NuScenesTaxonomy(
        num_classes=num,
        all_cat_ids=all_cat_ids,
        cat_id_to_name=cat_id_to_name,
        cat_id_to_prompt=cat_id_to_prompt,
        class_names=class_names,
        class_texts_raw=class_texts_raw,
        cid2idx=cid2idx,
    )


def load_pilot_config(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config" / "nuscenes_pilot.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_nuscenes_root(cfg: dict[str, Any]) -> Path:
    import sys

    if sys.platform != "win32" and cfg.get("nuscenes_root_wsl"):
        root = Path(cfg["nuscenes_root_wsl"])
    else:
        root = Path(cfg["nuscenes_root"])
        if not root.is_dir() and cfg.get("nuscenes_root_wsl"):
            root = Path(cfg["nuscenes_root_wsl"])
    return root


def resolve_class_map_path(cfg: dict[str, Any]) -> Path:
    p = Path(cfg["class_map"])
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p
    return p
