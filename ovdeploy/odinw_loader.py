"""Load native ODinW COCO domains from data/odinw/<slug>/."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def domain_dir(slug: str) -> Path:
    return ROOT / "data" / "odinw" / slug


def load_domain_meta(slug: str) -> dict[str, Any]:
    p = domain_dir(slug) / "domain.json"
    return json.loads(p.read_text(encoding="utf-8"))


def load_odinw_coco(slug: str) -> dict[str, Any]:
    ann_path = domain_dir(slug) / "annotations.json"
    if not ann_path.is_file():
        raise FileNotFoundError(f"Missing {ann_path}")
    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    coco["_slug"] = slug
    coco["_images_dir"] = domain_dir(slug) / "images"
    return coco


def image_path(slug: str, file_name: str) -> Path:
    fn = file_name.replace("\\", "/")
    base = domain_dir(slug) / "images"
    for candidate in (base / fn, base / Path(fn).name):
        if candidate.is_file():
            return candidate
    return base / fn


def cat_id_to_name(coco: dict) -> dict[int, str]:
    return {c["id"]: c["name"] for c in coco.get("categories", [])}


def all_category_ids(coco: dict) -> list[int]:
    return [c["id"] for c in sorted(coco["categories"], key=lambda x: x["id"])]


def gt_by_image(coco: dict) -> dict[int, dict[str, list]]:
    out: dict[int, dict[str, list]] = {}
    for ann in coco.get("annotations", []):
        iid = ann["image_id"]
        out.setdefault(iid, {"boxes": [], "cat_ids": []})
        out[iid]["boxes"].append(ann["bbox"])
        out[iid]["cat_ids"].append(ann["category_id"])
    return out


def list_images(coco: dict, max_images: int = 0) -> list[dict]:
    images = list(coco.get("images", []))
    images.sort(key=lambda x: x["id"])
    if max_images:
        images = images[:max_images]
    return images


def sample_episode_vocab(
    coco: dict,
    domain_classes: list[str],
    vocab_size: int,
    seed: int,
) -> list[int]:
    """Pick COCO category ids matching domain class names; pad with frequent cats."""
    name_to_id = {c["name"].lower(): c["id"] for c in coco["categories"]}
    matched: list[int] = []
    for cls in domain_classes:
        key = cls.lower().strip()
        if key in name_to_id:
            matched.append(name_to_id[key])
        else:
            for nm, cid in name_to_id.items():
                if key in nm or nm in key:
                    matched.append(cid)
                    break
    matched = list(dict.fromkeys(matched))
    if len(matched) < vocab_size:
        rng = random.Random(seed)
        rest = [c["id"] for c in coco["categories"] if c["id"] not in matched]
        rng.shuffle(rest)
        matched.extend(rest[: max(0, vocab_size - len(matched))])
    return matched[:vocab_size]


def class_texts_for_ids(
    coco: dict,
    cat_ids: list[int],
    meta: dict[str, Any] | None = None,
) -> tuple[list[str], list[list[str]]]:
    id2name = cat_id_to_name(coco)
    prompt_map = (meta or {}).get("prompts") or {}
    names = [id2name.get(c, str(c)) for c in cat_ids]
    texts: list[list[str]] = []
    for cid, name in zip(cat_ids, names):
        key = name.lower()
        if key in prompt_map:
            texts.append(list(prompt_map[key]))
        else:
            texts.append([name])
    return names, texts
