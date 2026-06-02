"""Download ODinW-13 COCO domains into data/odinw/ (GLIP HF mirror)."""
from __future__ import annotations

import json
import os
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BLOB = "https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/odinw"
HF_REPO = "GLIPModel/GLIP"

ODINW_PROMPTS = {
    "package": ["package", "parcel", "cardboard box", "delivery box"],
    "pothole": ["pothole", "road pothole", "pavement hole"],
    "crab": ["crab", "sea crab"],
    "lobster": ["lobster", "maine lobster"],
    "shrimp": ["shrimp", "prawn"],
    "shellfish": ["shellfish", "seafood"],
}

DOMAINS = {
    "aquarium": {
        "paths": ["Aquarium/Aquarium Combined.v2-raw-1024.coco"],
        "hf_prefix": "odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco",
        "domain": "Aquarium",
        "classes": [
            "fish",
            "jellyfish",
            "penguin",
            "puffin",
            "shark",
            "starfish",
            "stingray",
        ],
    },
    "packages": {
        "paths": [
            "Packages/Packages Raw.v2-raw-1024.coco",
            "Packages/Packages.v2-raw-1024.coco",
        ],
        "hf_prefix": "odinw/Packages/Packages Raw.v2-raw-1024.coco",
        "hf_zip": "odinw_35/Packages.zip",
        "domain": "Packages",
        "classes": ["package"],
        "prompts": {"package": ODINW_PROMPTS["package"]},
    },
    "fryingpan": {
        "paths": [
            "Raccoon/Raccoon.v2-raw-1024.coco",
            "Raccoon/Raccoon.v1-raw-1024.coco",
        ],
        "hf_prefix": "odinw/Raccoon/Raccoon.v2-raw-1024.coco",
        "hf_zip": "odinw_35/Raccoon.zip",
        "domain": "Raccoon",
        "classes": ["raccoon"],
        "note": "ODinW-13 Raccoon domain (FryingPan not in ODinW-13)",
    },
    "thermal": {
        "hf_zip": "odinw_35/thermalDogsAndPeople.zip",
        "domain": "ThermalDogsAndPeople",
        "classes": ["dog", "person"],
        "note": "ODinW-13 Thermal Dogs and People",
    },
    "pothole": {
        "hf_zip": "odinw_35/pothole.zip",
        "domain": "Pothole",
        "classes": ["pothole"],
        "prompts": {"pothole": ODINW_PROMPTS["pothole"]},
        "note": "ODinW-13 Pothole",
    },
    "shellfish": {
        "hf_zip": "odinw_35/ShellfishOpenImages.zip",
        "domain": "ShellfishOpenImages",
        "classes": ["crab", "lobster", "shrimp", "shellfish"],
        "prompts": {
            "crab": ODINW_PROMPTS["crab"],
            "lobster": ODINW_PROMPTS["lobster"],
            "shrimp": ODINW_PROMPTS["shrimp"],
            "shellfish": ODINW_PROMPTS["shellfish"],
        },
        "note": "ODinW-13 Shellfish OpenImages",
    },
    "aerial": {
        "hf_zip": "odinw_35/AerialMaritimeDrone.zip",
        "domain": "AerialMaritimeDrone",
        "classes": [],
        "note": "ODinW-13 Aerial Maritime Drone",
    },
    "cottontail": {
        "hf_zip": "odinw_35/CottontailRabbits.zip",
        "domain": "CottontailRabbits",
        "classes": ["rabbit"],
        "note": "ODinW-13 Cottontail Rabbits",
    },
    "egohands": {
        "hf_zip": "odinw_35/EgoHands.zip",
        "domain": "EgoHands",
        "classes": ["hand"],
        "note": "ODinW-13 EgoHands generic",
    },
    "mushrooms": {
        "hf_zip": "odinw_35/NorthAmericaMushrooms.zip",
        "domain": "NorthAmericaMushrooms",
        "classes": [],
        "note": "ODinW-13 North America Mushrooms",
    },
    "pascalvoc": {
        "hf_zip": "odinw_35/PascalVOC.zip",
        "domain": "PascalVOC",
        "classes": [],
        "note": "ODinW-13 Pascal VOC",
    },
    "pistols": {
        "hf_zip": "odinw_35/pistols.zip",
        "domain": "pistols",
        "classes": ["pistol"],
        "note": "ODinW-13 pistols",
    },
    "vehicles": {
        "hf_zip": "odinw_35/VehiclesOpenImages.zip",
        "domain": "VehiclesOpenImages",
        "classes": [],
        "note": "ODinW-13 Vehicles OpenImages",
    },
}

ALL13_SLUGS = (
    "aquarium,aerial,cottontail,egohands,mushrooms,packages,pascalvoc,"
    "pistols,fryingpan,thermal,pothole,shellfish,vehicles"
)


def _blob_url(*parts: str) -> str:
    encoded = "/".join(urllib.parse.quote(p, safe="") for p in parts)
    return f"{BLOB}/{encoded}"


def _fetch(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ovdeploy/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        if len(data) < 100:
            return False
        dest.write_bytes(data)
        return True
    except Exception as e:
        print(f"FAIL {url}: {e}")
        return False


def _merge_coco(ann: dict) -> dict:
    images = ann.get("images", [])
    for im in images:
        fn = im.get("file_name", "")
        if "/" in fn:
            im["file_name"] = Path(fn).name
    ann["images"] = images
    return ann


def _finalize_domain(base: Path, slug: str, meta: dict, ann: dict, images_dir: Path) -> bool:
    ann = _merge_coco(ann)
    (base / "annotations.json").write_text(json.dumps(ann), encoding="utf-8")
    n_ok = sum(1 for im in ann.get("images", []) if (images_dir / im["file_name"]).is_file())
    classes = meta.get("classes") or [c["name"] for c in ann.get("categories", [])]
    domain_json = {
        "domain": meta["domain"],
        "classes": classes,
        "max_images": 100,
        "source": "roboflow_native",
        "coco_path": meta.get("hf_prefix", slug),
    }
    if meta.get("prompts"):
        domain_json["prompts"] = meta["prompts"]
    if meta.get("note"):
        domain_json["note"] = meta["note"]
    (base / "domain.json").write_text(json.dumps(domain_json, indent=2), encoding="utf-8")
    print(f"{slug}: {n_ok}/{len(ann.get('images', []))} images, {len(ann.get('categories', []))} cats")
    return n_ok >= 30


def download_domain_hf_zip(slug: str, meta: dict) -> bool:
    zip_rel = meta.get("hf_zip")
    if not zip_rel:
        return False
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    import zipfile

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False

    base = ROOT / "data" / "odinw" / slug
    base.mkdir(parents=True, exist_ok=True)
    try:
        zpath = hf_hub_download(
            repo_id=HF_REPO,
            filename=zip_rel,
            local_dir=str(ROOT / "data" / "odinw_zips"),
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        print(f"HF zip {zip_rel}: {e}")
        return False
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(base)
    ann_candidates = list(base.rglob("annotations_without_background.json"))
    if not ann_candidates:
        ann_candidates = list(base.rglob("*annotations*.json"))
    if not ann_candidates:
        return False
    best = max(
        ann_candidates,
        key=lambda p: len(json.loads(p.read_text(encoding="utf-8")).get("images", [])),
    )
    ann = json.loads(best.read_text(encoding="utf-8"))
    images_dir = base / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    split_dir = best.parent
    for im in ann.get("images", []):
        fn = Path(im.get("file_name", "")).name
        im["file_name"] = fn
        src = split_dir / fn
        if not src.is_file():
            src = split_dir / "images" / fn
        if not src.is_file():
            for found in base.rglob(fn):
                if found.is_file():
                    src = found
                    break
        dest = images_dir / fn
        if src.is_file() and not dest.is_file():
            shutil.copy2(src, dest)
    return _finalize_domain(base, slug, meta, ann, images_dir)


def download_domain_hf(slug: str, meta: dict) -> bool:
    """Download annotation + images from GLIPModel/GLIP odinw tree."""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        return False

    base = ROOT / "data" / "odinw" / slug
    images_dir = base / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    prefix = meta["hf_prefix"]

    ann_name = None
    for sp in ("valid", "test", "train"):
        candidate = f"{prefix}/{sp}/annotations_without_background.json"
        try:
            files = list_repo_files(HF_REPO)
            if candidate in files:
                ann_name = candidate
                break
        except Exception:
            pass
    if not ann_name:
        for sp in ("valid", "test", "train"):
            candidate = f"{prefix}/{sp}/annotations_without_background.json"
            try:
                hf_hub_download(HF_REPO, candidate, local_dir=str(base / "_hf"))
                ann_name = candidate
                break
            except Exception:
                continue
    if not ann_name:
        print(f"HF: no annotations for {slug}")
        return False

    ann_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=ann_name,
        local_dir=str(base / "_hf"),
        local_dir_use_symlinks=False,
    )
    ann = json.loads(Path(ann_path).read_text(encoding="utf-8"))
    split = ann_name.split("/")[-2]

    try:
        all_files = list_repo_files(HF_REPO)
    except Exception as e:
        print(f"list_repo_files: {e}")
        return False

    img_files = [
        f
        for f in all_files
        if f.startswith(f"{prefix}/{split}/") and f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    for rel in img_files:
        fn = Path(rel).name
        dest = images_dir / fn
        if dest.is_file():
            continue
        try:
            hf_hub_download(
                repo_id=HF_REPO,
                filename=rel,
                local_dir=str(base / "_hf"),
                local_dir_use_symlinks=False,
            )
            src = base / "_hf" / rel
            if src.is_file() and not dest.is_file():
                shutil.copy2(src, dest)
        except Exception as e:
            print(f"  skip {fn}: {e}")

    ok = _finalize_domain(base, slug, meta, ann, images_dir)
    shutil.rmtree(base / "_hf", ignore_errors=True)
    return ok


def download_domain_blob(slug: str, meta: dict, split: str = "valid") -> bool:
    base = ROOT / "data" / "odinw" / slug
    images_dir = base / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    paths = meta.get("paths") or [meta.get("path", "")]
    ann = None
    rel = ""
    used_split = split
    for rel_try in paths:
        for sp in (split, "test", "train"):
            parts = rel_try.split("/")
            ann_url = _blob_url(*parts, sp, "annotations_without_background.json")
            ann_local = base / "_ann_download.json"
            if _fetch(ann_url, ann_local):
                try:
                    ann = json.loads(ann_local.read_text(encoding="utf-8"))
                    rel = rel_try
                    used_split = sp
                    ann_local.unlink(missing_ok=True)
                    break
                except json.JSONDecodeError:
                    pass
        if ann:
            break
    if not ann:
        return False

    rel_parts = rel.split("/")
    for im in ann.get("images", []):
        fn = im["file_name"]
        dest = images_dir / fn
        if dest.is_file():
            continue
        for sp in (used_split, "test", "valid", "train"):
            img_url = _blob_url(*rel_parts, sp, fn)
            if _fetch(img_url, dest):
                break

    return _finalize_domain(base, slug, meta, ann, images_dir)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--domains", default=ALL13_SLUGS)
    p.add_argument("--hf-only", action="store_true")
    args = p.parse_args()
    slugs = [s.strip() for s in args.domains.split(",") if s.strip()]
    ok_all = True
    for slug in slugs:
        if slug not in DOMAINS:
            print(f"Unknown domain {slug}")
            ok_all = False
            continue
        meta = DOMAINS[slug]
        ok = False
        if meta.get("hf_prefix"):
            ok = download_domain_hf(slug, meta)
        if not ok and meta.get("hf_zip"):
            ok = download_domain_hf_zip(slug, meta)
        if not ok and not args.hf_only:
            ok = download_domain_blob(slug, meta)
        ok_all = ok_all and ok
    if not ok_all:
        sys.exit(1)
    print("ODinW download complete")


if __name__ == "__main__":
    main()
