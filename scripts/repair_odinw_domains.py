"""Re-normalize ODinW domains extracted from HF zips (pick largest split)."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.download_odinw_roboflow import DOMAINS, _finalize_domain, _merge_coco


def repair_slug(slug: str) -> bool:
    base = ROOT / "data" / "odinw" / slug
    if not base.is_dir():
        return False
    cands = list(base.rglob("annotations_without_background.json"))
    if not cands:
        return False
    best = max(cands, key=lambda p: len(json.loads(p.read_text(encoding="utf-8")).get("images", [])))
    ann = json.loads(best.read_text(encoding="utf-8"))
    ann = _merge_coco(ann)
    images_dir = base / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    split_dir = best.parent
    for im in ann.get("images", []):
        fn = im["file_name"]
        src = split_dir / fn
        if not src.is_file():
            for found in base.rglob(fn):
                if found.is_file():
                    src = found
                    break
        dest = images_dir / fn
        if src.is_file():
            shutil.copy2(src, dest)
    meta = DOMAINS[slug]
    return _finalize_domain(base, slug, meta, ann, images_dir)


def main() -> None:
    for slug in ("packages", "fryingpan"):
        ok = repair_slug(slug)
        print(f"{slug}: {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
