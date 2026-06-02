"""Create 500-image dev subset from LVIS minival (CPU)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ovdeploy.paths_util import load_paths, load_lvis_minival


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_paths()
    data = load_lvis_minival(cfg)
    cat_freq = {c["id"]: c.get("frequency", "c") for c in data["categories"]}
    img_to_freqs: dict[int, list[str]] = defaultdict(list)
    for a in data["annotations"]:
        img_to_freqs[a["image_id"]].append(cat_freq.get(a["category_id"], "c"))

    buckets: dict[str, list[int]] = defaultdict(list)
    for img in data["images"]:
        iid = img["id"]
        freqs = img_to_freqs.get(iid, ["c"])
        if "r" in freqs:
            buckets["r"].append(iid)
        elif "c" in freqs:
            buckets["c"].append(iid)
        else:
            buckets["f"].append(iid)

    rng = random.Random(args.seed)
    per_bucket = max(1, args.n // 3)
    selected: list[int] = []
    for key in ("r", "c", "f"):
        pool = buckets[key]
        rng.shuffle(pool)
        selected.extend(pool[:per_bucket])
    all_ids = [im["id"] for im in data["images"]]
    remaining = [i for i in all_ids if i not in selected]
    rng.shuffle(remaining)
    while len(selected) < args.n and remaining:
        selected.append(remaining.pop())
    selected = selected[: args.n]

    out = ROOT / cfg["outputs"]["dev_subset"]
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"n": len(selected), "seed": args.seed, "image_ids": selected}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(selected)} ids -> {out}")


if __name__ == "__main__":
    main()
