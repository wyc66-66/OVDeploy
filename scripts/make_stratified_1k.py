"""Stratified 1k minival subset excluding dev 500."""
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
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    cfg = load_paths()
    dev_path = ROOT / cfg["outputs"]["dev_subset"]
    dev_ids = set(json.loads(dev_path.read_text(encoding="utf-8"))["image_ids"])

    data = load_lvis_minival(cfg)
    cat_freq = {c["id"]: c.get("frequency", "c") for c in data["categories"]}
    img_to_freqs: dict[int, list[str]] = defaultdict(list)
    for a in data["annotations"]:
        if a["image_id"] in dev_ids:
            continue
        img_to_freqs[a["image_id"]].append(cat_freq.get(a["category_id"], "c"))

    buckets: dict[str, list[int]] = defaultdict(list)
    for img in data["images"]:
        iid = img["id"]
        if iid in dev_ids:
            continue
        freqs = img_to_freqs.get(iid, ["c"])
        if "r" in freqs:
            buckets["r"].append(iid)
        elif "c" in freqs:
            buckets["c"].append(iid)
        else:
            buckets["f"].append(iid)

    rng = random.Random(args.seed)
    per = max(1, args.n // 3)
    selected: list[int] = []
    for key in ("r", "c", "f"):
        pool = buckets[key]
        rng.shuffle(pool)
        selected.extend(pool[:per])
    pool = [i for i in buckets["r"] + buckets["c"] + buckets["f"] if i not in selected]
    rng.shuffle(pool)
    while len(selected) < args.n and pool:
        selected.append(pool.pop())
    selected = selected[: args.n]

    out = ROOT / "data/stratified_1k.json"
    out.write_text(
        json.dumps({"n": len(selected), "seed": args.seed, "image_ids": selected}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(selected)} ids -> {out}")


if __name__ == "__main__":
    main()
