"""Verify train vs eval episode image_id sets are disjoint."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ovdeploy.episode import load_episodes_dir


def collect_ids(directory: Path) -> set[int]:
    ids: set[int] = set()
    if not directory.is_dir():
        return ids
    for json_path in directory.rglob("*.json"):
        from ovdeploy.episode import load_episode

        ids.update(load_episode(json_path).image_ids)
    return ids


def main() -> None:
    dev_root = ROOT / "data/episodes/dev"
    train_root = ROOT / "data/episodes/train"
    dev_ids = collect_ids(dev_root)
    train_ids = collect_ids(train_root)
    overlap = dev_ids & train_ids
    report = {
        "n_dev_images": len(dev_ids),
        "n_train_images": len(train_ids),
        "overlap_count": len(overlap),
        "overlap_sample": sorted(overlap)[:20],
        "pass": len(overlap) == 0,
    }
    out = ROOT / "reports/REPORT_leakage_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
