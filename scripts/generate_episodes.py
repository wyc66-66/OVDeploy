"""Generate OVDeploy episode JSON files."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ovdeploy.episode import load_episodes_dir
from ovdeploy.generator import generate_all
from ovdeploy.paths_util import load_episodes_cfg, load_paths
from ovdeploy.vocab_shift import vocab_shift_stats


def git_hash() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_paths()
    ep_cfg = load_episodes_cfg()
    dev_path = ROOT / cfg["outputs"]["dev_subset"]
    if not dev_path.is_file():
        print("Run make_dev_subset.py first", file=sys.stderr)
        sys.exit(1)

    dev_ids = json.loads(dev_path.read_text(encoding="utf-8"))["image_ids"]
    lvis = __import__("ovdeploy.paths_util", fromlist=["load_lvis_minival"]).load_lvis_minival(cfg)
    all_ids = [im["id"] for im in lvis["images"]]
    train_pool = [i for i in all_ids if i not in set(dev_ids)][: ep_cfg["train"]["n_images_pool"]]

    out_root = ROOT / "data" / "episodes"
    summary = generate_all(dev_ids, train_pool, ep_cfg, out_root)

    train_eps = load_episodes_dir(out_root / "train" / "train_s42")
    dev_eps = []
    for d in (out_root / "dev").glob("dev_v30_s42_none"):
        dev_eps.extend(load_episodes_dir(d)[:20])
    vshift = vocab_shift_stats(train_eps, dev_eps)

    report = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_hash(),
        "n_dev_images": len(dev_ids),
        "n_train_pool": len(train_pool),
        "counts": summary["counts"],
        "total_episode_files": sum(summary["counts"].values()),
        "vocab_shift": vshift,
    }
    out = ROOT / cfg["reports"]["r1"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
