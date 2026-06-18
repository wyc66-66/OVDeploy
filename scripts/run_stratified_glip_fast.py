#!/usr/bin/env python3
"""Fast GDINO-T stratified held-out (B0+B5, configurable n_images)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--report",
        default="reports/REPORT_4b_gdino_stratified_1k.json",
    )
    args = parser.parse_args()

    from ovdeploy.b0_cache import ensure_b0_preds
    from ovdeploy.episode import Episode, EpisodeVocab
    from ovdeploy.infer import run_episode_infer
    from ovdeploy.paths_util import load_lvis_minival, load_paths
    from ovdeploy.vocab import freq_sorted_cat_ids

    cfg = load_paths()
    lvis = load_lvis_minival(cfg)
    strat = json.loads((ROOT / "data/stratified_1k.json").read_text(encoding="utf-8"))
    image_ids = strat["image_ids"][: args.max_images]
    device = args.device

    print(f"GDINO stratified: n={len(image_ids)} device={device}", flush=True)
    b0_cache = ensure_b0_preds(image_ids, lvis, device=device, backbone="glip")

    rows = []
    for vs in [10, 30, 100]:
        vocab = freq_sorted_cat_ids(lvis)[:vs]
        ep = Episode(
            episode_id=f"strat_gdino_v{vs}",
            image_ids=image_ids,
            vocab=EpisodeVocab(cat_ids=vocab),
            vocab_size=vs,
            noise="none",
            split="stratified_1k",
        )
        for bl in ("B0_full", "B5_subset"):
            r = run_episode_infer(
                ep,
                bl,
                lvis,
                b0_preds_by_image=b0_cache,
                episode_vocab_for_oov=vocab,
                backbone="glip",
                device=device,
            )
            rows.append(
                {
                    "vocab_size": vs,
                    "baseline": bl,
                    "backbone": "glip",
                    "EpisodicAP_mean": r["EpisodicAP_mean"],
                    "OOV_FP_mean": r["OOV_FP_mean"],
                    "mode": device,
                    "n_images": r["n_images"],
                }
            )
            print(f"|V|={vs} {bl} OOV={r['OOV_FP_mean']:.3f}", flush=True)

    note = None
    if len(image_ids) < strat.get("n", 1000):
        note = f"subset n={len(image_ids)} of stratified_1k; same frequency-top-|V| protocol"

    report = {
        "status": "ok",
        "split": "stratified_1k",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_hash(),
        "gpu_used": device.startswith("cuda"),
        "backbone": "glip",
        "metrics_version": "v2",
        "n_images": len(image_ids),
        "note": note,
        "rows": rows,
    }
    out = ROOT / args.report
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
