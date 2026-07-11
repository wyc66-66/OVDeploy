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


def select_image_ids(strat: dict, args: argparse.Namespace) -> list[int]:
    ids = list(strat["image_ids"][: args.max_images])
    if args.num_shards > 1:
        ids = [ids[i] for i in range(args.shard_id, len(ids), args.num_shards)]
    else:
        start = max(0, args.start_index)
        end = args.end_index if args.end_index is not None else len(ids)
        ids = ids[start:end]
    return ids


def run_metrics_phase(
    image_ids: list[int],
    b0_cache: dict,
    lvis: dict,
    backbone: str,
    device: str,
) -> list[dict]:
    from ovdeploy.episode import Episode, EpisodeVocab
    from ovdeploy.infer import run_episode_infer
    from ovdeploy.vocab import freq_sorted_cat_ids

    rows: list[dict] = []
    print("Phase 2/2: stratified B0/B5 metrics...", flush=True)
    for vs in [10, 30, 100]:
        vocab = freq_sorted_cat_ids(lvis)[:vs]
        ep = Episode(
            episode_id=f"strat_{backbone}_v{vs}",
            image_ids=image_ids,
            vocab=EpisodeVocab(cat_ids=vocab),
            vocab_size=vs,
            noise="none",
            split="stratified_1k",
        )
        for bl in ("B0_full", "B5_subset"):
            print(f"Starting |V|={vs} {bl}...", flush=True)
            r = run_episode_infer(
                ep,
                bl,
                lvis,
                b0_preds_by_image=b0_cache,
                episode_vocab_for_oov=vocab,
                backbone=backbone,
                device=device,
            )
            rows.append(
                {
                    "vocab_size": vs,
                    "baseline": bl,
                    "backbone": backbone,
                    "EpisodicAP_mean": r["EpisodicAP_mean"],
                    "OOV_FP_mean": r["OOV_FP_mean"],
                    "mode": device,
                    "n_images": r["n_images"],
                }
            )
            print(f"|V|={vs} {bl} OOV={r['OOV_FP_mean']:.3f}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--report",
        default="reports/REPORT_4b_gdino_stratified_1k.json",
    )
    parser.add_argument(
        "--backbone",
        default="glip",
        help="glip (GDINO-tiny) or gdino_base",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only build B0 cache (Phase 1); no report.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Skip Phase 1; run metrics using existing B0 cache.",
    )
    args = parser.parse_args()

    from ovdeploy.b0_cache import ensure_b0_preds
    from ovdeploy.paths_util import load_lvis_minival, load_paths

    cfg = load_paths()
    lvis = load_lvis_minival(cfg)
    strat = json.loads((ROOT / "data/stratified_1k.json").read_text(encoding="utf-8"))
    full_ids = list(strat["image_ids"][: args.max_images])
    image_ids = select_image_ids(strat, args)
    device = args.device

    if args.cache_only and args.metrics_only:
        raise SystemExit("Use only one of --cache-only or --metrics-only")

    shard_note = ""
    if args.num_shards > 1:
        shard_note = f" shard {args.shard_id}/{args.num_shards}"
    elif args.start_index or args.end_index is not None:
        shard_note = f" slice [{args.start_index}:{args.end_index or len(full_ids)}]"

    print(
        f"GDINO stratified: n={len(image_ids)} device={device} backbone={args.backbone}{shard_note}",
        flush=True,
    )

    b0_cache: dict = {}
    if not args.metrics_only:
        print("Phase 1/2: building B0 full-vocab cache...", flush=True)
        b0_cache = ensure_b0_preds(image_ids, lvis, device=device, backbone=args.backbone)
        print(f"Phase 1/2 done: B0 cache ready ({len(b0_cache)} images)", flush=True)
        if args.cache_only:
            print("cache-only: stopping before Phase 2")
            return

    if args.metrics_only:
        print("metrics-only: loading B0 cache for full image set...", flush=True)
        b0_cache = ensure_b0_preds(full_ids, lvis, device=device, backbone=args.backbone)
        image_ids = full_ids

    rows = run_metrics_phase(image_ids, b0_cache, lvis, args.backbone, device)

    note = None
    if len(full_ids) < strat.get("n", 1000):
        note = f"subset n={len(full_ids)} of stratified_1k; same frequency-top-|V| protocol"
    if shard_note:
        note = (note + "; " if note else "") + shard_note.strip()

    report = {
        "status": "ok",
        "split": "stratified_1k",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_hash(),
        "gpu_used": device.startswith("cuda"),
        "backbone": args.backbone,
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
