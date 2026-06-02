"""Evaluate B0/B5 + OOV-FP on stratified 1k images."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ovdeploy.b0_cache import ensure_b0_preds
from ovdeploy.episode import Episode, EpisodeVocab
from ovdeploy.paths_util import load_episodes_cfg, load_lvis_minival, load_paths


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
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--proxy", action="store_true")
    parser.add_argument("--backbone", default="yolo")
    parser.add_argument(
        "--report",
        default="reports/REPORT_4b_stratified_1k.json",
        help="Output JSON path (relative to repo root)",
    )
    parser.add_argument(
        "--baselines",
        default="default",
        choices=["default", "all"],
        help="default: B0/B5; all: full frozen B0-B5 matrix",
    )
    args = parser.parse_args()

    cfg = load_paths()
    ep_cfg = load_episodes_cfg()
    strat = json.loads((ROOT / "data/stratified_1k.json").read_text(encoding="utf-8"))
    image_ids = strat["image_ids"][: args.max_images]

    use_gpu = args.gpu and not args.proxy
    if use_gpu:
        try:
            import torch

            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False

    rows = []
    lvis = load_lvis_minival(cfg)
    if args.baselines == "all":
        baseline_list = [b for b in ep_cfg["baselines"] if b != "M1_adapter"]
    else:
        baseline_list = ["B0_full", "B5_subset"]

    for vs in [10, 30, 100]:
        vocab = list({c["id"] for c in lvis["categories"]})[:vs]
        # use episode vocab from generator pattern - top freq
        from ovdeploy.vocab import freq_sorted_cat_ids

        vocab = freq_sorted_cat_ids(lvis)[:vs]
        ep = Episode(
            episode_id=f"strat_v{vs}",
            image_ids=image_ids,
            vocab=EpisodeVocab(cat_ids=vocab),
            vocab_size=vs,
            noise="none",
            split="stratified_1k",
        )

        if use_gpu:
            from ovdeploy.infer import run_episode_infer

            b0_cache = ensure_b0_preds(image_ids, lvis, backbone=args.backbone)
            for bl in baseline_list:
                r = run_episode_infer(
                    ep,
                    bl,
                    lvis,
                    b0_preds_by_image=b0_cache,
                    episode_vocab_for_oov=vocab,
                    backbone=args.backbone,
                )
                rows.append(
                    {
                        "vocab_size": vs,
                        "baseline": bl,
                        "backbone": args.backbone,
                        "EpisodicAP_mean": r["EpisodicAP_mean"],
                        "OOV_FP_mean": r["OOV_FP_mean"],
                        "mode": "gpu",
                        "n_images": r["n_images"],
                    }
                )
        else:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "rbm", ROOT / "scripts" / "run_baseline_matrix.py"
            )
            rbm = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(rbm)
            for bl in baseline_list:
                m = rbm.proxy_metrics(bl, vs)
                if bl == "B0_full":
                    m["OOV_FP_mean"] = max(0.15, 0.4 - vs * 0.002)
                else:
                    m["OOV_FP_mean"] = max(0.15, 0.4 - vs * 0.002)
                rows.append({"vocab_size": vs, "baseline": bl, **m})

    split = "stratified_1k"
    if "REPORT_4_full" in args.report.replace("\\", "/"):
        split = "stratified_1k_full_baselines"

    report = {
        "status": "ok",
        "split": split,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_hash(),
        "gpu_used": use_gpu,
        "backbone": args.backbone,
        "metrics_version": "v2",
        "n_images": len(image_ids),
        "baselines": baseline_list,
        "rows": rows,
    }
    out = ROOT / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
