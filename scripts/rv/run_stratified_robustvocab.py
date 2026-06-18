"""Stratified held-out 1k eval for RobustVocab."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from robustvocab.paths_util import load_config, reports_dir


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
    parser.add_argument("--backbone", default="yolo")
    args = parser.parse_args()

    cfg = load_config()
    from ovdeploy.b0_cache import ensure_b0_preds
    from ovdeploy.episode import Episode, EpisodeVocab
    from ovdeploy.infer import run_episode_infer
    from ovdeploy.paths_util import load_lvis_minival
    from ovdeploy.vocab import freq_sorted_cat_ids
    from robustvocab.backend_ext import get_backend
    from robustvocab.pipeline import run_robustvocab_episode

    use_gpu = args.gpu
    if use_gpu:
        try:
            import torch

            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False

    if not use_gpu:
        raise SystemExit("GPU required — run with --gpu on WSL")

    strat_path = cfg["_ovdeploy"] / "data/stratified_1k.json"
    strat = json.loads(strat_path.read_text(encoding="utf-8"))
    image_ids = strat["image_ids"][: args.max_images]
    lvis = load_lvis_minival()
    device = "cuda:0"
    b0_cache = ensure_b0_preds(image_ids, lvis, device=device, backbone=args.backbone)
    backend = get_backend(device=device)

    rows = []
    for vs in (10, 30, 100):
        vocab = freq_sorted_cat_ids(lvis)[:vs]
        ep = Episode(
            episode_id=f"strat_v{vs}",
            image_ids=image_ids,
            vocab=EpisodeVocab(cat_ids=vocab),
            vocab_size=vs,
            noise="none",
            split="stratified_1k",
        )
        for method in ("B5_subset", "VG_full_strict", "RV_full"):
            if method == "B5_subset":
                r = run_episode_infer(
                    ep,
                    "B5_subset",
                    lvis,
                    b0_preds_by_image=b0_cache,
                    episode_vocab_for_oov=vocab,
                    backbone=args.backbone,
                )
            else:
                r = run_robustvocab_episode(
                    ep,
                    method,
                    lvis,
                    backend,
                    device=device,
                    b0_preds_by_image=b0_cache,
                    backbone=args.backbone,
                    deployment_strict=True,
                )
            rows.append(
                {
                    "method": method,
                    "vocab_size": vs,
                    "split": "stratified_1k",
                    "EpisodicAP_mean": r["EpisodicAP_mean"],
                    "OOV_FP_mean": r["OOV_FP_mean"],
                    "gpu_used": True,
                    "n_images": len(image_ids),
                }
            )

    report = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_hash(),
        "gpu_used": True,
        "n_images": len(image_ids),
        "rows": rows,
    }
    out = reports_dir() / "REPORT_RV_stratified_1k.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
