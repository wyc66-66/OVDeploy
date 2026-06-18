"""Ablation study for RobustVocab components."""
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


def aggregate(acc: list[dict], key: str = "EpisodicAP_mean") -> float:
    vals = [r[key] for r in acc if key in r]
    return sum(vals) / max(len(vals), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--config-key", default="dev_v10_s42_missing_class")
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config()
    from ovdeploy.b0_cache import ensure_b0_preds
    from ovdeploy.episode import load_episodes_dir
    from ovdeploy.infer import run_episode_infer
    from ovdeploy.paths_util import load_lvis_minival
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
        raise SystemExit("GPU required for ablation eval — use wsl with --gpu")

    device = "cuda:0"
    lvis = load_lvis_minival()
    ep_dir = cfg["_ovdeploy"] / "data/episodes/dev" / args.config_key
    eps = load_episodes_dir(ep_dir)[: args.max_episodes]
    iids = list({i for ep in eps for i in ep.image_ids})
    b0_cache = ensure_b0_preds(iids, lvis, device=device, backbone="yolo")
    backend = get_backend(device=device)

    variants = [
        ("B5_baseline", "B5_subset", {}, True),
        ("RV_full", "RV_full", {}, True),
        ("w/o_CoOccur", "RV_full", {"use_cooccur": False}, True),
        ("w/o_TwoRound", "RV_full", {"two_round_probe": False}, True),
        ("oracle_hint", "RV_full", {}, False),
    ]

    rows = []
    for label, method, overrides, strict in variants:
        acc = []
        for ep in eps:
            if method == "B5_subset":
                r = run_episode_infer(
                    ep,
                    "B5_subset",
                    lvis,
                    max_images=args.max_images,
                    b0_preds_by_image=b0_cache,
                    episode_vocab_for_oov=list(ep.vocab.cat_ids),
                    backbone="yolo",
                )
            else:
                r = run_robustvocab_episode(
                    ep,
                    method,
                    lvis,
                    backend,
                    max_images=args.max_images,
                    device=device,
                    b0_preds_by_image=b0_cache,
                    backbone="yolo",
                    deployment_strict=strict,
                    recover_overrides=overrides or None,
                )
            acc.append(r)
        rows.append(
            {
                "variant": label,
                "method": method,
                "config": args.config_key,
                "deployment_strict": strict,
                "recover_overrides": overrides,
                "EpisodicAP_mean": aggregate(acc),
                "OOV_FP_mean": aggregate(acc, "OOV_FP_mean"),
                "n_episodes": len(acc),
                "gpu_used": True,
            }
        )

    report = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_hash(),
        "gpu_used": True,
        "rows": rows,
    }
    out = reports_dir() / "REPORT_RV_ablation.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
