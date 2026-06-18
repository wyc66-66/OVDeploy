"""Run VocabGuard evaluation -> REPORT_VG_*.json"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vocabguard.paths_util import checkpoints_dir, load_config, reports_dir


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


def aggregate(rows: list[dict], key: str = "EpisodicAP_mean") -> float:
    vals = [r[key] for r in rows if key in r]
    return sum(vals) / max(len(vals), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--config-key", default="dev_v10_s42_none")
    parser.add_argument("--noise", default="")
    parser.add_argument("--vocab-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--methods", default="B5_subset,B4_clip,B1_oracle,VG_router,VG_full,M2_calib")
    parser.add_argument("--backbone", default="yolo")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--proxy", action="store_true")
    parser.add_argument("--report", default="reports/REPORT_VG_dev.json")
    parser.add_argument("--calib-ckpt", default="")
    args = parser.parse_args()

    cfg = load_config()
    sys.path.insert(0, str(cfg["_ovdeploy"]))

    from ovdeploy.b0_cache import ensure_b0_preds
    from ovdeploy.episode import load_episodes_dir
    from ovdeploy.paths_util import load_lvis_minival

    use_gpu = args.gpu and not args.proxy
    if use_gpu:
        try:
            import torch

            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False

    device = "cuda:0" if use_gpu else "cpu"
    ep_root = cfg["_ovdeploy"] / "data/episodes" / args.split

    if args.config_key:
        ep_dirs = [ep_root / args.config_key]
    else:
        ep_dirs = sorted(ep_root.glob("*"))

    ep_dirs = [d for d in ep_dirs if d.is_dir()]
    if not ep_dirs:
        raise SystemExit(f"No episode dirs under {ep_root}")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    lvis = load_lvis_minival()
    calib = Path(args.calib_ckpt) if args.calib_ckpt else checkpoints_dir() / "calib_head_s42.pt"

    rows = []
    for ep_dir in ep_dirs:
        eps = load_episodes_dir(ep_dir)[: args.max_episodes]
        iids = list({i for ep in eps for i in ep.image_ids})
        b0_cache = None
        if use_gpu:
            b0_cache = ensure_b0_preds(iids, lvis, device=device, backbone=args.backbone)

        if args.backbone in ("yolo", "yolo_ext"):
            from vocabguard.backend import get_extended_backend
            from vocabguard.infer_pipeline import run_vocabguard_episode

            backend = get_extended_backend("yolo", device=device)
        else:
            from ovdeploy.backends.base import get_backend
            from vocabguard.infer_pipeline import run_vocabguard_episode

            backend = get_backend(args.backbone, device=device)

        for method in methods:
            acc = []
            for ep in eps:
                if method in ("B0_full", "B5_subset") and method == "B0_full":
                    from vocabguard.infer_pipeline import run_baseline_via_ovdeploy

                    r = run_baseline_via_ovdeploy(
                        ep, method, lvis, args.max_images, b0_cache, args.backbone
                    )
                elif method == "B5_subset" and args.backbone != "yolo":
                    from vocabguard.infer_pipeline import run_baseline_via_ovdeploy

                    r = run_baseline_via_ovdeploy(
                        ep, "B5_subset", lvis, args.max_images, b0_cache, args.backbone
                    )
                elif method in ("B0_full",):
                    from vocabguard.infer_pipeline import run_baseline_via_ovdeploy

                    r = run_baseline_via_ovdeploy(
                        ep, method, lvis, args.max_images, b0_cache, args.backbone
                    )
                else:
                    ckpt = calib if method == "M2_calib" else None
                    r = run_vocabguard_episode(
                        ep,
                        method,
                        lvis,
                        backend,
                        max_images=args.max_images,
                        device=device,
                        b0_preds_by_image=b0_cache,
                        calib_ckpt=ckpt,
                        backbone=args.backbone,
                    )
                acc.append(r)
            rows.append(
                {
                    "method": method,
                    "config": ep_dir.name,
                    "EpisodicAP_mean": aggregate(acc),
                    "OOV_FP_mean": aggregate(acc, "OOV_FP_mean"),
                    "FP_nonGT_mean": aggregate(acc, "FP_nonGT_mean"),
                    "n_episodes": len(acc),
                    "gpu_used": use_gpu,
                }
            )

    report = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_hash(),
        "split": args.split,
        "backbone": args.backbone,
        "gpu_used": use_gpu,
        "rows": rows,
        "summary": {
            "best_epiap": max(rows, key=lambda r: r["EpisodicAP_mean"]) if rows else {},
            "best_oov": min(rows, key=lambda r: r["OOV_FP_mean"]) if rows else {},
        },
    }

    out = ROOT / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
