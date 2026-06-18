"""Run RobustVocab evaluation -> REPORT_RV_*.json"""
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


def aggregate(rows: list[dict], key: str = "EpisodicAP_mean") -> float:
    vals = [r[key] for r in rows if key in r]
    return sum(vals) / max(len(vals), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--config-key", default="dev_v10_s42_none")
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument(
        "--methods",
        default="B5_subset,VG_full_strict,RV_recover,RV_full",
    )
    parser.add_argument("--backbone", default="yolo")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--proxy", action="store_true")
    parser.add_argument(
        "--deployment-strict",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-deployment-strict",
        action="store_true",
        help="Allow oracle dropped hints (ablation only)",
    )
    parser.add_argument("--report", default="reports/REPORT_RV_dev.json")
    parser.add_argument(
        "--recover-overrides",
        default="",
        help='JSON dict passed to run_robustvocab_episode recover_overrides',
    )
    args = parser.parse_args()

    if args.proxy:
        subprocess.run(
            [sys.executable, "scripts/run_proxy_eval.py", "--report", args.report],
            cwd=ROOT,
            check=True,
        )
        return

    cfg = load_config()
    deployment_strict = args.deployment_strict and not args.no_deployment_strict
    recover_overrides = json.loads(args.recover_overrides) if args.recover_overrides else None

    from ovdeploy.b0_cache import ensure_b0_preds
    from ovdeploy.episode import load_episodes_dir
    from ovdeploy.paths_util import load_lvis_minival
    from ovdeploy.infer import run_episode_infer
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
        if args.gpu:
            raise SystemExit(
                "CUDA unavailable but --gpu was set; refusing proxy fallback. "
                "Run on WSL with conda env yoloworld5070."
            )
        raise SystemExit("GPU required — use --gpu on WSL or --proxy for CPU estimates only")

    device = "cuda:0" if use_gpu else "cpu"
    ep_root = cfg["_ovdeploy"] / "data/episodes" / args.split
    ep_dirs = [ep_root / args.config_key] if args.config_key else sorted(ep_root.glob("*"))
    ep_dirs = [d for d in ep_dirs if d.is_dir()]
    if not ep_dirs:
        raise SystemExit(f"No episode dirs under {ep_root}")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    lvis = load_lvis_minival()
    backend = get_backend(device=device)

    rows = []
    for ep_dir in ep_dirs:
        eps = load_episodes_dir(ep_dir)[: args.max_episodes]
        iids = list({i for ep in eps for i in ep.image_ids})
        b0_cache = ensure_b0_preds(iids, lvis, device=device, backbone=args.backbone)

        for method in methods:
            acc = []
            for ep in eps:
                if method in ("B4_clip", "B1_oracle", "VG_router", "VG_full", "M2_calib"):
                    from vocabguard.infer_pipeline import run_vocabguard_episode

                    r = run_vocabguard_episode(
                        ep,
                        method.replace("_strict", ""),
                        lvis,
                        backend._inner,
                        max_images=args.max_images,
                        device=device,
                        b0_preds_by_image=b0_cache,
                        backbone=args.backbone,
                    )
                elif method == "B5_subset":
                    r = run_episode_infer(
                        ep,
                        "B5_subset",
                        lvis,
                        max_images=args.max_images,
                        b0_preds_by_image=b0_cache,
                        episode_vocab_for_oov=list(ep.vocab.cat_ids),
                        backbone=args.backbone,
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
                        backbone=args.backbone,
                        deployment_strict=deployment_strict,
                        recover_overrides=recover_overrides,
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
                    "deployment_strict": deployment_strict,
                }
            )

    report = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_hash(),
        "split": args.split,
        "backbone": args.backbone,
        "gpu_used": use_gpu,
        "deployment_strict": deployment_strict,
        "rows": rows,
    }
    out = ROOT / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
