"""Run nuScenes-OVDeploy pilot evaluation -> REPORT_nuscenes_main.json."""
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


def resolve_report_path(cfg: dict, override: str | None) -> Path:
    if override:
        p = Path(override)
        return p if p.is_absolute() else ROOT / p
    key = "report_out_wsl" if sys.platform != "win32" else "report_out"
    p = Path(cfg[key])
    if not p.is_absolute():
        p = ROOT / p
    return p


def resolve_episodes_dir(cfg: dict, override: Path | None, vocab_size: int, seed: int) -> Path:
    if override is not None:
        return override
    key = "episodes_out_wsl" if sys.platform != "win32" else "episodes_out"
    base = Path(cfg[key])
    if not base.is_absolute():
        base = ROOT / base
    noise = cfg.get("noise", "none")
    return base / f"dev_v{vocab_size}_s{seed}_{noise}"


def main() -> None:
    parser = argparse.ArgumentParser(description="nuScenes episodic eval (B0/B5)")
    parser.add_argument("--config", default="config/nuscenes_pilot.yaml")
    parser.add_argument("--episodes-dir", type=Path, default=None)
    parser.add_argument("--baseline", choices=["B0_full", "B5_subset"], default=None)
    parser.add_argument("--vocab-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", default=None)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--merge-only", action="store_true", help="Merge partial report rows")
    args = parser.parse_args()

    from ovdeploy.nuscenes.taxonomy import (
        load_pilot_config,
        load_taxonomy,
        resolve_class_map_path,
        resolve_nuscenes_root,
    )

    cfg = load_pilot_config(ROOT / args.config)
    report_path = resolve_report_path(cfg, args.report)
    episodes_dir = resolve_episodes_dir(cfg, args.episodes_dir, args.vocab_size, args.seed)

    baselines = [args.baseline] if args.baseline else cfg.get("baselines", ["B0_full", "B5_subset"])

    use_gpu = args.gpu
    if use_gpu:
        try:
            import torch

            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False

    rows: list[dict] = []
    if report_path.is_file() and args.merge_only:
        rows = json.loads(report_path.read_text(encoding="utf-8")).get("rows", [])

    status = "gpu_nuscenes_pilot" if use_gpu else "cpu_nuscenes_pilot_skipped"

    if use_gpu:
        from ovdeploy.episode import load_episodes_dir
        from ovdeploy.nuscenes.gt import NuScenesGT
        from ovdeploy.nuscenes.infer import ensure_nuscenes_b0_preds, run_nuscenes_episode_infer

        nusc_root = resolve_nuscenes_root(cfg)
        if not nusc_root.is_dir():
            raise SystemExit(f"nuScenes root not found: {nusc_root}")

        taxonomy = load_taxonomy(resolve_class_map_path(cfg))
        oov_thr = float(cfg.get("score_thresh_oov", 0.05))
        gt_index = NuScenesGT(
            nusc_root,
            version=cfg.get("version", "v1.0-mini"),
            camera=cfg["camera"],
            taxonomy=taxonomy,
        )

        if not episodes_dir.is_dir():
            raise SystemExit(f"Episodes dir not found: {episodes_dir}\nRun build_nuscenes_episodes.py first.")

        eps = load_episodes_dir(episodes_dir)
        if args.max_episodes:
            eps = eps[: args.max_episodes]

        all_iids = list({i for ep in eps for i in ep.image_ids})
        b0_cache = ensure_nuscenes_b0_preds(all_iids, gt_index, taxonomy)

        for bl in baselines:
            acc = []
            for ep in eps:
                try:
                    r = run_nuscenes_episode_infer(
                        ep,
                        bl,
                        gt_index,
                        taxonomy,
                        max_images=args.max_images,
                        b0_preds_by_image=b0_cache,
                        episode_vocab_for_oov=list(ep.vocab.cat_ids),
                        oov_score_thr=oov_thr,
                    )
                    acc.append(r)
                except Exception as e:
                    rows.append(
                        {
                            "vocab_size": args.vocab_size,
                            "baseline": bl,
                            "error": str(e),
                            "mode": "gpu",
                        }
                    )
                    break
            if acc:
                rows = [x for x in rows if not (x.get("vocab_size") == args.vocab_size and x.get("baseline") == bl)]
                rows.append(
                    {
                        "vocab_size": args.vocab_size,
                        "baseline": bl,
                        "EpisodicAP_mean": sum(x["EpisodicAP_mean"] for x in acc) / len(acc),
                        "OOV_FP_mean": sum(x["OOV_FP_mean"] for x in acc) / len(acc),
                        "text_ms_per_image": sum(x["text_ms_per_image"] for x in acc) / len(acc),
                        "n_episodes": len(acc),
                        "mode": "gpu",
                    }
                )

    report = {
        "status": status,
        "dataset": "nuscenes-mini",
        "camera": cfg.get("camera", "CAM_FRONT"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_hash(),
        "metrics_version": "v2",
        "episodes_dir": str(episodes_dir),
        "rows": rows,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {report_path} ({len(rows)} rows, status={status})")


if __name__ == "__main__":
    main()
