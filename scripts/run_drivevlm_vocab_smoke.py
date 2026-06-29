#!/usr/bin/env python3
"""Smoke eval: DriveVLM vocabulary episodes -> REPORT_drivevlm_vocab_smoke.json."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_PAPER = Path(__file__).resolve().parents[3] / "submission-a"
if str(ROOT_PAPER) not in sys.path:
    sys.path.insert(0, str(ROOT_PAPER))

from _pilot_layout import pilot_layout  # noqa: E402

_REPO, _CFG_DIR, _DATA_DIR = pilot_layout(Path(__file__))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

PILOT = Path(__file__).resolve().parents[1]
DEFAULT_EP_DIR = PILOT / "data" / "episodes_drivevlm_vocab" / "dev"
DEFAULT_REPORT = PILOT / "reports" / "REPORT_drivevlm_vocab_smoke.json"
DEFAULT_CLASS_MAP = _CFG_DIR / "nuscenes_class_map.yaml"


def git_hash() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT_PAPER,
            timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-dir", type=Path, default=DEFAULT_EP_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--oov-thr", type=float, default=0.05)
    args = parser.parse_args()

    from ovdeploy.episode import load_episodes_dir
    from ovdeploy.nuscenes.gt import NuScenesGT
    from ovdeploy.nuscenes.infer import ensure_nuscenes_b0_preds, run_nuscenes_episode_infer
    from ovdeploy.nuscenes.taxonomy import load_taxonomy, resolve_nuscenes_root

    if not args.episodes_dir.is_dir():
        raise SystemExit(f"Episodes dir missing: {args.episodes_dir}\nRun build_drivevlm_episodes.py first.")

    try:
        from ovdeploy.nuscenes.taxonomy import load_pilot_config

        pilot_cfg = load_pilot_config(_REPO / "config" / "nuscenes_pilot.yaml")
        nusc_root = args.root or resolve_nuscenes_root(pilot_cfg)
    except Exception:
        nusc_root = args.root or Path("d:/data/nuscenes")

    use_gpu = args.gpu
    if use_gpu:
        try:
            import torch

            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False

    taxonomy = load_taxonomy(args.class_map)
    eps = load_episodes_dir(args.episodes_dir)
    camera = eps[0].meta.get("camera", "CAM_FRONT") if eps else "CAM_FRONT"

    rows: list[dict] = []
    episode_rows: list[dict] = []

    if not use_gpu:
        print("CUDA not available; computing OOV from B0 cache only (B5 skipped).")
        from ovdeploy.metrics import episodic_ap_per_image_v2, oov_fp_rate
        from ovdeploy.nuscenes.infer import load_nuscenes_b0_preds

        gt_index = NuScenesGT(
            nusc_root, version="v1.0-mini", camera=camera, taxonomy=taxonomy
        )
        for ep in eps:
            ap_list: list[float] = []
            oov_list: list[float] = []
            vocab = list(ep.vocab.cat_ids)
            for iid in ep.image_ids:
                b0p = load_nuscenes_b0_preds(iid)
                if b0p is None:
                    continue
                gt = gt_index.get_gt(iid)
                ap_list.append(
                    episodic_ap_per_image_v2(b0p, gt["boxes"], gt["cat_ids"], vocab)
                )
                oov_list.append(oov_fp_rate(b0p, vocab, score_thr=args.oov_thr))
            if not ap_list:
                continue
            episode_rows.append(
                {
                    "episode_id": ep.episode_id,
                    "drivevlm_prompt": ep.meta.get("drivevlm_prompt", ""),
                    "vocab_size": ep.vocab_size,
                    "B0_EpisodicAP": sum(ap_list) / len(ap_list),
                    "B0_OOV_FP": sum(oov_list) / len(oov_list),
                    "scene": ep.meta.get("scene", ""),
                }
            )
        if episode_rows:
            rows.append(
                {
                    "baseline": "B0_full",
                    "EpisodicAP_mean": sum(r["B0_EpisodicAP"] for r in episode_rows) / len(episode_rows),
                    "OOV_FP_mean": sum(r["B0_OOV_FP"] for r in episode_rows) / len(episode_rows),
                    "n_episodes": len(episode_rows),
                    "mode": "b0_cache",
                }
            )
    else:
        gt_index = NuScenesGT(
            nusc_root, version="v1.0-mini", camera=camera, taxonomy=taxonomy
        )
        all_iids = list({i for ep in eps for i in ep.image_ids})
        b0_cache = ensure_nuscenes_b0_preds(all_iids, gt_index, taxonomy)

        for bl in ("B0_full", "B5_subset"):
            acc_ap: list[float] = []
            acc_oov: list[float] = []
            for ep in eps:
                r = run_nuscenes_episode_infer(
                    ep,
                    bl,
                    gt_index,
                    taxonomy,
                    b0_preds_by_image=b0_cache,
                    episode_vocab_for_oov=list(ep.vocab.cat_ids),
                    oov_score_thr=args.oov_thr,
                )
                acc_ap.append(r["EpisodicAP_mean"])
                if bl == "B0_full":
                    acc_oov.append(r["OOV_FP_mean"])
                episode_rows.append(
                    {
                        "episode_id": ep.episode_id,
                        "baseline": bl,
                        "drivevlm_prompt": ep.meta.get("drivevlm_prompt", ""),
                        "vocab_size": ep.vocab_size,
                        "EpisodicAP_mean": r["EpisodicAP_mean"],
                        "OOV_FP_mean": r.get("OOV_FP_mean"),
                        "scene": ep.meta.get("scene", ""),
                    }
                )
            rows.append(
                {
                    "baseline": bl,
                    "EpisodicAP_mean": sum(acc_ap) / len(acc_ap),
                    "OOV_FP_mean": sum(acc_oov) / len(acc_oov) if acc_oov else None,
                    "n_episodes": len(eps),
                    "mode": "gpu",
                }
            )

    report = {
        "status": "drivevlm_vocab_smoke",
        "dataset": "nuscenes-mini",
        "prototype": "drivevlm_scene_vocabulary",
        "camera": camera,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_hash(),
        "metrics_version": "v2",
        "episodes_dir": str(args.episodes_dir),
        "rows": rows,
        "episodes": episode_rows,
        "note": "Prototype smoke eval; mentor-defined vocab extension for trial Option A.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {args.report} ({len(rows)} aggregate rows)")


if __name__ == "__main__":
    main()
