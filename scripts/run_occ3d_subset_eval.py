#!/usr/bin/env python3
"""Eval Occ3D semantic subset episodes -> REPORT_occ3d_subset.json."""
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
PILOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PILOT / "data" / "episodes_occ3d_subset"
DEFAULT_REPORT = PILOT / "reports" / "REPORT_occ3d_subset.json"
DEFAULT_CLASS_MAP = _CFG_DIR / "nuscenes_class_map.yaml"


def git_hash() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_REPO,
            timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def eval_subset_b0_cache(
    eps: list,
    gt_index,
    taxonomy,
    oov_thr: float,
) -> tuple[dict, list[dict]]:
    from ovdeploy.metrics import episodic_ap_per_image_v2, oov_fp_rate
    from ovdeploy.nuscenes.infer import load_nuscenes_b0_preds

    episode_rows: list[dict] = []
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
            oov_list.append(oov_fp_rate(b0p, vocab, score_thr=oov_thr))
        if not ap_list:
            continue
        episode_rows.append(
            {
                "episode_id": ep.episode_id,
                "scene": ep.meta.get("scene", ""),
                "occ3d_subset": ep.meta.get("occ3d_subset", ""),
                "vocab_size": ep.vocab_size,
                "B0_EpisodicAP": sum(ap_list) / len(ap_list),
                "B0_OOV_FP": sum(oov_list) / len(oov_list),
            }
        )

    if not episode_rows:
        return {}, episode_rows

    agg = {
        "baseline": "B0_full",
        "EpisodicAP_mean": sum(r["B0_EpisodicAP"] for r in episode_rows) / len(episode_rows),
        "OOV_FP_mean": sum(r["B0_OOV_FP"] for r in episode_rows) / len(episode_rows),
        "n_episodes": len(episode_rows),
        "mode": "b0_cache",
    }
    return agg, episode_rows


def eval_subset_gpu(
    eps: list,
    gt_index,
    taxonomy,
    oov_thr: float,
) -> tuple[list[dict], list[dict]]:
    from ovdeploy.nuscenes.infer import ensure_nuscenes_b0_preds, run_nuscenes_episode_infer

    all_iids = list({i for ep in eps for i in ep.image_ids})
    b0_cache = ensure_nuscenes_b0_preds(all_iids, gt_index, taxonomy)

    rows: list[dict] = []
    episode_rows: list[dict] = []

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
                oov_score_thr=oov_thr,
            )
            acc_ap.append(r["EpisodicAP_mean"])
            if bl == "B0_full":
                acc_oov.append(r["OOV_FP_mean"])
            episode_rows.append(
                {
                    "episode_id": ep.episode_id,
                    "baseline": bl,
                    "scene": ep.meta.get("scene", ""),
                    "occ3d_subset": ep.meta.get("occ3d_subset", ""),
                    "vocab_size": ep.vocab_size,
                    "EpisodicAP_mean": r["EpisodicAP_mean"],
                    "OOV_FP_mean": r.get("OOV_FP_mean"),
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
    return rows, episode_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--oov-thr", type=float, default=0.05)
    args = parser.parse_args()

    from ovdeploy.episode import load_episodes_dir
    from ovdeploy.nuscenes.gt import NuScenesGT
    from ovdeploy.nuscenes.taxonomy import load_taxonomy, resolve_nuscenes_root

    if not args.data_dir.is_dir():
        raise SystemExit(f"Data dir missing: {args.data_dir}\nRun build_occ3d_subset_episodes.py first.")

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
    subset_dirs = sorted([p for p in args.data_dir.iterdir() if p.is_dir()])
    if not subset_dirs:
        raise SystemExit(f"No subset dirs under {args.data_dir}")

    camera = "CAM_FRONT"
    gt_index = NuScenesGT(
        nusc_root, version="v1.0-mini", camera=camera, taxonomy=taxonomy
    )

    all_rows: list[dict] = []
    all_episodes: list[dict] = []

    for subdir in subset_dirs:
        subset_id = subdir.name
        eps = load_episodes_dir(subdir)
        if not eps:
            print(f"Skip {subset_id}: no episodes")
            continue

        if use_gpu:
            agg_rows, ep_rows = eval_subset_gpu(eps, gt_index, taxonomy, args.oov_thr)
            for ar in agg_rows:
                ar["subset_id"] = subset_id
                ar["vocab_size"] = eps[0].vocab_size
            all_rows.extend(agg_rows)
            all_episodes.extend(ep_rows)
        else:
            agg, ep_rows = eval_subset_b0_cache(eps, gt_index, taxonomy, args.oov_thr)
            if agg:
                agg["subset_id"] = subset_id
                agg["vocab_size"] = eps[0].vocab_size
                all_rows.append(agg)
            for er in ep_rows:
                er["subset_id"] = subset_id
            all_episodes.extend(ep_rows)

        print(f"Evaluated {subset_id}: n={len(eps)} mode={'gpu' if use_gpu else 'b0_cache'}")

    report = {
        "status": "occ3d_semantic_subset",
        "dataset": "nuscenes-mini",
        "prototype": "occ3d_detection_gt_proxy",
        "camera": camera,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_hash(),
        "metrics_version": "v2",
        "data_dir": str(args.data_dir),
        "rows": all_rows,
        "episodes": all_episodes,
        "note": "Occ3D semantic subsets on nuScenes detection GT proxy — not voxel Occ3D mAP.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {args.report} ({len(all_rows)} aggregate rows)")


if __name__ == "__main__":
    main()
