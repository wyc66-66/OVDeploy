"""ODinW-13 cross-domain eval for VocabGuard -> REPORT_VG_odinw.json."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vocabguard.infer_pipeline import _b0_hint_classes
from vocabguard.oov_guard import OOVGuard
from vocabguard.paths_util import load_config, reports_dir
from vocabguard.router import VocabRouter


DEFAULT_DOMAINS = (
    "aquarium,aerial,cottontail,egohands,mushrooms,packages,pascalvoc,"
    "pistols,fryingpan,thermal,pothole,shellfish,vehicles"
)


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


def _domain_prompts(coco: dict, cat_ids: list[int], meta: dict) -> dict[int, list[str]]:
    from ovdeploy.odinw_loader import class_texts_for_ids

    _, texts = class_texts_for_ids(coco, cat_ids, meta)
    out: dict[int, list[str]] = {}
    for cid, tx in zip(cat_ids, texts):
        out[cid] = tx if isinstance(tx, list) else [str(tx)]
    return out


def _clip_scores_domain(
    image_rgb: np.ndarray,
    cat_ids: list[int],
    prompts: dict[int, list[str]],
    freq: list[int],
) -> dict[int, float]:
    from ovdeploy.clip_vocab import clip_topk_cat_ids

    ranked = clip_topk_cat_ids(image_rgb, cat_ids, prompts, len(cat_ids), freq, seed=0)
    scores = {c: 0.0 for c in cat_ids}
    for rank, cid in enumerate(reversed(ranked)):
        scores[cid] = (rank + 1) / max(len(ranked), 1)
    return scores


def run_odinw_method(
    slug: str,
    method: str,
    vocab_size: int,
    max_images: int,
    device: str,
    b0_cache: dict[int, list[dict]],
    router: VocabRouter,
    guard: OOVGuard,
) -> dict:
    from ovdeploy.odinw_infer import _predict_domain, run_odinw_episode
    from ovdeploy.odinw_loader import (
        all_category_ids,
        gt_by_image,
        image_path,
        list_images,
        load_domain_meta,
        load_odinw_coco,
        sample_episode_vocab,
    )
    from ovdeploy.odinw_metrics import episodic_ap_per_image, fp_non_gt_rate, oov_fp_rate

    if method == "B5_subset":
        r = run_odinw_episode(
            slug,
            "B5_subset",
            vocab_size,
            max_images=max_images,
            device=device,
            b0_preds_by_image=b0_cache,
            backbone="yolo",
        )
        return {
            "EpisodicAP_mean": r["EpisodicAP_mean"],
            "OOV_FP_mean": r["OOV_FP_mean"],
            "FP_nonGT_mean": 0.0,
            "n_images": r["n_images"],
        }

    coco = load_odinw_coco(slug)
    meta = load_domain_meta(slug)
    images = list_images(coco, max_images)
    episode_vocab = sample_episode_vocab(coco, meta.get("classes", []), vocab_size, 42)
    gt_map = gt_by_image(coco)
    all_ids = all_category_ids(coco)
    prompts = _domain_prompts(coco, all_ids, meta)

    ap_list: list[float] = []
    oov_list: list[float] = []
    fp_list: list[float] = []

    for im in images:
        iid = im["id"]
        path = image_path(slug, im["file_name"])
        img = cv2.imread(str(path))
        if img is None:
            continue
        rgb = img[:, :, [2, 1, 0]]
        b0_hints = _b0_hint_classes(b0_cache.get(iid, []), episode_vocab)

        core = list(episode_vocab)
        cand = [c for c in all_ids if c not in set(core)]
        pool = cand[: router.cand_pool]
        probe = core + pool
        scores = _clip_scores_domain(rgb, probe, prompts, all_ids)
        ranked = sorted(pool, key=lambda c: -scores.get(c, 0.0))
        budget = min(len(core) + router.delta, len(all_ids))
        routed = core + ranked[: max(0, budget - len(core))]

        from ovdeploy.backends.base import get_backend

        backend = get_backend("yolo", device=device)
        preds = _predict_domain(backend, slug, rgb, routed, coco, meta)

        gt = gt_map.get(iid, {"boxes": [], "cat_ids": []})
        ap_list.append(
            episodic_ap_per_image(preds, gt["boxes"], gt["cat_ids"], episode_vocab)
        )
        fp_list.append(fp_non_gt_rate(preds, episode_vocab))
        if method == "VG_full":
            guarded = guard.guard(b0_cache.get(iid, []), episode_vocab, in_vocab_preds=preds)
            oov_list.append(oov_fp_rate(guarded, episode_vocab))
        else:
            oov_list.append(oov_fp_rate(b0_cache.get(iid, []), episode_vocab))

    return {
        "EpisodicAP_mean": float(np.mean(ap_list)) if ap_list else 0.0,
        "OOV_FP_mean": float(np.mean(oov_list)) if oov_list else 0.0,
        "FP_nonGT_mean": float(np.mean(fp_list)) if fp_list else 0.0,
        "n_images": len(ap_list),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--vocab-sizes", default="10,30")
    parser.add_argument("--domains", default=DEFAULT_DOMAINS)
    parser.add_argument("--force-b0-cache", action="store_true")
    parser.add_argument(
        "--report",
        default="reports/REPORT_VG_odinw.json",
        help="Output report path relative to project root",
    )
    args = parser.parse_args()

    cfg = load_config()
    ov_root = cfg["_ovdeploy"]
    sys.path.insert(0, str(ov_root))
    odinw_base = ov_root / "data" / "odinw"
    if not odinw_base.is_dir():
        raise SystemExit(
            f"ODinW data missing at {odinw_base}. "
            "Run: wsl bash submission-a/scripts/wsl_odinw_full13.sh"
        )

    use_gpu = args.gpu
    if use_gpu:
        try:
            import torch

            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False
    device = "cuda:0" if use_gpu else "cpu"

    vg_cfg = cfg.get("vocabguard", {})
    router = VocabRouter(
        delta=int(vg_cfg.get("router_delta", 3)),
        cand_pool=int(vg_cfg.get("router_cand_pool", 128)),
        use_clip_fallback=True,
    )
    guard = OOVGuard(
        alpha=float(vg_cfg.get("guard_alpha", 2.0)),
        beta=float(vg_cfg.get("guard_beta", 0.3)),
        tau=float(vg_cfg.get("guard_tau", 0.5)),
    )

    from ovdeploy.odinw_infer import ensure_odinw_b0_cache
    from ovdeploy.odinw_loader import list_images, load_domain_meta, load_odinw_coco

    vocab_sizes = [int(x) for x in args.vocab_sizes.split(",") if x.strip()]
    slugs = [s.strip() for s in args.domains.split(",") if s.strip()]
    methods = ("B5_subset", "VG_router", "VG_full")
    rows: list[dict] = []
    domains_run: set[str] = set()

    for slug in slugs:
        meta_path = odinw_base / slug / "domain.json"
        ann_path = odinw_base / slug / "annotations.json"
        if not meta_path.is_file() or not ann_path.is_file():
            print(f"Skip {slug}: missing domain data")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        domain = meta["domain"]
        domains_run.add(domain)
        coco = load_odinw_coco(slug)
        images = list_images(coco, args.max_images)
        image_ids = [im["id"] for im in images]
        b0_cache = {}
        if use_gpu and image_ids:
            b0_cache = ensure_odinw_b0_cache(
                slug,
                image_ids,
                coco,
                device=device,
                backbone="yolo",
                force=args.force_b0_cache,
            )

        for vs in vocab_sizes:
            for method in methods:
                metrics = run_odinw_method(
                    slug,
                    method,
                    vs,
                    args.max_images,
                    device,
                    b0_cache,
                    router,
                    guard,
                )
                rows.append(
                    {
                        "domain": domain,
                        "slug": slug,
                        "vocab_size": vs,
                        "method": method,
                        "EpisodicAP_mean": metrics["EpisodicAP_mean"],
                        "OOV_FP_mean": metrics["OOV_FP_mean"],
                        "FP_nonGT_mean": metrics.get("FP_nonGT_mean", 0.0),
                        "n_images": metrics["n_images"],
                        "mode": "gpu" if use_gpu else "cpu",
                        "source": "roboflow_native",
                        "gpu_used": use_gpu,
                    }
                )

    report = {
        "status": "ok" if rows else "incomplete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_hash(),
        "backbone": "yolo",
        "gpu_used": use_gpu,
        "n_domains": len(domains_run),
        "rows": rows,
    }
    out = ROOT / args.report
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out} ({len(rows)} rows, {len(domains_run)} domains)")


if __name__ == "__main__":
    main()
