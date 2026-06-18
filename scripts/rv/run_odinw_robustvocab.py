"""ODinW-13 cross-domain eval for RobustVocab (GPU)."""
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
    parser.add_argument("--max-images", type=int, default=30)
    parser.add_argument("--proxy", action="store_true")
    args = parser.parse_args()

    if args.proxy:
        subprocess.run([sys.executable, "scripts/rv/run_proxy_odinw.py"], cwd=ROOT, check=True)
        return

    if not args.gpu:
        raise SystemExit("GPU required — use --gpu on WSL")

    cfg = load_config()
    ov = cfg["_ovdeploy"]
    odinw_root = ov / "data" / "odinw"
    if not odinw_root.is_dir():
        raise SystemExit(
            f"Missing {odinw_root} — run Paper 2 setup: bash scripts/wsl_odinw_full13.sh"
        )

    try:
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA not available")
    except ImportError as e:
        raise SystemExit(f"torch not available: {e}") from e

    from ovdeploy.odinw_infer import ensure_odinw_b0_cache, run_odinw_episode
    from ovdeploy.odinw_loader import list_images, load_odinw_coco, load_domain_meta
    from robustvocab.odinw_runner import run_odinw_rv_full

    rows = []
    device = "cuda:0"

    for slug_dir in sorted(odinw_root.iterdir()):
        if not slug_dir.is_dir():
            continue
        slug = slug_dir.name
        try:
            meta = load_domain_meta(slug)
            coco = load_odinw_coco(slug)
        except FileNotFoundError:
            continue
        images = list_images(coco, args.max_images)
        if len(images) < 3:
            continue
        image_ids = [im["id"] for im in images]
        b0_cache = ensure_odinw_b0_cache(slug, image_ids, coco, device=device, backbone="yolo")

        for vs in (10, 30):
            for bl in ("B5_subset",):
                r = run_odinw_episode(
                    slug,
                    bl,
                    vs,
                    max_images=args.max_images,
                    device=device,
                    b0_preds_by_image=b0_cache,
                    backbone="yolo",
                )
                rows.append(
                    {
                        "domain": meta.get("domain", slug),
                        "vocab_size": vs,
                        "method": bl,
                        "EpisodicAP_mean": r.get("EpisodicAP_mean", 0),
                        "OOV_FP_mean": r.get("OOV_FP_mean", 0),
                        "gpu_used": True,
                    }
                )
            r_rv = run_odinw_rv_full(
                slug,
                vs,
                max_images=args.max_images,
                device=device,
                b0_preds_by_image=b0_cache,
                backbone="yolo",
            )
            rows.append({**r_rv, "gpu_used": True})

    if not rows:
        raise SystemExit("No ODinW domains evaluated — check data/odinw setup")

    report = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_hash(),
        "gpu_used": True,
        "rows": rows,
    }
    out = reports_dir() / "REPORT_RV_odinw.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
