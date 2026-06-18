"""Measure inference latency: B5 vs VG_router."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vocabguard.paths_util import load_config, reports_dir


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
    parser.add_argument("--n-images", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config()
    sys.path.insert(0, str(cfg["_ovdeploy"]))

    from ovdeploy.b0_cache import ensure_b0_preds
    from ovdeploy.episode import load_episodes_dir
    from ovdeploy.infer import run_episode_infer
    from ovdeploy.paths_util import load_lvis_minival
    from vocabguard.backend import get_extended_backend
    from vocabguard.infer_pipeline import run_vocabguard_episode

    use_gpu = args.gpu
    if use_gpu:
        try:
            import torch

            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False

    device = "cuda:0" if use_gpu else "cpu"
    lvis = load_lvis_minival()
    ep_dir = cfg["_ovdeploy"] / "data/episodes/dev/dev_v30_s42_none"
    eps = load_episodes_dir(ep_dir)[:1]
    if not eps:
        raise SystemExit("No episodes found")
    ep = eps[0]
    iids = ep.image_ids[: args.n_images]
    ep.image_ids = iids

    b0_cache = ensure_b0_preds(iids, lvis, device=device) if use_gpu else None
    backend = get_extended_backend("yolo", device=device)

    rows = []
    for method in ("B5_subset", "VG_router"):
        if use_gpu and device.startswith("cuda"):
            import torch

            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if method == "B5_subset":
            run_episode_infer(
                ep, "B5_subset", lvis,
                b0_preds_by_image=b0_cache,
                episode_vocab_for_oov=list(ep.vocab.cat_ids),
            )
        else:
            run_vocabguard_episode(
                ep, "VG_router", lvis, backend,
                device=device, b0_preds_by_image=b0_cache,
            )
        if use_gpu and device.startswith("cuda"):
            import torch

            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        rows.append(
            {
                "method": method,
                "n_images": len(iids),
                "total_sec": round(elapsed, 3),
                "ms_per_image": round(1000 * elapsed / max(len(iids), 1), 1),
                "gpu_used": use_gpu,
            }
        )

    report = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_hash(),
        "gpu_used": use_gpu,
        "rows": rows,
    }
    out = reports_dir() / "REPORT_VG_latency.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
