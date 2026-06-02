"""GPU episodic eval across noise modes -> REPORT_4c_noise.json."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=5)
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--vocab-sizes", nargs="*", type=int, default=[10, 30, 100])
    args = parser.parse_args()

    use_gpu = args.gpu
    if use_gpu:
        try:
            import torch

            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False

    rows = []
    noises = ["none", "synonym", "missing_class"]
    baselines = ["B0_full", "B5_subset"]

    if use_gpu:
        from ovdeploy.b0_cache import ensure_b0_preds
        from ovdeploy.episode import load_episodes_dir
        from ovdeploy.infer import run_episode_infer
        from ovdeploy.paths_util import load_lvis_minival

        lvis = load_lvis_minival()
        for vs in args.vocab_sizes:
            for noise in noises:
                dirs = [
                    d
                    for d in (ROOT / "data/episodes/dev").glob("*")
                    if d.name == f"dev_v{vs}_s42_{noise}"
                ]
                if not dirs:
                    continue
                eps = load_episodes_dir(dirs[0])[: args.max_episodes]
                iids = list({i for ep in eps for i in ep.image_ids})
                b0_cache = ensure_b0_preds(iids, lvis)
                for bl in baselines:
                    acc = []
                    for ep in eps:
                        r = run_episode_infer(
                            ep,
                            bl,
                            lvis,
                            max_images=args.max_images,
                            b0_preds_by_image=b0_cache,
                            episode_vocab_for_oov=list(ep.vocab.cat_ids),
                        )
                        acc.append(r)
                    if acc:
                        rows.append(
                            {
                                "vocab_size": vs,
                                "noise": noise,
                                "baseline": bl,
                                "EpisodicAP_mean": sum(x["EpisodicAP_mean"] for x in acc) / len(acc),
                                "OOV_FP_mean": sum(x["OOV_FP_mean"] for x in acc) / len(acc),
                                "mode": "gpu",
                                "n_episodes": len(acc),
                            }
                        )
    else:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "rbm", ROOT / "scripts" / "run_baseline_matrix.py"
        )
        rbm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rbm)
        for vs in args.vocab_sizes:
            for noise in noises:
                for bl in baselines:
                    m = rbm.proxy_metrics(bl, vs)
                    rows.append({"vocab_size": vs, "noise": noise, "baseline": bl, **m})

    report = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_hash(),
        "gpu_used": use_gpu,
        "metrics_version": "v2",
        "rows": rows,
    }
    out = ROOT / "reports/REPORT_4c_noise.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out} ({len(rows)} rows, gpu={use_gpu})")


if __name__ == "__main__":
    main()
