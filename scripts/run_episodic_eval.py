"""Full / dev episodic evaluation -> REPORT_4 (B0-B5, no M1)."""
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
    parser.add_argument("--split", choices=["dev", "eval", "full"], default="dev")
    parser.add_argument("--report", default="reports/REPORT_4_main.json")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--proxy", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--backbone", default="yolo")
    args = parser.parse_args()

    cfg_ep = __import__("ovdeploy.paths_util", fromlist=["load_episodes_cfg"]).load_episodes_cfg()
    baselines = [b for b in cfg_ep["baselines"] if b != "M1_adapter"]

    use_gpu = args.gpu and not args.proxy
    if use_gpu:
        try:
            import torch

            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False

    rows = []
    if use_gpu:
        from ovdeploy.b0_cache import ensure_b0_preds
        from ovdeploy.episode import load_episodes_dir
        from ovdeploy.infer import run_episode_infer
        from ovdeploy.paths_util import load_lvis_minival

        lvis = load_lvis_minival()
        for vs in cfg_ep["vocab_sizes"]:
            dirs = [d for d in (ROOT / "data/episodes/dev").glob("*") if f"dev_v{vs}_s42_none" in d.name]
            if not dirs:
                continue
            eps = load_episodes_dir(dirs[0])[: args.max_episodes]
            iids = list({i for ep in eps for i in ep.image_ids})
            b0_cache = ensure_b0_preds(iids, lvis, backbone=args.backbone)
            for bl in baselines:
                acc = []
                for ep in eps:
                    try:
                        r = run_episode_infer(
                            ep,
                            bl,
                            lvis,
                            max_images=args.max_images,
                            b0_preds_by_image=b0_cache,
                            episode_vocab_for_oov=list(ep.vocab.cat_ids),
                            backbone=args.backbone,
                        )
                        acc.append(r)
                    except Exception as e:
                        rows.append({"vocab_size": vs, "baseline": bl, "error": str(e), "mode": "gpu"})
                        break
                if acc:
                    rows.append(
                        {
                            "vocab_size": vs,
                            "baseline": bl,
                            "EpisodicAP_mean": sum(x["EpisodicAP_mean"] for x in acc) / len(acc),
                            "OOV_FP_mean": sum(x["OOV_FP_mean"] for x in acc) / len(acc),
                            "text_ms_per_image": sum(x["text_ms_per_image"] for x in acc) / len(acc),
                            "mode": "gpu",
                        }
                    )
        status = "gpu_main_table"
    else:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "rbm", ROOT / "scripts" / "run_baseline_matrix.py"
        )
        rbm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rbm)
        for vs in cfg_ep["vocab_sizes"]:
            for bl in baselines:
                m = rbm.proxy_metrics(bl, vs)
                rows.append({"vocab_size": vs, "baseline": bl, **m})
        status = "proxy_main_table"

    report = {
        "status": status,
        "split": args.split,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_hash(),
        "metrics_version": "v2",
        "federated_AP_reference": 22.7,
        "rows": rows,
    }
    out = ROOT / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out} ({status}, {len(rows)} rows)")


if __name__ == "__main__":
    main()
