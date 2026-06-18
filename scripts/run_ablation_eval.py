"""Ablation study for VocabGuard components."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vocabguard.oov_guard import OOVGuard
from vocabguard.paths_util import load_config, reports_dir
from vocabguard.router import VocabRouter


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
    parser.add_argument("--config-key", default="dev_v30_s42_none")
    parser.add_argument("--max-episodes", type=int, default=3)
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
    ep_dir = cfg["_ovdeploy"] / "data/episodes/dev" / args.config_key
    eps = load_episodes_dir(ep_dir)[: args.max_episodes]
    iids = list({i for ep in eps for i in ep.image_ids})
    b0_cache = ensure_b0_preds(iids, lvis, device=device) if use_gpu else None
    backend = get_extended_backend("yolo", device=device)

    variants = [
        ("B5_baseline", "B5_subset", None, None, None),
        ("VG_full", "VG_full", VocabRouter(delta=3), OOVGuard(), None),
        ("w/o_Guard", "VG_router", VocabRouter(delta=3), None, None),
        ("w/o_Router", "VG_router", VocabRouter(delta=0), OOVGuard(), None),
        ("CLIP_router", "VG_router", VocabRouter(delta=3, use_clip_fallback=True), OOVGuard(), None),
        ("hard_mask", "VG_full_hard", VocabRouter(delta=3), OOVGuard(), "hard"),
        ("delta_0", "VG_router", VocabRouter(delta=0), OOVGuard(), None),
        ("delta_10", "VG_router", VocabRouter(delta=10), OOVGuard(), None),
    ]

    rows = []
    calib = ROOT / "data/checkpoints/calib_head_s42.pt"

    for name, method, router, guard, hard in variants:
        acc = []
        for ep in eps:
            if method == "B5_subset":
                r = run_episode_infer(
                    ep, "B5_subset", lvis,
                    b0_preds_by_image=b0_cache,
                    episode_vocab_for_oov=list(ep.vocab.cat_ids),
                )
            elif method == "VG_full_hard":
                g = OOVGuard()
                r = run_vocabguard_episode(
                    ep, "VG_full", lvis, backend,
                    router=router, guard=g,
                    device=device, b0_preds_by_image=b0_cache,
                )
                # apply hard mask on OOV for audit metric
                if b0_cache:
                    oovs = []
                    for iid in ep.image_ids:
                        b0p = b0_cache.get(iid, [])
                        guarded = g.guard(b0p, ep.vocab.cat_ids, hard_mask=True)
                        from ovdeploy.metrics import oov_fp_rate
                        oovs.append(oov_fp_rate(guarded, ep.vocab.cat_ids))
                    r["OOV_FP_mean"] = float(sum(oovs) / max(len(oovs), 1))
            elif method.startswith("VG"):
                ckpt = calib if method == "M2_calib" and calib.is_file() else None
                m = "M2_calib" if method == "M2_calib" else method
                r = run_vocabguard_episode(
                    ep, m, lvis, backend,
                    router=router, guard=guard,
                    calib_ckpt=ckpt,
                    device=device, b0_preds_by_image=b0_cache,
                )
            else:
                continue
            acc.append(r)
        if acc:
            rows.append(
                {
                    "variant": name,
                    "method": method,
                    "EpisodicAP_mean": sum(x["EpisodicAP_mean"] for x in acc) / len(acc),
                    "OOV_FP_mean": sum(x["OOV_FP_mean"] for x in acc) / len(acc),
                    "n_episodes": len(acc),
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
    out = reports_dir() / "REPORT_VG_ablation.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Variant & EpisodicAP & OOV-FP \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['variant'].replace('_', ' ')} & {r['EpisodicAP_mean']:.1f} & {100*r['OOV_FP_mean']:.0f}\\% \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    tex = ROOT / "paper/tables/ablation.tex"
    tex.parent.mkdir(parents=True, exist_ok=True)
    tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
