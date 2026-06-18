"""GLIP-T native stratified 1k OOV with RobustVocab guarded path."""
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

BACKBONE = "glip_native"
VOCAB_SIZES = (10, 30, 100)
REPORT_NAME = "REPORT_RV_glip_stratified.json"
OOV_CHECKPOINT_EVERY = 100


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


def _parse_vocab_sizes(raw: str | None) -> tuple[int, ...]:
    if not raw:
        return VOCAB_SIZES
    out = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    return out or VOCAB_SIZES


def _row_tau(row: dict, fallback: float) -> float:
    if "glip_guard_tau" in row:
        return float(row["glip_guard_tau"])
    return fallback


def _completed_vocab_sizes(
    rows: list[dict], glip_tau: float, vocab_sizes: tuple[int, ...]
) -> set[int]:
    done: set[int] = set()
    for vs in vocab_sizes:
        methods = {r.get("method") for r in rows if r.get("vocab_size") == vs}
        if not {"B0_full", "B5_subset"}.issubset(methods):
            continue
        rv_rows = [
            r for r in rows if r.get("vocab_size") == vs and r.get("method") == "RV_full"
        ]
        if not rv_rows:
            continue
        if abs(_row_tau(rv_rows[-1], glip_tau) - glip_tau) < 1e-6:
            done.add(vs)
    return done


def _report_path() -> Path:
    return reports_dir() / REPORT_NAME


def _oov_sidecar_path(vocab_size: int) -> Path:
    return reports_dir() / f"REPORT_RV_glip_oov_progress_v{vocab_size}.json"


def _write_checkpoint(
    rows: list[dict],
    n_images: int,
    *,
    glip_guard_tau: float,
    vocab_sizes: tuple[int, ...],
) -> Path:
    out = _report_path()
    done = _completed_vocab_sizes(rows, glip_guard_tau, vocab_sizes)
    report = {
        "status": "partial" if len(done) < len(vocab_sizes) else "ok",
        "split": "stratified_1k",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_hash(),
        "gpu_used": True,
        "backbone": BACKBONE,
        "n_images": n_images,
        "glip_guard_tau": glip_guard_tau,
        "rows": rows,
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Checkpoint {out} ({len(rows)} rows)", flush=True)
    return out


def _load_resume_rows(max_images: int, resume: bool, glip_tau: float) -> list[dict]:
    if not resume:
        return []
    out = _report_path()
    if not out.is_file():
        return []
    data = json.loads(out.read_text(encoding="utf-8"))
    if int(data.get("n_images", 0)) != max_images:
        print(
            f"Resume skip: existing n_images={data.get('n_images')} != {max_images}",
            flush=True,
        )
        return []
    rows = [
        r
        for r in data.get("rows", [])
        if not (
            r.get("method") == "RV_full"
            and abs(_row_tau(r, glip_tau) - glip_tau) >= 1e-6
        )
    ]
    return rows


def _load_oov_progress(vocab_size: int, n_images: int, glip_tau: float) -> dict | None:
    p = _oov_sidecar_path(vocab_size)
    if not p.is_file():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    if int(data.get("n_images", 0)) != n_images:
        return None
    if abs(float(data.get("glip_guard_tau", -1)) - glip_tau) >= 1e-6:
        return None
    return data


def _save_oov_progress(
    vocab_size: int,
    n_images: int,
    glip_tau: float,
    done_ids: list[int],
    oov_sum: float,
    oov_count: int,
) -> None:
    p = _oov_sidecar_path(vocab_size)
    payload = {
        "vocab_size": vocab_size,
        "n_images": n_images,
        "glip_guard_tau": glip_tau,
        "done_ids": done_ids,
        "oov_sum": oov_sum,
        "oov_count": oov_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(p)
    print(
        f"OOV sidecar |V|={vocab_size}: {oov_count}/{n_images} images",
        flush=True,
    )


def _clear_oov_progress(vocab_size: int) -> None:
    p = _oov_sidecar_path(vocab_size)
    if p.is_file():
        p.unlink()


def _get_glip_backend(device: str, lvis):
    from ovdeploy.backends.base import get_backend
    from ovdeploy.paths_util import cat_id_to_index, load_class_texts, load_paths
    from ovdeploy.vocab import subset_class_texts

    backend = get_backend(BACKBONE, device=device)
    paths = load_paths()
    ann_path = paths["_yolo"] / paths["data"]["lvis_minival_ann"]
    lvis_local = json.loads(ann_path.read_text(encoding="utf-8"))
    id_to_im = {im["id"]: im for im in lvis_local["images"]}
    names, texts_raw = load_class_texts()
    cid2idx = cat_id_to_index(lvis)
    return backend, id_to_im, names, texts_raw, cid2idx, subset_class_texts


def _compute_rv_oov_incremental(
    image_ids: list[int],
    vocab: list[int],
    vocab_size: int,
    b0_cache: dict,
    guard,
    lvis,
    device: str,
    glip_tau: float,
    *,
    resume: bool,
) -> float:
    from ovdeploy.metrics import oov_fp_rate

    backend, id_to_im, names, texts_raw, cid2idx, subset_class_texts = _get_glip_backend(
        device, lvis
    )
    texts_sub = subset_class_texts(texts_raw, vocab, cid2idx)

    done_set: set[int] = set()
    oov_sum = 0.0
    oov_count = 0
    if resume:
        prog = _load_oov_progress(vocab_size, len(image_ids), glip_tau)
        if prog:
            done_set = set(int(x) for x in prog.get("done_ids", []))
            oov_sum = float(prog.get("oov_sum", 0.0))
            oov_count = int(prog.get("oov_count", 0))
            print(
                f"Resume OOV |V|={vocab_size}: {oov_count} images already done",
                flush=True,
            )

    since_checkpoint = 0
    for iid in image_ids:
        if iid in done_set:
            continue
        im = id_to_im.get(iid)
        if not im:
            continue
        import cv2

        path = backend.image_path(im["file_name"])
        img = cv2.imread(str(path))
        if img is None:
            continue
        rgb = img[:, :, [2, 1, 0]]
        preds = backend.predict(
            rgb,
            texts_sub,
            vocab,
            iid,
            class_names=names,
            class_texts_raw=texts_sub,
            cid2idx=cid2idx,
        )
        b0p = b0_cache.get(iid, [])
        guarded = guard.guard(b0p, vocab, in_vocab_preds=preds)
        oov_sum += oov_fp_rate(guarded, vocab)
        oov_count += 1
        done_set.add(iid)
        since_checkpoint += 1
        if since_checkpoint >= OOV_CHECKPOINT_EVERY:
            _save_oov_progress(
                vocab_size,
                len(image_ids),
                glip_tau,
                sorted(done_set),
                oov_sum,
                oov_count,
            )
            since_checkpoint = 0

    if oov_count == 0:
        return 1.0
    _save_oov_progress(
        vocab_size,
        len(image_ids),
        glip_tau,
        sorted(done_set),
        oov_sum,
        oov_count,
    )
    return float(oov_sum / oov_count)


def _b5_epi_from_rows(rows: list[dict], vs: int, ep, lvis, b0_cache, device: str) -> float:
    for r in rows:
        if r.get("vocab_size") == vs and r.get("method") == "B5_subset":
            return float(r["EpisodicAP_mean"])
    from ovdeploy.infer import run_episode_infer

    r_b5 = run_episode_infer(
        ep,
        "B5_subset",
        lvis,
        b0_preds_by_image=b0_cache,
        episode_vocab_for_oov=ep.vocab.cat_ids,
        backbone=BACKBONE,
        device=device,
    )
    return float(r_b5["EpisodicAP_mean"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--vocab-sizes", type=str, default=None)
    parser.add_argument("--proxy", action="store_true")
    args = parser.parse_args()

    if args.proxy:
        subprocess.run([sys.executable, "scripts/rv/run_proxy_glip.py"], cwd=ROOT, check=True)
        return

    if not args.gpu:
        raise SystemExit("GPU required — use --gpu on WSL")

    try:
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA not available")
    except ImportError as e:
        raise SystemExit(f"torch not available: {e}") from e

    vocab_sizes = _parse_vocab_sizes(args.vocab_sizes)

    cfg = load_config()
    ov = cfg["_ovdeploy"]
    strat_path = ov / "data/stratified_1k.json"
    if not strat_path.is_file():
        raise SystemExit(f"Missing {strat_path}")

    from ovdeploy.b0_cache import ensure_b0_preds
    from ovdeploy.episode import Episode, EpisodeVocab
    from ovdeploy.infer import run_episode_infer
    from ovdeploy.paths_util import load_lvis_minival
    from ovdeploy.vocab import freq_sorted_cat_ids
    from vocabguard.oov_guard import OOVGuard

    strat = json.loads(strat_path.read_text(encoding="utf-8"))
    image_ids = strat["image_ids"][: args.max_images]
    lvis = load_lvis_minival()
    device = "cuda:0"
    n_images = len(image_ids)

    rv_cfg = cfg.get("robustvocab", {})
    glip_tau = float(rv_cfg.get("glip_guard_tau", rv_cfg.get("guard_tau", 0.5)))
    print(
        f"GLIP native stratified: n={n_images} tau={glip_tau} |V|={vocab_sizes}",
        flush=True,
    )

    rows = _load_resume_rows(n_images, args.resume, glip_tau)
    done_vs = _completed_vocab_sizes(rows, glip_tau, vocab_sizes)
    if done_vs:
        print(f"Resume: skipping completed |V| in {sorted(done_vs)}", flush=True)

    b0_cache = ensure_b0_preds(image_ids, lvis, device=device, backbone=BACKBONE)
    guard = OOVGuard(
        alpha=float(rv_cfg.get("guard_alpha", 2.0)),
        beta=float(rv_cfg.get("guard_beta", 0.3)),
        tau=glip_tau,
    )

    for vs in vocab_sizes:
        if vs in done_vs:
            continue
        vocab = freq_sorted_cat_ids(lvis)[:vs]
        ep = Episode(
            episode_id=f"strat_glip_v{vs}",
            image_ids=image_ids,
            vocab=EpisodeVocab(cat_ids=vocab),
            vocab_size=vs,
            noise="none",
            split="stratified_1k",
        )
        have_baseline = {r.get("method") for r in rows if r.get("vocab_size") == vs}
        if "B0_full" not in have_baseline or "B5_subset" not in have_baseline:
            for bl in ("B0_full", "B5_subset"):
                if bl in have_baseline:
                    continue
                r = run_episode_infer(
                    ep,
                    bl,
                    lvis,
                    b0_preds_by_image=b0_cache,
                    episode_vocab_for_oov=vocab,
                    backbone=BACKBONE,
                    device=device,
                )
                rows.append(
                    {
                        "vocab_size": vs,
                        "method": bl,
                        "backbone": BACKBONE,
                        "EpisodicAP_mean": r["EpisodicAP_mean"],
                        "OOV_FP_mean": r["OOV_FP_mean"],
                        "gpu_used": True,
                        "n_images": r["n_images"],
                    }
                )
            _write_checkpoint(rows, n_images, glip_guard_tau=glip_tau, vocab_sizes=vocab_sizes)
            print(f"|V|={vs} baselines done (checkpoint before OOV loop)", flush=True)

        if _oov_sidecar_path(vs).is_file() and _load_oov_progress(vs, n_images, glip_tau) is None:
            _clear_oov_progress(vs)

        epi_b5 = _b5_epi_from_rows(rows, vs, ep, lvis, b0_cache, device)
        oov_guarded = _compute_rv_oov_incremental(
            image_ids,
            vocab,
            vs,
            b0_cache,
            guard,
            lvis,
            device,
            glip_tau,
            resume=args.resume,
        )

        rows = [r for r in rows if not (r.get("vocab_size") == vs and r.get("method") == "RV_full")]
        rows.append(
            {
                "vocab_size": vs,
                "method": "RV_full",
                "backbone": BACKBONE,
                "EpisodicAP_mean": epi_b5,
                "OOV_FP_mean": oov_guarded,
                "glip_guard_tau": glip_tau,
                "gpu_used": True,
                "n_images": n_images,
            }
        )
        b0_oov = next(
            r["OOV_FP_mean"] for r in rows if r["vocab_size"] == vs and r["method"] == "B0_full"
        )
        print(f"|V|={vs} B0 OOV={b0_oov:.3f} RV_full OOV={oov_guarded:.3f}", flush=True)
        _write_checkpoint(rows, n_images, glip_guard_tau=glip_tau, vocab_sizes=vocab_sizes)

    out = _write_checkpoint(rows, n_images, glip_guard_tau=glip_tau, vocab_sizes=vocab_sizes)
    data = json.loads(out.read_text(encoding="utf-8"))
    if _completed_vocab_sizes(rows, glip_tau, VOCAB_SIZES) == set(VOCAB_SIZES):
        data["status"] = "ok"
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
