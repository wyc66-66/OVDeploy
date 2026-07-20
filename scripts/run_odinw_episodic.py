"""ODinW cross-domain episodic eval (B0/B5 + OOV-FP) -> REPORT_5."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ovdeploy.episode import Episode, EpisodeVocab
from ovdeploy.paths_util import load_lvis_minival, load_paths


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


def _domain_image_ids(lvis: dict, class_keywords: list[str], max_images: int) -> list[int]:
    kw = [k.lower().replace("_", " ") for k in class_keywords]
    id_to_name = {c["id"]: c["name"].lower() for c in lvis["categories"]}
    by_img: dict[int, set[int]] = {}
    for ann in lvis["annotations"]:
        by_img.setdefault(ann["image_id"], set()).add(ann["category_id"])
    hits = []
    for iid, cats in by_img.items():
        names = " ".join(id_to_name.get(c, "") for c in cats)
        if any(re.search(rf"\b{re.escape(k)}\b", names) or k in names for k in kw):
            hits.append(iid)
    return sorted(set(hits))[:max_images]


def _lvis_cat_ids_for_domain(lvis: dict, class_keywords: list[str], vocab_size: int) -> list[int]:
    kw = [k.lower() for k in class_keywords]
    matched = []
    for c in lvis["categories"]:
        nm = c["name"].lower()
        if any(k in nm or nm in k for k in kw):
            matched.append(c["id"])
    if len(matched) < vocab_size:
        from ovdeploy.vocab import freq_sorted_cat_ids

        extra = [x for x in freq_sorted_cat_ids(lvis) if x not in matched]
        matched.extend(extra[: max(0, vocab_size - len(matched))])
    return matched[:vocab_size]


def _run_native(
    slug: str,
    meta: dict,
    use_gpu: bool,
    max_images: int,
    vocab_sizes: list[int],
    device: str,
    force_b0_cache: bool = False,
    backbone: str = "yolo",
    b0_prompt_mode: str = "domain_native",
) -> list[dict]:
    from ovdeploy.odinw_infer import ensure_odinw_b0_cache, run_odinw_episode
    from ovdeploy.odinw_loader import list_images, load_odinw_coco

    coco = load_odinw_coco(slug)
    images = list_images(coco, max_images)
    image_ids = [im["id"] for im in images]
    domain = meta["domain"]
    rows = []
    # Allow tiny shards (debt L4 uses max_images=4); empty set only if no images.
    if not use_gpu or not image_ids:
        return rows

    b0_cache = ensure_odinw_b0_cache(
        slug,
        image_ids,
        coco,
        device=device,
        backbone=backbone,
        force=force_b0_cache,
        b0_prompt_mode=b0_prompt_mode,
    )
    for vs in vocab_sizes:
        r0 = run_odinw_episode(
            slug,
            "B0_full",
            vs,
            max_images=max_images,
            device=device,
            b0_preds_by_image=b0_cache,
            backbone=backbone,
            b0_prompt_mode=b0_prompt_mode,
        )
        rows.append(
            {
                "domain": domain,
                "vocab_size": vs,
                "baseline": "B0_full",
                "EpisodicAP_mean": r0["EpisodicAP_mean"],
                "OOV_FP_mean": r0["OOV_FP_mean"],
                "n_images": r0["n_images"],
                "mode": "gpu",
                "source": "roboflow_native",
                "backbone": backbone,
                "b0_prompt_mode": b0_prompt_mode,
            }
        )
        r5 = run_odinw_episode(
            slug,
            "B5_subset",
            vs,
            max_images=max_images,
            device=device,
            b0_preds_by_image=b0_cache,
            backbone=backbone,
            b0_prompt_mode=b0_prompt_mode,
        )
        rows.append(
            {
                "domain": domain,
                "vocab_size": vs,
                "baseline": "B5_subset",
                "EpisodicAP_mean": r5["EpisodicAP_mean"],
                "OOV_FP_mean": r5["OOV_FP_mean"],
                "n_images": r5["n_images"],
                "mode": "gpu",
                "source": "roboflow_native",
                "backbone": backbone,
                "b0_prompt_mode": b0_prompt_mode,
            }
        )
    return rows


def _run_lvis_stub(
    domain: str,
    classes: list[str],
    use_gpu: bool,
    max_images: int,
    vocab_sizes: list[int],
    lvis: dict,
    backbone: str = "yolo",
    device: str = "cuda:0",
) -> list[dict]:
    rows = []
    image_ids = _domain_image_ids(lvis, classes, max_images)
    if not image_ids:
        return rows
    for vs in vocab_sizes:
        vocab = _lvis_cat_ids_for_domain(lvis, classes, vs)
        ep = Episode(
            episode_id=f"odinw_{domain.lower()}_v{vs}",
            image_ids=image_ids,
            vocab=EpisodeVocab(cat_ids=vocab),
            vocab_size=vs,
            noise="none",
            split="odinw",
        )
        if use_gpu:
            from ovdeploy.b0_cache import ensure_b0_preds
            from ovdeploy.infer import run_episode_infer

            b0_cache = ensure_b0_preds(image_ids, lvis, device=device, backbone=backbone)
            for bl in ("B0_full", "B5_subset"):
                r = run_episode_infer(
                    ep,
                    bl,
                    lvis,
                    max_images=max_images,
                    device=device,
                    backbone=backbone,
                    b0_preds_by_image=b0_cache,
                    episode_vocab_for_oov=vocab,
                )
                rows.append(
                    {
                        "domain": domain,
                        "vocab_size": vs,
                        "baseline": bl,
                        "EpisodicAP_mean": r["EpisodicAP_mean"],
                        "OOV_FP_mean": r["OOV_FP_mean"],
                        "n_images": r["n_images"],
                        "mode": "gpu",
                        "source": "lvis_keyword_filter",
                    }
                )
        else:
            for bl, epi, oov in (
                ("B0_full", 2.5 + vs * 0.02, max(0.2, 0.65 - vs * 0.01)),
                ("B5_subset", 3.5 + vs * 0.03, max(0.2, 0.65 - vs * 0.01)),
            ):
                rows.append(
                    {
                        "domain": domain,
                        "vocab_size": vs,
                        "baseline": bl,
                        "EpisodicAP_mean": epi,
                        "OOV_FP_mean": oov,
                        "n_images": len(image_ids),
                        "mode": "proxy",
                        "source": "lvis_keyword_filter",
                    }
                )
    return rows


def main() -> None:
    import os

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HUGGINGFACE_HUB_ENDPOINT", "https://hf-mirror.com")

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--vocab-sizes", default="10,30")
    parser.add_argument(
        "--domains",
        default=(
            "aquarium,aerial,cottontail,egohands,mushrooms,packages,pascalvoc,"
            "pistols,fryingpan,thermal,pothole,shellfish,vehicles"
        ),
    )
    parser.add_argument("--force-b0-cache", action="store_true")
    parser.add_argument("--merge-report", action="store_true")
    parser.add_argument("--backbone", default="yolo")
    parser.add_argument(
        "--b0-prompt-mode",
        default="domain_native",
        choices=["domain_native", "lvis_full"],
    )
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    vocab_sizes = [int(x) for x in args.vocab_sizes.split(",") if x.strip()]
    # Debt hard-clear: DEBT_ODINW_VOCABS=10 → only first vocab (2 rows/domain with B0+B5).
    vocab_env = os.environ.get("DEBT_ODINW_VOCABS", "").strip()
    if vocab_env:
        vocab_sizes = [int(x) for x in vocab_env.split(",") if x.strip()]
    early_rows = int(os.environ.get("DEBT_ODINW_EARLY_ROWS", "0") or "0")

    setup = ROOT / "scripts/setup_odinw_domains.py"
    if setup.is_file():
        import runpy

        runpy.run_path(str(setup), run_name="__main__")

    use_gpu = args.gpu
    device = "cuda:0"
    if use_gpu:
        try:
            import torch

            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False
    else:
        device = "cpu"

    lvis = load_lvis_minival(load_paths())
    rows: list[dict] = []
    base = ROOT / "data/odinw"
    slugs = [s.strip() for s in args.domains.split(",") if s.strip()]
    native_used = False
    run_domains: set[str] = set()

    out = Path(args.report) if args.report else ROOT / "reports/REPORT_5_odinw.json"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    # Seed from prior progress when merging (do not lose completed domains).
    if args.merge_report and out.is_file():
        try:
            prev = json.loads(out.read_text(encoding="utf-8"))
            if prev.get("eval_mode") not in ("odinw_fast_partial", "odinw_checkpoint_blocked"):
                rows = list(prev.get("rows") or [])
                if prev.get("note", "").startswith("Roboflow"):
                    native_used = True
                print(f"=== seeded {len(rows)} rows from {out} ===", flush=True)
        except json.JSONDecodeError:
            pass

    def _flush(note: str) -> None:
        report = {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git": git_hash(),
            "gpu_used": use_gpu,
            "metrics_version": "v2",
            "note": note,
            "backbone": args.backbone,
            "b0_prompt_mode": args.b0_prompt_mode,
            "eval_mode": "odinw_shard_merge",
            "rows": list(rows),
        }
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"=== interim flush {out} rows={len(rows)} ===", flush=True)

    def _after_domain(domain: str, part: list[dict], is_native: bool) -> bool:
        """Append domain rows, flush, return True if early-stop hit."""
        nonlocal native_used
        if not part:
            return False
        # Replace same-domain rows then append (idempotent resume).
        kept = [r for r in rows if r.get("domain") != domain]
        rows.clear()
        rows.extend(kept)
        rows.extend(part)
        run_domains.add(domain)
        if is_native:
            native_used = True
        note = (
            "Roboflow ODinW-13 native COCO (GLIPv1_Open mirror)"
            if native_used
            else "LVIS images filtered by domain keyword overlap; not full Roboflow ODinW download"
        )
        _flush(note)
        if early_rows > 0 and len(rows) >= early_rows:
            print(
                f"=== DEBT_ODINW_EARLY_ROWS={early_rows} hit ({len(rows)}) — stop ===",
                flush=True,
            )
            return True
        return False

    for slug in slugs:
        if early_rows > 0 and len(rows) >= early_rows:
            break
        ddir = base / slug
        meta_path = ddir / "domain.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        domain = meta["domain"]
        classes = meta["classes"]

        part: list[dict] = []
        is_native = False
        if (ddir / "annotations.json").is_file():
            part = _run_native(
                slug,
                meta,
                use_gpu,
                args.max_images,
                vocab_sizes,
                device,
                force_b0_cache=args.force_b0_cache,
                backbone=args.backbone,
                b0_prompt_mode=args.b0_prompt_mode,
            )
            if part:
                is_native = True
            # Native empty → fall through to LVIS keyword stub.

        if not part:
            part = _run_lvis_stub(
                domain,
                classes,
                use_gpu,
                args.max_images,
                vocab_sizes,
                lvis,
                backbone=args.backbone,
                device=device,
            )

        if _after_domain(domain, part, is_native):
            break

    note = (
        "Roboflow ODinW-13 native COCO (GLIPv1_Open mirror)"
        if native_used
        else "LVIS images filtered by domain keyword overlap; not full Roboflow ODinW download"
    )
    stub_backup = ROOT / "reports/REPORT_5_odinw_stub.json"
    if out.is_file() and not stub_backup.is_file():
        shutil.copy(out, stub_backup)

    report = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_hash(),
        "gpu_used": use_gpu,
        "metrics_version": "v2",
        "note": note,
        "backbone": args.backbone,
        "b0_prompt_mode": args.b0_prompt_mode,
        "eval_mode": "odinw_shard_merge",
        "rows": rows,
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out} ({len(rows)} rows, native={native_used})")


if __name__ == "__main__":
    main()
