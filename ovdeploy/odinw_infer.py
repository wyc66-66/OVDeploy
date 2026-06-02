"""YOLO-World inference on native ODinW COCO domains."""
from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from ovdeploy.backends.base import get_backend
from ovdeploy.odinw_loader import (
    all_category_ids,
    class_texts_for_ids,
    gt_by_image,
    image_path,
    list_images,
    load_odinw_coco,
    sample_episode_vocab,
)
from ovdeploy.odinw_metrics import episodic_ap_per_image, oov_fp_rate


def _predict_domain(
    backend,
    slug: str,
    image_rgb: np.ndarray,
    vocab_ids: list[int],
    coco: dict,
    meta: dict | None = None,
) -> list[dict]:
    from ovdeploy.odinw_loader import load_domain_meta

    if meta is None:
        meta = load_domain_meta(slug)
    names, texts = class_texts_for_ids(coco, vocab_ids, meta)
    id2idx = {cid: i for i, cid in enumerate(vocab_ids)}
    return backend.predict(
        image_rgb,
        texts,
        vocab_ids,
        image_id=0,
        class_names=names,
        class_texts_raw=texts,
        cid2idx=id2idx,
    )


def run_odinw_episode(
    slug: str,
    baseline: str,
    vocab_size: int,
    seed: int = 42,
    max_images: int = 100,
    device: str = "cuda:0",
    b0_preds_by_image: dict[int, list[dict]] | None = None,
    backbone: str = "yolo",
) -> dict[str, Any]:
    coco = load_odinw_coco(slug)
    from ovdeploy.odinw_loader import load_domain_meta

    meta = load_domain_meta(slug)
    domain_classes = meta.get("classes", [])
    all_ids = all_category_ids(coco)
    episode_vocab = sample_episode_vocab(coco, domain_classes, vocab_size, seed)
    gt_map = gt_by_image(coco)
    images = list_images(coco, max_images)

    if baseline == "B0_full":
        infer_vocab = all_ids
    elif baseline in ("B5_subset", "M1_adapter"):
        infer_vocab = episode_vocab
    else:
        infer_vocab = episode_vocab

    backend = get_backend(backbone, device=device)
    ap_list = []
    oov_list = []
    t0 = time.perf_counter()

    for im in images:
        iid = im["id"]
        path = image_path(slug, im["file_name"])
        if slug != getattr(backend, "name", ""):
            pass
        img = cv2.imread(str(path))
        if img is None:
            continue
        rgb = img[:, :, [2, 1, 0]]
        preds = _predict_domain(backend, slug, rgb, infer_vocab, coco, meta)
        gt = gt_map.get(iid, {"boxes": [], "cat_ids": []})
        ap_list.append(
            episodic_ap_per_image(preds, gt["boxes"], gt["cat_ids"], episode_vocab)
        )
        if b0_preds_by_image is not None:
            oov_list.append(oov_fp_rate(b0_preds_by_image.get(iid, []), episode_vocab))

    elapsed = time.perf_counter() - t0
    return {
        "domain": meta.get("domain", slug),
        "vocab_size": vocab_size,
        "baseline": baseline,
        "n_images": len(ap_list),
        "EpisodicAP_mean": float(np.mean(ap_list)) if ap_list else 0.0,
        "OOV_FP_mean": float(np.mean(oov_list)) if oov_list else 0.0,
        "latency_ms_per_image": (elapsed / max(len(ap_list), 1)) * 1000,
        "episode_vocab": episode_vocab,
    }


def ensure_odinw_b0_cache(
    slug: str,
    image_ids: list[int],
    coco: dict,
    device: str = "cuda:0",
    backbone: str = "yolo",
    force: bool = False,
) -> dict[int, list[dict]]:
    import json
    from pathlib import Path

    cache_dir = Path(__file__).resolve().parents[1] / "data" / "b0_cache" / "odinw" / slug
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[int, list[dict]] = {}
    all_ids = all_category_ids(coco)
    backend = get_backend(backbone, device=device)
    id_to_im = {im["id"]: im for im in coco["images"]}
    from ovdeploy.odinw_loader import load_domain_meta

    meta = load_domain_meta(slug)

    for iid in image_ids:
        cp = cache_dir / f"{iid}.json"
        if cp.is_file() and not force:
            out[iid] = json.loads(cp.read_text(encoding="utf-8"))
            continue
        im = id_to_im.get(iid)
        if not im:
            continue
        path = image_path(slug, im["file_name"])
        img = cv2.imread(str(path))
        if img is None:
            continue
        rgb = img[:, :, [2, 1, 0]]
        preds = _predict_domain(backend, slug, rgb, all_ids, coco, meta)
        cp.write_text(json.dumps(preds), encoding="utf-8")
        out[iid] = preds

    for iid in image_ids:
        if iid not in out:
            cp = cache_dir / f"{iid}.json"
            if cp.is_file():
                out[iid] = json.loads(cp.read_text(encoding="utf-8"))
    return out
