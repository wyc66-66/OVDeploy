"""Run RobustVocab (RV_full) on native ODinW domains."""
from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from ovdeploy.odinw_loader import (
    class_texts_for_ids,
    gt_by_image,
    image_path,
    list_images,
    load_domain_meta,
    load_odinw_coco,
    sample_episode_vocab,
)
from ovdeploy.odinw_metrics import episodic_ap_per_image, oov_fp_rate

from robustvocab.paths_util import load_config
from robustvocab.recover import VocabRecover
from vocabguard.oov_guard import OOVGuard


def run_odinw_rv_full(
    slug: str,
    vocab_size: int,
    seed: int = 42,
    max_images: int = 30,
    device: str = "cuda:0",
    b0_preds_by_image: dict[int, list[dict]] | None = None,
    backbone: str = "yolo",
    deployment_strict: bool = True,
    recover_overrides: dict | None = None,
) -> dict[str, Any]:
    from ovdeploy.backends.base import get_backend

    cfg = load_config()
    rv_cfg = cfg.get("robustvocab", {})
    overrides = recover_overrides or {}

    coco = load_odinw_coco(slug)
    meta = load_domain_meta(slug)
    domain_classes = meta.get("classes", [])
    episode_vocab = sample_episode_vocab(coco, domain_classes, vocab_size, seed)
    gt_map = gt_by_image(coco)
    images = list_images(coco, max_images)

    backend = get_backend(backbone, device=device)
    recover = VocabRecover(
        delta=int(overrides.get("recover_delta", rv_cfg.get("recover_delta", 8))),
        delta_missing=int(overrides.get("recover_delta_missing", rv_cfg.get("recover_delta_missing", 20))),
        b0_score_thr=float(overrides.get("b0_score_thr", rv_cfg.get("b0_score_thr", 0.15))),
        b0_top_k=int(overrides.get("b0_top_k", rv_cfg.get("b0_top_k", 32))),
        cooccur_weight=float(overrides.get("cooccur_weight", rv_cfg.get("cooccur_weight", 0.35))),
        use_cooccur=bool(overrides.get("use_cooccur", rv_cfg.get("use_cooccur", True))),
        two_round=bool(overrides.get("two_round_probe", rv_cfg.get("two_round_probe", True))),
    )
    guard = OOVGuard(
        alpha=float(rv_cfg.get("guard_alpha", 2.0)),
        beta=float(rv_cfg.get("guard_beta", 0.3)),
        tau=float(rv_cfg.get("guard_tau", 0.5)),
    )

    names, texts = class_texts_for_ids(coco, episode_vocab, meta)
    id2idx = {cid: i for i, cid in enumerate(episode_vocab)}

    ap_list, oov_list = [], []
    t0 = time.perf_counter()

    for im in images:
        iid = im["id"]
        path = image_path(slug, im["file_name"])
        img = cv2.imread(str(path))
        if img is None:
            continue
        rgb = img[:, :, [2, 1, 0]]
        b0_preds = list(b0_preds_by_image.get(iid, [])) if b0_preds_by_image else []

        routed = episode_vocab
        if b0_preds:
            lvis_stub = {"categories": [{"id": c["id"]} for c in coco["categories"]]}
            freq_stub = [c["id"] for c in coco["categories"]]
            routed = recover.recover(
                episode_vocab,
                vocab_size,
                rgb,
                iid,
                backend,
                lvis_stub,
                b0_preds=b0_preds,
                noise="missing_class",
                freq_cats=freq_stub,
                all_cat_ids=[c["id"] for c in coco["categories"]],
            )
            names_r, texts_r = class_texts_for_ids(coco, routed, meta)
            id2idx_r = {cid: i for i, cid in enumerate(routed)}
            if hasattr(backend, "score_classes"):
                preds = backend.predict(
                    rgb, texts_r, routed, iid,
                    class_names=names_r, class_texts_raw=texts_r, cid2idx=id2idx_r,
                )
            else:
                preds = backend.predict(
                    rgb, texts, episode_vocab, iid,
                    class_names=names, class_texts_raw=texts, cid2idx=id2idx,
                )
        else:
            preds = backend.predict(
                rgb, texts, episode_vocab, iid,
                class_names=names, class_texts_raw=texts, cid2idx=id2idx,
            )

        gt = gt_map.get(iid, {"boxes": [], "cat_ids": []})
        ap_list.append(
            episodic_ap_per_image(preds, gt["boxes"], gt["cat_ids"], episode_vocab)
        )
        if b0_preds_by_image is not None:
            guarded = guard.guard(b0_preds, episode_vocab, in_vocab_preds=preds)
            oov_list.append(oov_fp_rate(guarded, episode_vocab))

    elapsed = time.perf_counter() - t0
    return {
        "domain": meta.get("domain", slug),
        "vocab_size": vocab_size,
        "method": "RV_full",
        "n_images": len(ap_list),
        "EpisodicAP_mean": float(np.mean(ap_list)) if ap_list else 0.0,
        "OOV_FP_mean": float(np.mean(oov_list)) if oov_list else 0.0,
        "latency_ms_per_image": (elapsed / max(len(ap_list), 1)) * 1000,
    }
