"""Unified VocabGuard inference pipeline on OVDeploy episodes."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from ovdeploy.b0_cache import ensure_b0_preds, load_b0_preds
from ovdeploy.clip_vocab import clip_topk_cat_ids
from ovdeploy.episode import Episode
from ovdeploy.generator import image_gt_cat_ids, select_vocab_for_baseline
from ovdeploy.infer import run_episode_infer
from ovdeploy.metrics import episodic_ap_per_image, fp_non_gt_rate, oov_fp_rate
from ovdeploy.paths_util import cat_id_to_index, load_class_texts, load_lvis_minival
from ovdeploy.vocab import freq_sorted_cat_ids

from vocabguard.calib_head import apply_calib_bias, load_calib
from vocabguard.oov_guard import OOVGuard
from vocabguard.paths_util import load_config
from vocabguard.router import VocabRouter


def _prompts_map(lvis: dict, class_texts_raw: list) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for i, c in enumerate(sorted(lvis["categories"], key=lambda x: x["id"])):
        if i < len(class_texts_raw):
            t = class_texts_raw[i]
            out[c["id"]] = t if isinstance(t, list) else [str(t)]
        else:
            out[c["id"]] = [c["name"]]
    return out


def _load_image(backend, im: dict) -> np.ndarray | None:
    path = backend.image_path(im["file_name"])
    image = cv2.imread(str(path))
    if image is None:
        return None
    return image[:, :, [2, 1, 0]]


def _episode_dropped_hints(episode: Episode) -> list[int]:
    core = set(episode.vocab.cat_ids)
    return [int(k) for k in episode.vocab.prompts if int(k) not in core]


def _merge_router_hints(
    dropped: Sequence[int],
    b0_hints: Sequence[int],
) -> tuple[list[int], list[int]]:
    priority = list(dropped)
    b0_only: list[int] = []
    priority_set = set(priority)
    for c in b0_hints:
        if c not in priority_set:
            b0_only.append(c)
    return priority, b0_only


def _b0_hint_classes(
    b0_preds: list[dict] | None,
    user_vocab: Sequence[int],
    score_thr: float = 0.25,
) -> list[int]:
    if not b0_preds:
        return []
    core = set(user_vocab)
    scored: dict[int, float] = {}
    for p in b0_preds:
        cid = p.get("category_id", -1)
        if cid in core or cid < 0:
            continue
        s = float(p.get("score", 0.0))
        if s >= score_thr:
            scored[cid] = max(scored.get(cid, 0.0), s)
    return [c for c, _ in sorted(scored.items(), key=lambda x: -x[1])]


def run_vocabguard_episode(
    episode: Episode,
    method: str,
    lvis: dict,
    backend,
    router: VocabRouter | None = None,
    guard: OOVGuard | None = None,
    calib_ckpt: Path | None = None,
    max_images: int = 0,
    device: str = "cuda:0",
    b0_preds_by_image: dict[int, list[dict]] | None = None,
    backbone: str = "yolo",
) -> dict[str, Any]:
    """Run VG_router, VG_full, or M2_calib on one episode."""
    cfg = load_config()
    vg_cfg = cfg.get("vocabguard", {})
    router = router or VocabRouter(
        delta=int(vg_cfg.get("router_delta", 3)),
        cand_pool=int(vg_cfg.get("router_cand_pool", 64)),
        missing_class_extra_delta=int(vg_cfg.get("missing_class_extra_delta", 3)),
    )
    guard = guard or OOVGuard(
        alpha=float(vg_cfg.get("guard_alpha", 2.0)),
        beta=float(vg_cfg.get("guard_beta", 0.3)),
        tau=float(vg_cfg.get("guard_tau", 0.5)),
    )

    class_names, class_texts_raw = load_class_texts()
    cid2idx = cat_id_to_index(lvis)
    all_cat_ids = [c["id"] for c in lvis["categories"]]
    freq_cats = freq_sorted_cat_ids(lvis)
    prompts_map = _prompts_map(lvis, class_texts_raw)

    from ovdeploy.paths_util import load_paths

    paths = load_paths()
    ann_path = paths["_yolo"] / paths["data"]["lvis_minival_ann"]
    lvis_local = json.loads(ann_path.read_text(encoding="utf-8"))
    gt_by_img: dict = {}
    for a in lvis_local["annotations"]:
        gt_by_img.setdefault(a["image_id"], {"boxes": [], "cat_ids": []})
        gt_by_img[a["image_id"]]["boxes"].append(a["bbox"])
        gt_by_img[a["image_id"]]["cat_ids"].append(a["category_id"])
    id_to_im = {im["id"]: im for im in lvis_local["images"]}

    images = [id_to_im[iid] for iid in episode.image_ids if iid in id_to_im]
    if max_images:
        images = images[:max_images]

    user_vocab = list(episode.vocab.cat_ids)
    vs = min(episode.vocab_size, len(all_cat_ids))

    calib_model = None
    if method == "M2_calib" and calib_ckpt and calib_ckpt.is_file():
        calib_model = load_calib(
            calib_ckpt,
            feat_dim=int(vg_cfg.get("calib_feat_dim", 256)),
            hidden=int(vg_cfg.get("calib_hidden", 128)),
        )
        calib_model.eval()

    ap_list, fp_list, oov_list = [], [], []
    t0 = time.perf_counter()

    for im in images:
        iid = im["id"]
        image_rgb = _load_image(backend, im)
        if image_rgb is None:
            continue

        if method in ("VG_router", "VG_full", "M2_calib"):
            priority_hints: list[int] = []
            b0_hints: list[int] = []
            cand_pool_override = None
            if episode.noise == "missing_class":
                dropped = _episode_dropped_hints(episode)
                raw_b0: list[int] = []
                if b0_preds_by_image is not None:
                    raw_b0 = _b0_hint_classes(
                        b0_preds_by_image.get(iid, []),
                        user_vocab,
                        score_thr=float(vg_cfg.get("b0_hint_score_thr", 0.25)),
                    )
                priority_hints, b0_hints = _merge_router_hints(dropped, raw_b0)
                cand_pool_override = int(vg_cfg.get("router_cand_pool_missing", 256))
            routed_vocab = router.route(
                user_vocab,
                vs,
                image_rgb,
                iid,
                backend,
                lvis,
                freq_cats=freq_cats,
                all_cat_ids=all_cat_ids,
                noise=episode.noise,
                b0_hint_classes=b0_hints,
                priority_hint_classes=priority_hints,
                cand_pool_override=cand_pool_override,
            )
            if hasattr(backend, "predict") and backend.name == "yolo_ext":
                preds = backend.predict(image_rgb, routed_vocab, iid, lvis)
            else:
                preds = backend.predict(
                    image_rgb,
                    [],
                    routed_vocab,
                    iid,
                    class_names=class_names,
                    class_texts_raw=class_texts_raw,
                    cid2idx=cid2idx,
                )
            if calib_model is not None and hasattr(backend, "encode_image"):
                import torch

                feat = backend.encode_image(image_rgb, iid, lvis)
                with torch.no_grad():
                    bias = (
                        calib_model(torch.from_numpy(feat).unsqueeze(0).float(), len(routed_vocab))
                        .squeeze(0)
                        .numpy()
                    )
                preds = apply_calib_bias(preds, bias, routed_vocab)

            gt = gt_by_img.get(iid, {"boxes": [], "cat_ids": []})
            ap_list.append(
                episodic_ap_per_image(preds, gt["boxes"], gt["cat_ids"], user_vocab)
            )
            fp_list.append(fp_non_gt_rate(preds, user_vocab))

            if method in ("VG_full", "M2_calib") and b0_preds_by_image is not None:
                b0p = list(b0_preds_by_image.get(iid, []))
                guarded = guard.guard(b0p, user_vocab, in_vocab_preds=preds)
                oov_list.append(oov_fp_rate(guarded, user_vocab))
            elif b0_preds_by_image is not None:
                b0p = b0_preds_by_image.get(iid, [])
                oov_list.append(oov_fp_rate(b0p, user_vocab))

        elif method == "B5_subset":
            if hasattr(backend, "predict") and getattr(backend, "name", "") == "yolo_ext":
                preds = backend.predict(image_rgb, user_vocab, iid, lvis)
            else:
                preds = backend.predict(
                    image_rgb, [], user_vocab, iid,
                    class_names=class_names, class_texts_raw=class_texts_raw, cid2idx=cid2idx,
                )
            gt = gt_by_img.get(iid, {"boxes": [], "cat_ids": []})
            ap_list.append(episodic_ap_per_image(preds, gt["boxes"], gt["cat_ids"], user_vocab))
            fp_list.append(fp_non_gt_rate(preds, user_vocab))
            if b0_preds_by_image is not None:
                oov_list.append(oov_fp_rate(b0_preds_by_image.get(iid, []), user_vocab))

        elif method == "B4_clip":
            vocab_ids = clip_topk_cat_ids(
                image_rgb, all_cat_ids, prompts_map, vs, freq_cats, seed=episode.seed + iid
            )
            if hasattr(backend, "predict") and getattr(backend, "name", "") == "yolo_ext":
                preds = backend.predict(image_rgb, vocab_ids, iid, lvis)
            else:
                preds = backend.predict(
                    image_rgb, [], vocab_ids, iid,
                    class_names=class_names, class_texts_raw=class_texts_raw, cid2idx=cid2idx,
                )
            gt = gt_by_img.get(iid, {"boxes": [], "cat_ids": []})
            ap_list.append(episodic_ap_per_image(preds, gt["boxes"], gt["cat_ids"], user_vocab))
            fp_list.append(fp_non_gt_rate(preds, user_vocab))
            if b0_preds_by_image is not None:
                oov_list.append(oov_fp_rate(b0_preds_by_image.get(iid, []), user_vocab))

        elif method == "B1_oracle":
            img_cats = image_gt_cat_ids(lvis)
            import random

            rng = random.Random(episode.seed + iid)
            vocab_ids = select_vocab_for_baseline(
                "B1_oracle", [iid], vs, img_cats, all_cat_ids, freq_cats, rng
            )
            if hasattr(backend, "predict") and getattr(backend, "name", "") == "yolo_ext":
                preds = backend.predict(image_rgb, vocab_ids, iid, lvis)
            else:
                preds = backend.predict(
                    image_rgb, [], vocab_ids, iid,
                    class_names=class_names, class_texts_raw=class_texts_raw, cid2idx=cid2idx,
                )
            gt = gt_by_img.get(iid, {"boxes": [], "cat_ids": []})
            ap_list.append(episodic_ap_per_image(preds, gt["boxes"], gt["cat_ids"], user_vocab))
            fp_list.append(fp_non_gt_rate(preds, user_vocab))
            if b0_preds_by_image is not None:
                oov_list.append(oov_fp_rate(b0_preds_by_image.get(iid, []), user_vocab))

    elapsed = time.perf_counter() - t0
    return {
        "episode_id": episode.episode_id,
        "method": method,
        "backbone": backbone,
        "vocab_size": vs,
        "noise": episode.noise,
        "n_images": len(ap_list),
        "EpisodicAP_mean": float(np.mean(ap_list)) if ap_list else 0.0,
        "FP_nonGT_mean": float(np.mean(fp_list)) if fp_list else 0.0,
        "OOV_FP_mean": float(np.mean(oov_list)) if oov_list else 0.0,
        "latency_ms_per_image": (elapsed / max(len(ap_list), 1)) * 1000,
    }


def run_baseline_via_ovdeploy(
    episode: Episode,
    baseline: str,
    lvis: dict,
    max_images: int = 0,
    b0_preds_by_image: dict | None = None,
    backbone: str = "yolo",
) -> dict[str, Any]:
    """Delegate B0/B5 to OVDeploy infer for exact baseline parity."""
    return run_episode_infer(
        episode,
        baseline,
        lvis,
        max_images=max_images,
        b0_preds_by_image=b0_preds_by_image,
        episode_vocab_for_oov=list(episode.vocab.cat_ids),
        backbone=backbone,
    )
