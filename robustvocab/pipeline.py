"""Unified RobustVocab inference on OVDeploy episodes (deployment-strict)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from ovdeploy.episode import Episode
from ovdeploy.metrics import episodic_ap_per_image, fp_non_gt_rate, oov_fp_rate
from ovdeploy.paths_util import cat_id_to_index, load_class_texts, load_lvis_minival
from ovdeploy.vocab import freq_sorted_cat_ids

from robustvocab.paths_util import load_config
from robustvocab.prompt_align import PromptAlign
from robustvocab.recover import VocabRecover

from vocabguard.oov_guard import OOVGuard
from vocabguard.router import VocabRouter


def _load_image(backend, im: dict) -> np.ndarray | None:
    path = backend.image_path(im["file_name"])
    image = cv2.imread(str(path))
    if image is None:
        return None
    return image[:, :, [2, 1, 0]]


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


def _gt_index(lvis_local: dict) -> dict:
    gt_by_img: dict = {}
    for a in lvis_local["annotations"]:
        gt_by_img.setdefault(a["image_id"], {"boxes": [], "cat_ids": []})
        gt_by_img[a["image_id"]]["boxes"].append(a["bbox"])
        gt_by_img[a["image_id"]]["cat_ids"].append(a["category_id"])
    return gt_by_img


def _build_modules(
    cfg: dict,
    recover_overrides: dict | None = None,
) -> tuple[VocabRouter, VocabRecover, OOVGuard, PromptAlign]:
    rv = cfg.get("robustvocab", {})
    overrides = recover_overrides or {}
    router = VocabRouter(
        delta=int(rv.get("router_delta", 5)),
        cand_pool=int(rv.get("router_cand_pool", 128)),
        missing_class_extra_delta=int(rv.get("missing_class_extra_delta", 15)),
    )
    recover = VocabRecover(
        delta=int(overrides.get("recover_delta", rv.get("recover_delta", 8))),
        delta_missing=int(overrides.get("recover_delta_missing", rv.get("recover_delta_missing", 20))),
        b0_score_thr=float(overrides.get("b0_score_thr", rv.get("b0_score_thr", 0.15))),
        b0_top_k=int(overrides.get("b0_top_k", rv.get("b0_top_k", 32))),
        cooccur_weight=float(overrides.get("cooccur_weight", rv.get("cooccur_weight", 0.35))),
        use_cooccur=bool(overrides.get("use_cooccur", rv.get("use_cooccur", True))),
        two_round=bool(overrides.get("two_round_probe", rv.get("two_round_probe", True))),
        round2_top=int(overrides.get("round2_top", rv.get("round2_top", 12))),
        cand_pool=int(overrides.get("router_cand_pool_missing", rv.get("router_cand_pool_missing", 256))),
    )
    guard = OOVGuard(
        alpha=float(rv.get("guard_alpha", 2.0)),
        beta=float(rv.get("guard_beta", 0.3)),
        tau=float(rv.get("guard_tau", 0.5)),
    )
    align = PromptAlign()
    return router, recover, guard, align


def _route_vocab(
    method: str,
    episode: Episode,
    user_vocab: list[int],
    vs: int,
    image_rgb: np.ndarray,
    iid: int,
    backend,
    lvis: dict,
    router: VocabRouter,
    recover: VocabRecover,
    b0_preds: list[dict] | None,
    deployment_strict: bool,
    freq_cats: list[int],
    all_cat_ids: list[int],
    rv_cfg: dict,
) -> list[int]:
    noise = episode.noise
    if noise == "synonym" and method == "RV_full":
        return list(user_vocab)

    use_recover = method in ("RV_recover", "RV_full") and (
        noise == "missing_class" or method.startswith("RV")
    )

    if use_recover and noise == "missing_class":
        return recover.recover(
            user_vocab,
            vs,
            image_rgb,
            iid,
            backend,
            lvis,
            b0_preds=b0_preds,
            noise=noise,
            freq_cats=freq_cats,
            all_cat_ids=all_cat_ids,
        )

    b0_hints: list[int] = []
    priority_hints: list[int] = []
    cand_pool_override = None

    if noise == "missing_class" and not deployment_strict:
        from vocabguard.infer_pipeline import _episode_dropped_hints, _merge_router_hints

        dropped = _episode_dropped_hints(episode)
        raw_b0 = _b0_hint_classes(
            b0_preds,
            user_vocab,
            score_thr=float(rv_cfg.get("b0_hint_score_thr", 0.10)),
        )
        priority_hints, b0_hints = _merge_router_hints(dropped, raw_b0)
        cand_pool_override = int(rv_cfg.get("router_cand_pool_missing", 256))
    elif noise == "missing_class" and deployment_strict:
        b0_hints = _b0_hint_classes(
            b0_preds,
            user_vocab,
            score_thr=float(rv_cfg.get("b0_hint_score_thr", 0.10)),
        )
        cand_pool_override = int(rv_cfg.get("router_cand_pool_missing", 256))

    return router.route(
        user_vocab,
        vs,
        image_rgb,
        iid,
        backend,
        lvis,
        freq_cats=freq_cats,
        all_cat_ids=all_cat_ids,
        noise=noise,
        b0_hint_classes=b0_hints,
        priority_hint_classes=priority_hints,
        cand_pool_override=cand_pool_override,
    )


def run_robustvocab_episode(
    episode: Episode,
    method: str,
    lvis: dict,
    backend,
    max_images: int = 0,
    device: str = "cuda:0",
    b0_preds_by_image: dict[int, list[dict]] | None = None,
    backbone: str = "yolo",
    deployment_strict: bool | None = None,
    recover_overrides: dict | None = None,
) -> dict[str, Any]:
    """Run VG_full_strict, RV_recover, or RV_full on one episode."""
    cfg = load_config()
    rv_cfg = cfg.get("robustvocab", {})
    if deployment_strict is None:
        deployment_strict = bool(rv_cfg.get("deployment_strict", True))

    router, recover, guard, align = _build_modules(cfg, recover_overrides)

    from ovdeploy.paths_util import load_paths

    paths = load_paths()
    ann_path = paths["_yolo"] / paths["data"]["lvis_minival_ann"]
    lvis_local = json.loads(ann_path.read_text(encoding="utf-8"))
    id_to_im = {im["id"]: im for im in lvis_local["images"]}
    gt_by_img = _gt_index(lvis_local)

    images = [id_to_im[iid] for iid in episode.image_ids if iid in id_to_im]
    if max_images:
        images = images[:max_images]

    user_vocab = list(episode.vocab.cat_ids)
    vs = min(episode.vocab_size, len(lvis["categories"]))
    all_cat_ids = [c["id"] for c in lvis["categories"]]
    freq_cats = freq_sorted_cat_ids(lvis)
    episode_prompts = episode.vocab.prompts

    ap_list, fp_list, oov_list = [], [], []
    t0 = time.perf_counter()

    use_prompt_align = (
        method == "RV_full"
        and episode.noise == "synonym"
        and bool(rv_cfg.get("synonym_dual_prompt", False))
    )
    use_guard = method in ("VG_full_strict", "RV_recover", "RV_full")

    for im in images:
        iid = im["id"]
        image_rgb = _load_image(backend, im)
        if image_rgb is None:
            continue

        b0_preds = list(b0_preds_by_image.get(iid, [])) if b0_preds_by_image else []

        if method in ("VG_full_strict", "RV_recover", "RV_full"):
            routed_vocab = _route_vocab(
                method,
                episode,
                user_vocab,
                vs,
                image_rgb,
                iid,
                backend,
                lvis,
                router,
                recover,
                b0_preds,
                deployment_strict,
                freq_cats,
                all_cat_ids,
                rv_cfg,
            )

            if use_prompt_align:
                preds = align.predict(
                    backend,
                    image_rgb,
                    routed_vocab,
                    iid,
                    lvis,
                    episode_prompts,
                    use_dual=bool(rv_cfg.get("dual_prompt_fusion", True)),
                )
            else:
                preds = backend.predict(image_rgb, routed_vocab, iid, lvis)

            gt = gt_by_img.get(iid, {"boxes": [], "cat_ids": []})
            ap_list.append(
                episodic_ap_per_image(preds, gt["boxes"], gt["cat_ids"], user_vocab)
            )
            fp_list.append(fp_non_gt_rate(preds, user_vocab))

            if use_guard and b0_preds_by_image is not None:
                guarded = guard.guard(b0_preds, user_vocab, in_vocab_preds=preds)
                oov_list.append(oov_fp_rate(guarded, user_vocab))
            elif b0_preds_by_image is not None:
                oov_list.append(oov_fp_rate(b0_preds, user_vocab))

        elif method == "B5_subset":
            preds = backend.predict(image_rgb, user_vocab, iid, lvis)
            gt = gt_by_img.get(iid, {"boxes": [], "cat_ids": []})
            ap_list.append(episodic_ap_per_image(preds, gt["boxes"], gt["cat_ids"], user_vocab))
            fp_list.append(fp_non_gt_rate(preds, user_vocab))
            if b0_preds_by_image is not None:
                oov_list.append(oov_fp_rate(b0_preds, user_vocab))

        else:
            from vocabguard.infer_pipeline import run_vocabguard_episode

            vg_method = method
            if method == "VG_full":
                vg_method = "VG_full"
            return run_vocabguard_episode(
                episode,
                vg_method,
                lvis,
                backend,
                max_images=max_images,
                device=device,
                b0_preds_by_image=b0_preds_by_image,
                backbone=backbone,
            )

    elapsed = time.perf_counter() - t0
    return {
        "episode_id": episode.episode_id,
        "method": method,
        "backbone": backbone,
        "vocab_size": vs,
        "noise": episode.noise,
        "deployment_strict": deployment_strict,
        "n_images": len(ap_list),
        "EpisodicAP_mean": float(np.mean(ap_list)) if ap_list else 0.0,
        "FP_nonGT_mean": float(np.mean(fp_list)) if fp_list else 0.0,
        "OOV_FP_mean": float(np.mean(oov_list)) if oov_list else 0.0,
        "latency_ms_per_image": (elapsed / max(len(ap_list), 1)) * 1000,
    }
