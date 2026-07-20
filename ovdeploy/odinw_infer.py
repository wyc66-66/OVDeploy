"""YOLO-World inference on native ODinW COCO domains."""
from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from ovdeploy.backends.base import get_backend
from ovdeploy.episode import Episode
from ovdeploy.odinw_loader import (
    all_category_ids,
    class_texts_for_ids,
    gt_by_image,
    list_images,
    sample_episode_vocab,
)
from ovdeploy.scenario.coco import load_scenario_coco, load_scenario_meta, scenario_image_path
from ovdeploy.odinw_metrics import episodic_ap_per_image, oov_fp_rate


def _lvis_all_cat_ids(lvis: dict) -> list[int]:
    return [c["id"] for c in lvis["categories"]]


def _predict_domain(
    backend,
    slug: str,
    image_rgb: np.ndarray,
    vocab_ids: list[int],
    coco: dict,
    meta: dict | None = None,
    lvis_prompts: tuple[list[str], list, dict[int, int]] | None = None,
) -> list[dict]:
    if meta is None:
        meta = load_scenario_meta(slug)
    if lvis_prompts is not None:
        names, texts, cid2idx = lvis_prompts
        id2idx = {cid: i for i, cid in enumerate(vocab_ids)}
        return backend.predict(
            image_rgb,
            names,
            vocab_ids,
            image_id=0,
            class_names=names,
            class_texts_raw=texts,
            cid2idx=id2idx,
        )
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


def _resolve_b0_vocab(
    baseline: str,
    all_ids: list[int],
    episode_vocab: list[int],
    b0_prompt_mode: str,
    lvis: dict | None,
) -> tuple[list[int], tuple | None]:
    if baseline != "B0_full":
        return episode_vocab, None
    if b0_prompt_mode == "lvis_full" and lvis is not None:
        from ovdeploy.paths_util import cat_id_to_index, load_class_texts, load_paths

        cfg = load_paths()
        names, texts = load_class_texts(cfg)
        all_lvis = _lvis_all_cat_ids(lvis)
        cid2idx = cat_id_to_index(lvis)
        names_sub, texts_sub = [], []
        for cid in all_lvis:
            idx = cid2idx.get(cid)
            if idx is not None and idx < len(names):
                names_sub.append(names[idx])
                texts_sub.append(texts[idx] if isinstance(texts[idx], list) else [texts[idx]])
            else:
                names_sub.append(str(cid))
                texts_sub.append([str(cid)])
        return all_lvis, (names_sub, texts_sub, cid2idx)
    return all_ids, None


def run_odinw_episode(
    slug: str,
    baseline: str,
    vocab_size: int,
    seed: int = 42,
    max_images: int = 100,
    device: str = "cuda:0",
    b0_preds_by_image: dict[int, list[dict]] | None = None,
    backbone: str = "yolo",
    b0_prompt_mode: str = "domain_native",
) -> dict[str, Any]:
    return run_odinw_episode_lvis(
        slug, baseline, vocab_size, seed, max_images, device,
        b0_preds_by_image, backbone, b0_prompt_mode,
    )


def run_odinw_episode_lvis(
    slug: str,
    baseline: str,
    vocab_size: int,
    seed: int = 42,
    max_images: int = 100,
    device: str = "cuda:0",
    b0_preds_by_image: dict[int, list[dict]] | None = None,
    backbone: str = "yolo",
    b0_prompt_mode: str = "domain_native",
) -> dict[str, Any]:
    coco = load_scenario_coco(slug)
    from ovdeploy.paths_util import load_lvis_minival, load_paths

    meta = load_scenario_meta(slug)
    domain_classes = meta.get("classes", [])
    all_ids = all_category_ids(coco)
    episode_vocab = sample_episode_vocab(coco, domain_classes, vocab_size, seed)
    gt_map = gt_by_image(coco)
    images = list_images(coco, max_images)

    lvis = load_lvis_minival(load_paths()) if b0_prompt_mode == "lvis_full" else None
    infer_vocab, lvis_prompts = _resolve_b0_vocab(
        baseline, all_ids, episode_vocab, b0_prompt_mode, lvis
    )
    if baseline in ("B5_subset", "M1_adapter"):
        infer_vocab = episode_vocab
        lvis_prompts = None

    if b0_preds_by_image is None and baseline == "B0_full":
        b0_preds_by_image = ensure_odinw_b0_cache(
            slug,
            [im["id"] for im in images],
            coco,
            device=device,
            backbone=backbone,
            b0_prompt_mode=b0_prompt_mode,
        )

    backend = get_backend(backbone, device=device)
    ap_list = []
    oov_list = []
    t0 = time.perf_counter()

    for im in images:
        iid = im["id"]
        path = scenario_image_path(slug, im["file_name"])
        img = cv2.imread(str(path))
        if img is None:
            continue
        rgb = img[:, :, [2, 1, 0]]
        preds = _predict_domain(
            backend, slug, rgb, infer_vocab, coco, meta, lvis_prompts
        )
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
        "b0_prompt_mode": b0_prompt_mode,
        "backbone": backbone,
    }


def run_odinw_episode_from_episode_json(
    episode: Episode,
    slug: str,
    baseline: str,
    device: str = "cuda:0",
    backbone: str = "yolo",
    b0_prompt_mode: str = "domain_native",
) -> dict[str, Any]:
    return run_odinw_episode_lvis(
        slug,
        baseline,
        episode.vocab_size,
        seed=episode.seed,
        max_images=len(episode.image_ids),
        device=device,
        backbone=backbone,
        b0_prompt_mode=b0_prompt_mode,
    )


def ensure_odinw_b0_cache(
    slug: str,
    image_ids: list[int],
    coco: dict,
    device: str = "cuda:0",
    backbone: str = "yolo",
    force: bool = False,
    b0_prompt_mode: str = "domain_native",
) -> dict[int, list[dict]]:
    import json
    from pathlib import Path

    from ovdeploy.paths_util import load_lvis_minival, load_paths

    mode_tag = "lvisfull" if b0_prompt_mode == "lvis_full" else "native"
    cache_dir = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "b0_cache"
        / "odinw"
        / slug
        / f"{backbone}_{mode_tag}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[int, list[dict]] = {}
    all_ids = all_category_ids(coco)
    lvis = load_lvis_minival(load_paths()) if b0_prompt_mode == "lvis_full" else None
    infer_vocab, lvis_prompts = _resolve_b0_vocab(
        "B0_full", all_ids, all_ids[:10], b0_prompt_mode, lvis
    )
    backend = get_backend(backbone, device=device)
    id_to_im = {im["id"]: im for im in coco["images"]}
    from ovdeploy.scenario.coco import load_scenario_meta

    meta = load_scenario_meta(slug)

    for iid in image_ids:
        cp = cache_dir / f"{iid}.json"
        if cp.is_file() and not force:
            out[iid] = json.loads(cp.read_text(encoding="utf-8"))
            continue
        im = id_to_im.get(iid)
        if not im:
            continue
        path = scenario_image_path(slug, im["file_name"])
        img = cv2.imread(str(path))
        if img is None:
            continue
        rgb = img[:, :, [2, 1, 0]]
        preds = _predict_domain(
            backend, slug, rgb, infer_vocab, coco, meta, lvis_prompts
        )
        cp.write_text(json.dumps(preds), encoding="utf-8")
        out[iid] = preds

    for iid in image_ids:
        if iid not in out:
            cp = cache_dir / f"{iid}.json"
            if cp.is_file():
                out[iid] = json.loads(cp.read_text(encoding="utf-8"))
    return out
