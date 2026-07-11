"""Frozen detector inference with episode vocabulary subset."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ovdeploy.backends.base import get_backend
from ovdeploy.baselines import vocab_for_baseline
from ovdeploy.clip_vocab import clip_topk_cat_ids
from ovdeploy.episode import Episode
from ovdeploy.generator import image_gt_cat_ids, select_vocab_for_baseline
from ovdeploy.metrics import episodic_ap_per_image, fp_non_gt_rate, oov_fp_rate
from ovdeploy.paths_util import cat_id_to_index, load_class_texts, load_paths
from ovdeploy.vocab import freq_sorted_cat_ids


def _prompts_map(lvis: dict, class_texts_raw: list) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for i, c in enumerate(sorted(lvis["categories"], key=lambda x: x["id"])):
        if i < len(class_texts_raw):
            t = class_texts_raw[i]
            out[c["id"]] = t if isinstance(t, list) else [str(t)]
        else:
            out[c["id"]] = [c["name"]]
    return out


def _vocab_for_image(
    baseline: str,
    episode: Episode,
    iid: int,
    img_cats: dict,
    all_cat_ids: list[int],
    freq_cats: list[int],
    prompts_map: dict[int, list[str]],
    image_rgb: np.ndarray,
    vs: int,
) -> list[int]:
    import random

    rng = random.Random(episode.seed + iid)
    if baseline == "B0_full":
        return list(all_cat_ids)
    if baseline in ("B5_subset", "M1_adapter"):
        return list(episode.vocab.cat_ids)
    if baseline == "B4_clip":
        k = min(vs, len(all_cat_ids))
        return clip_topk_cat_ids(
            image_rgb, all_cat_ids, prompts_map, k, freq_cats, seed=episode.seed + iid
        )
    bl_map = {
        "B1_oracle": "B1_oracle",
        "B2_freq": "B2_freq",
        "B3_random": "B3_random",
    }
    if baseline in bl_map:
        return select_vocab_for_baseline(
            bl_map[baseline],
            [iid],
            vs,
            img_cats,
            all_cat_ids,
            freq_cats,
            rng,
            oracle_delta=3,
        )
    return vocab_for_baseline(
        baseline, episode, img_cats, all_cat_ids, freq_cats, oracle_delta=3
    )


def run_episode_infer(
    episode: Episode,
    baseline: str,
    lvis: dict,
    max_images: int = 0,
    adapter_bias: np.ndarray | None = None,
    device: str = "cuda:0",
    save_records: bool = False,
    b0_preds_by_image: dict[int, list[dict]] | None = None,
    episode_vocab_for_oov: list[int] | None = None,
    backbone: str = "yolo",
) -> dict[str, Any]:
    import cv2

    cfg = load_paths()
    yolo: Path = cfg["_yolo"]
    class_names, class_texts_raw = load_class_texts(cfg)
    cid2idx = cat_id_to_index(lvis)
    all_cat_ids = [c["id"] for c in lvis["categories"]]
    freq_cats = freq_sorted_cat_ids(lvis)
    img_cats = image_gt_cat_ids(lvis)
    prompts_map = _prompts_map(lvis, class_texts_raw)
    vs = min(episode.vocab_size, len(all_cat_ids))

    backend = get_backend(backbone, device=device)
    ann_path = yolo / cfg["data"]["lvis_minival_ann"]
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

    per_image = baseline in ("B4_clip", "B1_oracle", "B3_random")
    global_vocab = None
    if not per_image:
        global_vocab = _vocab_for_image(
            baseline,
            episode,
            episode.image_ids[0] if episode.image_ids else 0,
            img_cats,
            all_cat_ids,
            freq_cats,
            prompts_map,
            np.zeros((64, 64, 3), dtype=np.uint8),
            vs,
        )

    records = []
    ap_list = []
    fp_list = []
    oov_list = []
    t0 = time.perf_counter()
    text_ms = 0.0

    from concurrent.futures import ThreadPoolExecutor

    def _load_image(im: dict) -> tuple[int, np.ndarray | None]:
        iid = im["id"]
        path = backend.image_path(im["file_name"])
        image = cv2.imread(str(path))
        if image is None:
            return iid, None
        return iid, image[:, :, [2, 1, 0]]

    with ThreadPoolExecutor(max_workers=2) as pool:
        if images:
            next_future = pool.submit(_load_image, images[0])
        else:
            next_future = None

        for idx, im in enumerate(images):
            iid, image_rgb = next_future.result() if next_future else (im["id"], None)
            if idx + 1 < len(images):
                next_future = pool.submit(_load_image, images[idx + 1])
            else:
                next_future = None

            if image_rgb is None:
                continue

            vocab_ids = (
                _vocab_for_image(
                    baseline,
                    episode,
                    iid,
                    img_cats,
                    all_cat_ids,
                    freq_cats,
                    prompts_map,
                    image_rgb,
                    vs,
                )
                if per_image
                else global_vocab or list(episode.vocab.cat_ids)
            )

            t_text = time.perf_counter()
            if (
                baseline == "B0_full"
                and b0_preds_by_image is not None
                and not save_records
                and iid in b0_preds_by_image
            ):
                preds = list(b0_preds_by_image[iid])
            else:
                preds = backend.predict(
                    image_rgb,
                    [],
                    vocab_ids,
                    iid,
                    class_names=class_names,
                    class_texts_raw=class_texts_raw,
                    cid2idx=cid2idx,
                )
            text_ms += (time.perf_counter() - t_text) * 1000

            if adapter_bias is not None:
                for p in preds:
                    li = p.get("label_idx", 0)
                    if li < len(adapter_bias):
                        p["score"] = p["score"] + float(adapter_bias[li])

            gt = gt_by_img.get(iid, {"boxes": [], "cat_ids": []})
            ap_list.append(
                episodic_ap_per_image(preds, gt["boxes"], gt["cat_ids"], vocab_ids)
            )
            fp_list.append(fp_non_gt_rate(preds, vocab_ids))
            if b0_preds_by_image is not None and episode_vocab_for_oov is not None:
                b0p = b0_preds_by_image.get(iid, [])
                oov_list.append(oov_fp_rate(b0p, episode_vocab_for_oov))
            rec = {"image_id": iid, "n_preds": len(preds)}
            if save_records:
                rec["predictions"] = preds
            records.append(rec)

    elapsed = time.perf_counter() - t0
    return {
        "episode_id": episode.episode_id,
        "baseline": baseline,
        "backbone": backbone,
        "vocab_size": vs,
        "n_images": len(records),
        "EpisodicAP_mean": float(np.mean(ap_list)) if ap_list else 0.0,
        "FP_nonGT_mean": float(np.mean(fp_list)) if fp_list else 0.0,
        "OOV_FP_mean": float(np.mean(oov_list)) if oov_list else 0.0,
        "latency_ms_per_image": (elapsed / max(len(records), 1)) * 1000,
        "text_ms_per_image": text_ms / max(len(records), 1),
        "records": records if save_records else [],
    }


def smoke_two_images(episode_path: Path, baseline: str = "B5_subset", backbone: str = "yolo") -> dict:
    from ovdeploy.episode import load_episode
    from ovdeploy.paths_util import load_lvis_minival

    ep = load_episode(episode_path)
    lvis = load_lvis_minival()
    return run_episode_infer(ep, baseline, lvis, max_images=2, backbone=backbone)
