"""nuScenes episodic inference (B0_full / B5_subset)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2

from ovdeploy.backends.yolo_world import YoloWorldBackend
from ovdeploy.episode import Episode
from ovdeploy.metrics import episodic_ap_per_image_v2, oov_fp_rate
from ovdeploy.nuscenes.gt import NuScenesGT
from ovdeploy.nuscenes.taxonomy import NuScenesTaxonomy, load_taxonomy
from ovdeploy.paths_util import load_paths


def _vocab_for_baseline(baseline: str, episode: Episode, all_cat_ids: list[int]) -> list[int]:
    if baseline == "B0_full":
        return list(all_cat_ids)
    if baseline == "B5_subset":
        return list(episode.vocab.cat_ids)
    raise ValueError(f"Unsupported baseline: {baseline}")


def run_nuscenes_episode_infer(
    episode: Episode,
    baseline: str,
    gt_index: NuScenesGT,
    taxonomy: NuScenesTaxonomy,
    max_images: int = 0,
    device: str = "cuda:0",
    save_records: bool = False,
    b0_preds_by_image: dict[int, list[dict]] | None = None,
    episode_vocab_for_oov: list[int] | None = None,
    oov_score_thr: float = 0.05,
) -> dict[str, Any]:
    load_paths()
    backend = YoloWorldBackend(device=device)
    all_cat_ids = taxonomy.all_cat_ids
    vs = min(episode.vocab_size, len(all_cat_ids))

    image_ids = list(episode.image_ids)
    if max_images:
        image_ids = image_ids[:max_images]

    meta_paths = episode.meta.get("file_paths", [])
    meta_tokens = episode.meta.get("sample_tokens", [])
    id_to_path = dict(zip(episode.image_ids, meta_paths)) if meta_paths else {}
    if not id_to_path:
        id_to_path = gt_index.image_id_to_path

    global_vocab = _vocab_for_baseline(baseline, episode, all_cat_ids)

    ap_list: list[float] = []
    oov_list: list[float] = []
    records: list[dict] = []
    text_ms = 0.0
    t0 = time.perf_counter()

    for idx, iid in enumerate(image_ids):
        path_str = id_to_path.get(iid) or gt_index.image_id_to_path.get(iid)
        if not path_str:
            continue
        image = cv2.imread(path_str)
        if image is None:
            continue
        image_rgb = image[:, :, [2, 1, 0]]
        vocab_ids = global_vocab

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
                class_names=taxonomy.class_names,
                class_texts_raw=taxonomy.class_texts_raw,
                cid2idx=taxonomy.cid2idx,
            )
        text_ms += (time.perf_counter() - t_text) * 1000

        gt = gt_index.get_gt(iid)
        ap_list.append(
            episodic_ap_per_image_v2(preds, gt["boxes"], gt["cat_ids"], vocab_ids)
        )
        if b0_preds_by_image is not None and episode_vocab_for_oov is not None:
            b0p = b0_preds_by_image.get(iid, [])
            oov_list.append(oov_fp_rate(b0p, episode_vocab_for_oov, score_thr=oov_score_thr))

        rec: dict[str, Any] = {"image_id": iid, "n_preds": len(preds)}
        if meta_tokens and idx < len(meta_tokens):
            rec["sample_token"] = meta_tokens[idx]
        if save_records:
            rec["predictions"] = preds
        records.append(rec)

    elapsed = time.perf_counter() - t0
    n = max(len(ap_list), 1)
    return {
        "episode_id": episode.episode_id,
        "baseline": baseline,
        "vocab_size": vs,
        "EpisodicAP_mean": sum(ap_list) / len(ap_list) if ap_list else 0.0,
        "OOV_FP_mean": sum(oov_list) / len(oov_list) if oov_list else float("nan"),
        "text_ms_per_image": text_ms / n,
        "elapsed_s": elapsed,
        "n_images": len(ap_list),
        "records": records,
    }


def nuscenes_b0_cache_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "b0_cache" / "nuscenes_yolo"


def load_nuscenes_b0_preds(image_id: int) -> list[dict] | None:
    p = nuscenes_b0_cache_dir() / f"{image_id}.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save_nuscenes_b0_preds(image_id: int, preds: list[dict]) -> None:
    d = nuscenes_b0_cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{image_id}.json").write_text(json.dumps(preds), encoding="utf-8")


def ensure_nuscenes_b0_preds(
    image_ids: list[int],
    gt_index: NuScenesGT,
    taxonomy: NuScenesTaxonomy,
    device: str = "cuda:0",
    force: bool = False,
) -> dict[int, list[dict]]:
    from ovdeploy.episode import Episode, EpisodeVocab

    out: dict[int, list[dict]] = {}
    missing = [iid for iid in image_ids if force or load_nuscenes_b0_preds(iid) is None]

    for start in range(0, len(missing), 10):
        chunk = missing[start : start + 10]
        if not chunk:
            continue
        paths = [gt_index.image_id_to_path.get(i) for i in chunk]
        tokens = [gt_index.image_id_to_sample_token.get(i, "") for i in chunk]
        ep = Episode(
            episode_id=f"nuscenes_b0_cache_{start}",
            image_ids=chunk,
            vocab=EpisodeVocab(cat_ids=[]),
            vocab_size=taxonomy.num_classes,
            noise="none",
            split="cache",
            meta={"file_paths": paths, "sample_tokens": tokens},
        )
        result = run_nuscenes_episode_infer(
            ep,
            "B0_full",
            gt_index,
            taxonomy,
            device=device,
            save_records=True,
        )
        for rec in result.get("records", []):
            iid = rec["image_id"]
            preds = rec.get("predictions", [])
            save_nuscenes_b0_preds(iid, preds)
            out[iid] = preds

    for iid in image_ids:
        if iid not in out:
            cached = load_nuscenes_b0_preds(iid)
            if cached is not None:
                out[iid] = cached
    return out
