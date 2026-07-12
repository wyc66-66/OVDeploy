"""B0 full-vocab prediction cache for OOV-FP metric (per backbone)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cache_dir(backbone: str = "yolo") -> Path:
    key = (backbone or "yolo").lower()
    import sys

    if sys.platform != "win32":
        try:
            from ovdeploy.paths_util import load_paths

            cfg = load_paths()
            wsl_root = cfg.get("b0_cache_wsl")
            if wsl_root:
                native = Path(wsl_root).expanduser() / key
                native.mkdir(parents=True, exist_ok=True)
                return native
        except Exception:
            pass
    return ROOT / "data" / "b0_cache" / key


def cache_path(image_id: int, backbone: str = "yolo") -> Path:
    return cache_dir(backbone) / f"{image_id}.json"


def load_b0_preds(image_id: int, backbone: str = "yolo") -> list[dict] | None:
    p = cache_path(image_id, backbone)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    legacy_dir = ROOT / "data" / "b0_cache" / (backbone or "yolo").lower()
    legacy = legacy_dir / f"{image_id}.json"
    if legacy.is_file():
        return json.loads(legacy.read_text(encoding="utf-8"))
    if backbone in ("yolo", ""):
        legacy2 = ROOT / "data" / "b0_cache" / f"{image_id}.json"
        if legacy2.is_file():
            return json.loads(legacy2.read_text(encoding="utf-8"))
    return None


def save_b0_preds(image_id: int, preds: list[dict], backbone: str = "yolo") -> None:
    d = cache_dir(backbone)
    d.mkdir(parents=True, exist_ok=True)
    cache_path(image_id, backbone).write_text(json.dumps(preds), encoding="utf-8")


def ensure_b0_preds(
    image_ids: list[int],
    lvis: dict,
    device: str = "cuda:0",
    force: bool = False,
    backbone: str = "yolo",
) -> dict[int, list[dict]]:
    """Run B0_full once per image; return id -> preds."""
    import time

    import cv2

    from ovdeploy.backends.base import get_backend
    from ovdeploy.paths_util import cat_id_to_index, load_class_texts, load_paths

    out: dict[int, list[dict]] = {}
    missing = [
        iid for iid in image_ids if force or load_b0_preds(iid, backbone) is None
    ]
    n_total = len(image_ids)
    n_done = n_total - len(missing)

    for iid in image_ids:
        if iid not in missing:
            cached = load_b0_preds(iid, backbone)
            if cached is not None:
                out[iid] = cached

    if not missing:
        print(f"B0 cache {backbone}: {n_total}/{n_total} (all cached)", flush=True)
        return out

    import torch

    print(
        f"B0 cache torch: cuda_available={torch.cuda.is_available()} "
        f"device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}",
        flush=True,
    )

    cfg = load_paths()
    backend = get_backend(backbone, device=device)
    class_names, class_texts_raw = load_class_texts(cfg)
    cid2idx = cat_id_to_index(lvis)
    all_cat_ids = [c["id"] for c in lvis["categories"]]
    id_to_im = {im["id"]: im for im in lvis["images"]}

    if hasattr(backend, "warm_full_vocab"):
        backend.warm_full_vocab(class_names, class_texts_raw, all_cat_ids, cid2idx)
    elif hasattr(backend, "warm_gdino_full_vocab"):
        backend.warm_gdino_full_vocab(class_names, class_texts_raw, all_cat_ids, cid2idx)

    print(
        f"B0 cache {backbone}: {n_done}/{n_total} cached, "
        f"inferring {len(missing)} on {device}, cache_dir={cache_dir(backbone)}",
        flush=True,
    )

    from concurrent.futures import ThreadPoolExecutor

    missing_ims = [id_to_im[iid] for iid in missing if iid in id_to_im]

    def _load_image(im: dict):
        iid = im["id"]
        path = backend.image_path(im["file_name"])
        image = cv2.imread(str(path))
        if image is None:
            return iid, None
        return iid, image[:, :, [2, 1, 0]]

    with ThreadPoolExecutor(max_workers=4) as pool, ThreadPoolExecutor(max_workers=2) as save_pool:
        save_futures: list = []
        if missing_ims:
            next_future = pool.submit(_load_image, missing_ims[0])
        else:
            next_future = None

        for idx, im in enumerate(missing_ims, start=1):
            iid, image_rgb = next_future.result() if next_future else (im["id"], None)
            if idx < len(missing_ims):
                next_future = pool.submit(_load_image, missing_ims[idx])
            else:
                next_future = None

            if image_rgb is None:
                continue

            t0 = time.perf_counter()
            preds = backend.predict(
                image_rgb,
                [],
                all_cat_ids,
                iid,
                class_names=class_names,
                class_texts_raw=class_texts_raw,
                cid2idx=cid2idx,
            )
            elapsed = time.perf_counter() - t0
            save_futures.append(save_pool.submit(save_b0_preds, iid, preds, backbone))
            out[iid] = preds
            n_chunks = getattr(backend, "_last_gdino_n_chunks", 0)
            if idx <= 5 or idx % 5 == 0 or idx == len(missing_ims):
                print(
                    f"B0 cache {backbone}: {n_done + idx}/{n_total} "
                    f"(infer {idx}/{len(missing_ims)}, {elapsed:.1f}s, n_chunks={n_chunks})",
                    flush=True,
                )

        for fut in save_futures:
            fut.result()

    print(f"B0 cache {backbone}: {len(out)}/{n_total} ready", flush=True)
    return out


def ensure_b0_preds_legacy_episode(
    image_ids: list[int],
    lvis: dict,
    device: str = "cuda:0",
    force: bool = False,
    backbone: str = "yolo",
) -> dict[int, list[dict]]:
    """Legacy episode-based path (kept for parity tests)."""
    from ovdeploy.episode import Episode, EpisodeVocab
    from ovdeploy.infer import run_episode_infer

    out: dict[int, list[dict]] = {}
    missing = [
        iid for iid in image_ids if force or load_b0_preds(iid, backbone) is None
    ]
    if not missing:
        for iid in image_ids:
            cached = load_b0_preds(iid, backbone)
            if cached is not None:
                out[iid] = cached
        return out

    ep = Episode(
        episode_id=f"b0_cache_{backbone}_batch",
        image_ids=missing,
        vocab=EpisodeVocab(cat_ids=[]),
        vocab_size=1203,
        noise="none",
        split="cache",
    )
    result = run_episode_infer(
        ep, "B0_full", lvis, device=device, save_records=True, backbone=backbone
    )
    for rec in result.get("records", []):
        iid = rec["image_id"]
        preds = rec.get("predictions", [])
        save_b0_preds(iid, preds, backbone)
        out[iid] = preds

    for iid in image_ids:
        if iid not in out:
            cached = load_b0_preds(iid, backbone)
            if cached is not None:
                out[iid] = cached
    return out
