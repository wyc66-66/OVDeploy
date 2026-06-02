"""B0 full-vocab prediction cache for OOV-FP metric (per backbone)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cache_dir(backbone: str = "yolo") -> Path:
    key = (backbone or "yolo").lower()
    return ROOT / "data" / "b0_cache" / key


def cache_path(image_id: int, backbone: str = "yolo") -> Path:
    return cache_dir(backbone) / f"{image_id}.json"


def load_b0_preds(image_id: int, backbone: str = "yolo") -> list[dict] | None:
    p = cache_path(image_id, backbone)
    if not p.is_file():
        legacy = ROOT / "data" / "b0_cache" / f"{image_id}.json"
        if backbone in ("yolo", "") and legacy.is_file():
            return json.loads(legacy.read_text(encoding="utf-8"))
        return None
    return json.loads(p.read_text(encoding="utf-8"))


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
    from ovdeploy.episode import Episode, EpisodeVocab
    from ovdeploy.infer import run_episode_infer

    out: dict[int, list[dict]] = {}
    missing = [
        iid for iid in image_ids if force or load_b0_preds(iid, backbone) is None
    ]
    chunk_size = 25
    for start in range(0, len(missing), chunk_size):
        chunk = missing[start : start + chunk_size]
        if not chunk:
            continue
        ep = Episode(
            episode_id=f"b0_cache_{backbone}_{start}",
            image_ids=chunk,
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
