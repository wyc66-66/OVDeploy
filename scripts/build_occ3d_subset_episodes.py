#!/usr/bin/env python3
"""Build Occ3D semantic subset episodes (fixed |V| per subset, CAM_FRONT)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT_PAPER = Path(__file__).resolve().parents[3] / "submission-a"
if str(ROOT_PAPER) not in sys.path:
    sys.path.insert(0, str(ROOT_PAPER))

from ovdeploy.episode import Episode, EpisodeVocab, save_episode  # noqa: E402
from ovdeploy.nuscenes.gt import NuScenesGT, sample_token_to_image_id  # noqa: E402
from ovdeploy.nuscenes.taxonomy import (  # noqa: E402
    load_taxonomy,
    resolve_nuscenes_root,
)
from ovdeploy.vocab import build_prompts_for_cat  # noqa: E402

from _pilot_layout import pilot_layout  # noqa: E402

_REPO, _CFG_DIR, _DATA_DIR = pilot_layout(Path(__file__))
PILOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = _CFG_DIR / "occ3d_semantic_subsets.yaml"
DEFAULT_CLASS_MAP = _CFG_DIR / "nuscenes_class_map.yaml"


def _scene_has_subset_gt(
    gt_index: NuScenesGT,
    sample_tokens: list[str],
    subset_ids: set[int],
) -> bool:
    for t in sample_tokens:
        iid = sample_token_to_image_id(t)
        gt_cats = set(gt_index.get_gt(iid)["cat_ids"])
        if gt_cats & subset_ids:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=PILOT / "data" / "episodes_occ3d_subset",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    nusc_root = args.root
    if nusc_root is None:
        try:
            from ovdeploy.nuscenes.taxonomy import load_pilot_config

            pilot_cfg = load_pilot_config(_REPO / "config" / "nuscenes_pilot.yaml")
            nusc_root = resolve_nuscenes_root(pilot_cfg)
        except Exception:
            nusc_root = Path("d:/data/nuscenes")

    camera = cfg.get("camera", "CAM_FRONT")
    frames = int(cfg.get("frames_per_episode", 10))
    stride = int(cfg.get("frame_stride", 5))
    seed = int(cfg.get("seed", 42))
    scene_pool = cfg.get("scene_pool", [])
    taxonomy = load_taxonomy(args.class_map)

    gt = NuScenesGT(nusc_root, version="v1.0-mini", camera=camera, taxonomy=taxonomy)
    scenes = gt.cam_front_samples_by_scene()

    total_written = 0
    for subset in cfg.get("subsets", []):
        subset_id = subset["id"]
        vocab_ids = sorted(set(int(c) for c in subset["cat_ids"]))
        subset_set = set(vocab_ids)
        require_overlap = bool(subset.get("require_gt_overlap", False))
        out_dir = args.out / subset_id
        out_dir.mkdir(parents=True, exist_ok=True)

        prompts = {
            str(cid): build_prompts_for_cat(
                taxonomy.cat_id_to_name[cid],
                taxonomy.cat_id_to_prompt[cid],
            )
            for cid in vocab_ids
        }

        written = 0
        for scene_name in scene_pool:
            if scene_name not in scenes:
                print(f"Skip {subset_id}/{scene_name}: not in mini")
                continue
            rows = scenes[scene_name]
            if len(rows) < frames:
                continue

            window = rows[:frames]
            sample_tokens = [w[0] for w in window]
            if require_overlap and not _scene_has_subset_gt(gt, sample_tokens, subset_set):
                print(f"Skip {subset_id}/{scene_name}: no GT in subset")
                continue

            file_paths_meta = [Path(w[2]).name for w in window]
            image_ids = [sample_token_to_image_id(t) for t in sample_tokens]

            scene_tag = scene_name.replace("scene-", "")
            eid = f"occ3d_{subset_id}_{scene_tag}_s{seed}_none"
            ep = Episode(
                episode_id=eid,
                image_ids=image_ids,
                vocab=EpisodeVocab(cat_ids=vocab_ids, prompts=prompts),
                vocab_size=len(vocab_ids),
                noise="none",
                split="dev",
                seed=seed,
                meta={
                    "n_images": len(image_ids),
                    "scene": scene_name,
                    "sample_tokens": sample_tokens,
                    "file_paths": file_paths_meta,
                    "camera": camera,
                    "occ3d_subset": subset_id,
                    "occ3d_description": subset.get("description", ""),
                    "prototype": "occ3d_semantic_subset",
                },
            )
            save_episode(ep, out_dir / f"{eid}.json")
            written += 1

        print(f"Subset {subset_id}: wrote {written} episodes (|V|={len(vocab_ids)})")
        total_written += written

    print(f"Total episodes: {total_written} under {args.out}")


if __name__ == "__main__":
    main()
