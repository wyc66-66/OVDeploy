#!/usr/bin/env python3
"""Build DriveVLM-style scene vocabulary episodes from YAML config."""
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
    resolve_class_map_path,
    resolve_nuscenes_root,
)
from ovdeploy.vocab import build_prompts_for_cat  # noqa: E402

PILOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = PILOT / "config" / "drivevlm_vocab_episodes.yaml"
DEFAULT_CLASS_MAP = PILOT / "config" / "nuscenes_class_map.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=PILOT / "data" / "episodes_drivevlm_vocab" / "dev",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    nusc_root = args.root
    if nusc_root is None:
        try:
            from ovdeploy.nuscenes.taxonomy import load_pilot_config

            pilot_cfg = load_pilot_config(ROOT_PAPER / "config" / "nuscenes_pilot.yaml")
            nusc_root = resolve_nuscenes_root(pilot_cfg)
        except Exception:
            nusc_root = Path("d:/data/nuscenes")

    camera = cfg.get("camera", "CAM_FRONT")
    frames = int(cfg.get("frames_per_episode", 10))
    stride = int(cfg.get("frame_stride", 5))
    seed = int(cfg.get("seed", 42))
    taxonomy = load_taxonomy(args.class_map)

    gt = NuScenesGT(nusc_root, version="v1.0-mini", camera=camera, taxonomy=taxonomy)
    scenes = gt.cam_front_samples_by_scene()

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0

    for entry in cfg.get("episodes", []):
        scene_name = entry["scene"]
        if scene_name not in scenes:
            print(f"Skip {entry['id']}: scene {scene_name} not in mini split")
            continue
        rows = scenes[scene_name]
        if len(rows) < frames:
            print(f"Skip {entry['id']}: scene too short")
            continue

        window = rows[:frames]
        sample_tokens = [w[0] for w in window]
        file_paths = [w[2] for w in window]
        # Store filenames only in outreach JSON (no local absolute paths).
        file_paths_meta = [Path(p).name for p in file_paths]
        image_ids = [sample_token_to_image_id(t) for t in sample_tokens]
        vocab_ids = sorted(set(int(c) for c in entry["cat_ids"]))

        prompts = {
            str(cid): build_prompts_for_cat(
                taxonomy.cat_id_to_name[cid],
                taxonomy.cat_id_to_prompt[cid],
            )
            for cid in vocab_ids
        }

        eid = f"drivevlm_{entry['id']}_s{seed}_none"
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
                "drivevlm_prompt": entry.get("drivevlm_prompt", ""),
                "prototype": "drivevlm_vocab_smoke",
            },
        )
        save_episode(ep, args.out / f"{eid}.json")
        written += 1
        print(f"OK {eid} |V|={len(vocab_ids)} scene={scene_name}")

    print(f"Wrote {written} episodes to {args.out}")


if __name__ == "__main__":
    main()
