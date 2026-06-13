#!/usr/bin/env python3
"""Generate nuScenes-OVDeploy episode JSON (OVDeploy Episode schema)."""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

ROOT_PAPER2 = Path(__file__).resolve().parents[3].parent / "论文2"
if str(ROOT_PAPER2) not in sys.path:
    sys.path.insert(0, str(ROOT_PAPER2))

from ovdeploy.episode import Episode, EpisodeVocab, save_episode  # noqa: E402
from ovdeploy.nuscenes.gt import NuScenesGT, sample_token_to_image_id  # noqa: E402
from ovdeploy.nuscenes.taxonomy import (  # noqa: E402
    load_pilot_config,
    load_taxonomy,
    resolve_class_map_path,
    resolve_nuscenes_root,
)
from ovdeploy.vocab import build_prompts_for_cat  # noqa: E402


def _resolve_episodes_out(cfg: dict, override: Path | None) -> Path:
    if override is not None:
        return override
    import sys as _sys

    key = "episodes_out_wsl" if _sys.platform != "win32" else "episodes_out"
    p = Path(cfg[key])
    if not p.is_absolute():
        p = ROOT_PAPER2 / p
    return p


def _build_vocab(
    gt_cat_ids: set[int],
    vocab_size: int,
    all_cat_ids: list[int],
    freq_order: list[int],
    rng: random.Random,
) -> list[int]:
    vocab = [c for c in freq_order if c in gt_cat_ids]
    for c in freq_order:
        if len(vocab) >= vocab_size:
            break
        if c not in vocab:
            vocab.append(c)
    while len(vocab) < vocab_size:
        c = rng.choice(all_cat_ids)
        if c not in vocab:
            vocab.append(c)
    return vocab[:vocab_size]


def generate_episodes(
    gt: NuScenesGT,
    taxonomy,
    out_dir: Path,
    vocab_size: int,
    seed: int,
    noise: str,
    split: str,
    frames_per_episode: int,
    frame_stride: int,
    max_episodes: int,
) -> list[Episode]:
    rng = random.Random(seed)
    img_cats = gt.image_gt_cat_ids()
    all_ids = taxonomy.all_cat_ids
    freq = [c for c, _ in Counter(c for s in img_cats.values() for c in s).most_common()]
    for c in all_ids:
        if c not in freq:
            freq.append(c)

    episodes: list[Episode] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    e_idx = 0

    for scene_name, rows in sorted(gt.cam_front_samples_by_scene().items()):
        start = 0
        while start + frames_per_episode <= len(rows) and e_idx < max_episodes:
            window = rows[start : start + frames_per_episode]
            sample_tokens = [w[0] for w in window]
            file_paths = [w[2] for w in window]
            image_ids = [sample_token_to_image_id(t) for t in sample_tokens]

            gt_set: set[int] = set()
            for iid in image_ids:
                gt_set |= img_cats.get(iid, set())
            if not gt_set:
                start += frame_stride
                continue

            vocab_ids = _build_vocab(gt_set, vocab_size, all_ids, freq, rng)
            if not (gt_set & set(vocab_ids)):
                start += frame_stride
                continue

            prompts = {
                str(cid): build_prompts_for_cat(
                    taxonomy.cat_id_to_name[cid],
                    taxonomy.cat_id_to_prompt[cid],
                )
                for cid in vocab_ids
            }

            eid = f"nuscenes_{scene_name}_v{vocab_size}_s{seed}_{noise}_e{e_idx:03d}"
            ep = Episode(
                episode_id=eid,
                image_ids=image_ids,
                vocab=EpisodeVocab(cat_ids=vocab_ids, prompts=prompts),
                vocab_size=len(vocab_ids),
                noise=noise,
                split=split,
                seed=seed + e_idx,
                meta={
                    "n_images": len(image_ids),
                    "scene": scene_name,
                    "sample_tokens": sample_tokens,
                    "file_paths": file_paths,
                    "camera": gt.camera,
                },
            )
            save_episode(ep, out_dir / f"{eid}.json")
            episodes.append(ep)
            e_idx += 1
            start += frame_stride

    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Build nuScenes OVDeploy episodes")
    parser.add_argument("--root", type=Path, default=None, help="nuScenes dataset root")
    parser.add_argument("--out", type=Path, default=None, help="Output episode directory")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_PAPER2 / "config" / "nuscenes_pilot.yaml",
    )
    parser.add_argument("--class-map", type=Path, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate paths only")
    args = parser.parse_args()

    cfg = load_pilot_config(args.config)
    nusc_root = args.root or resolve_nuscenes_root(cfg)
    out_dir = _resolve_episodes_out(cfg, args.out)
    class_map = args.class_map or resolve_class_map_path(cfg)
    vocab_size = args.vocab_size or cfg["vocab_sizes"][1]  # MVP |V|=10
    seed = args.seed if args.seed is not None else cfg["seeds"][0]
    max_ep = args.max_episodes or cfg.get("max_episodes", 100)

    if not nusc_root.is_dir():
        raise SystemExit(
            f"nuScenes root not found: {nusc_root}\n"
            "Download v1.0-mini from https://www.nuscenes.org/nuscenes and set --root or nuscenes_pilot.yaml"
        )

    taxonomy = load_taxonomy(class_map)
    if args.dry_run:
        print(f"OK: root={nusc_root}, out={out_dir}, |V|={vocab_size}, max={max_ep}")
        return

    gt = NuScenesGT(nusc_root, version=cfg.get("version", "v1.0-mini"), camera=cfg["camera"])
    sub = out_dir / f"dev_v{vocab_size}_s{seed}_{cfg.get('noise', 'none')}"
    eps = generate_episodes(
        gt,
        taxonomy,
        sub,
        vocab_size=vocab_size,
        seed=seed,
        noise=cfg.get("noise", "none"),
        split=cfg.get("split", "dev"),
        frames_per_episode=int(cfg.get("frames_per_episode", 10)),
        frame_stride=int(cfg.get("frame_stride", 5)),
        max_episodes=max_ep,
    )
    print(f"Wrote {len(eps)} episodes to {sub}")
    if len(eps) < max_ep:
        print(f"Note: only {len(eps)} episodes (mini split); reduce --max-episodes or use full nuScenes.")


if __name__ == "__main__":
    main()
