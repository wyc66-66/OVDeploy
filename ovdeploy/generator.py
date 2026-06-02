"""Generate OVDeploy episodes from LVIS annotations."""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from ovdeploy.episode import Episode, EpisodeVocab, save_episode
from ovdeploy.paths_util import load_class_texts, load_lvis_minival
from ovdeploy.vocab import apply_noise, freq_sorted_cat_ids


def image_gt_cat_ids(lvis: dict) -> dict[int, set[int]]:
    out: dict[int, set[int]] = defaultdict(set)
    for ann in lvis["annotations"]:
        out[ann["image_id"]].add(ann["category_id"])
    return {k: set(v) for k, v in out.items()}


def build_episode(
    episode_id: str,
    image_ids: list[int],
    vocab_cat_ids: list[int],
    lvis: dict,
    class_texts_raw: list,
    noise: str,
    split: str,
    seed: int,
) -> Episode:
    rng = random.Random(seed)
    cat_id_to_name = {c["id"]: c["name"] for c in lvis["categories"]}
    cat_id_to_text = {}
    for i, c in enumerate(sorted(lvis["categories"], key=lambda x: x["id"])):
        if i < len(class_texts_raw):
            cat_id_to_text[c["id"]] = class_texts_raw[i]

    v_ids, prompts = apply_noise(vocab_cat_ids, cat_id_to_name, cat_id_to_text, noise, rng)
    return Episode(
        episode_id=episode_id,
        image_ids=image_ids,
        vocab=EpisodeVocab(cat_ids=v_ids, prompts=prompts),
        vocab_size=len(v_ids),
        noise=noise,
        split=split,
        seed=seed,
        meta={"n_images": len(image_ids)},
    )


def select_vocab_for_baseline(
    baseline: str,
    image_ids: list[int],
    vocab_size: int,
    img_cats: dict[int, set[int]],
    all_cat_ids: list[int],
    freq_cats: list[int],
    rng: random.Random,
    oracle_delta: int = 3,
) -> list[int]:
    if vocab_size >= len(all_cat_ids):
        return list(all_cat_ids)

    if baseline in ("B0_full", "B5_subset", "M1_adapter"):
        if baseline == "B0_full":
            return list(all_cat_ids)
        # B5/M1 use episode-specific vocab from generator caller
        pass

    if baseline == "B1_oracle":
        gt = set()
        for iid in image_ids:
            gt |= img_cats.get(iid, set())
        extra = [c for c in freq_cats if c not in gt][: max(0, vocab_size - len(gt))]
        vocab = list(gt) + extra[: oracle_delta]
        if len(vocab) < vocab_size:
            for c in freq_cats:
                if c not in vocab:
                    vocab.append(c)
                if len(vocab) >= vocab_size:
                    break
        return vocab[:vocab_size]

    if baseline == "B2_freq":
        return freq_cats[:vocab_size]

    if baseline == "B3_random":
        pool = list(all_cat_ids)
        rng.shuffle(pool)
        return pool[:vocab_size]

    return freq_cats[:vocab_size]


def generate_episode_batch(
    out_dir: Path,
    split: str,
    image_pool: list[int],
    vocab_size: int,
    seed: int,
    n_episodes: int,
    noise: str,
    lvis: dict,
    class_texts_raw: list,
    n_images_per_episode: int = 10,
) -> list[Episode]:
    rng = random.Random(seed)
    img_cats = image_gt_cat_ids(lvis)
    all_ids = [c["id"] for c in lvis["categories"]]
    freq_cats = freq_sorted_cat_ids(lvis)
    episodes: list[Episode] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_episodes):
        rng.shuffle(image_pool)
        ep_images = image_pool[: min(n_images_per_episode, len(image_pool))]
        gt = set()
        for iid in ep_images:
            gt |= img_cats.get(iid, set())
        vocab = list(gt)
        for c in freq_cats:
            if len(vocab) >= vocab_size:
                break
            if c not in vocab:
                vocab.append(c)
        while len(vocab) < vocab_size:
            c = rng.choice(all_ids)
            if c not in vocab:
                vocab.append(c)
        vocab = vocab[:vocab_size]

        eid = f"{split}_v{vocab_size}_s{seed}_n{noise}_e{i:03d}"
        ep = build_episode(eid, ep_images, vocab, lvis, class_texts_raw, noise, split, seed + i)
        save_episode(ep, out_dir / f"{eid}.json")
        episodes.append(ep)
    return episodes


def generate_all(
    dev_ids: list[int],
    train_pool: list[int],
    cfg_ep: dict[str, Any],
    out_root: Path,
) -> dict[str, Any]:
    lvis = load_lvis_minival()
    _, class_texts_raw = load_class_texts()
    summary: dict[str, Any] = {"episodes": [], "counts": {}}

    for seed in cfg_ep["seeds"]:
        for vs in cfg_ep["vocab_sizes"]:
            for noise in cfg_ep["noise_types"]:
                key = f"dev_v{vs}_s{seed}_{noise}"
                eps = generate_episode_batch(
                    out_root / "dev" / key,
                    "dev",
                    dev_ids,
                    vs,
                    seed,
                    cfg_ep["dev"]["n_episodes_per_config"],
                    noise,
                    lvis,
                    class_texts_raw,
                )
                summary["counts"][key] = len(eps)
                summary["episodes"].extend([e.episode_id for e in eps])

    for seed in cfg_ep["seeds"][:1]:
        key = f"train_s{seed}"
        eps = generate_episode_batch(
            out_root / "train" / key,
            "train",
            train_pool,
            30,
            seed,
            cfg_ep["train"]["n_episodes"],
            "none",
            lvis,
            class_texts_raw,
            n_images_per_episode=5,
        )
        summary["counts"][key] = len(eps)

    return summary
