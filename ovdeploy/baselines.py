"""Baseline vocabulary selection strategies."""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

from ovdeploy.generator import select_vocab_for_baseline

if TYPE_CHECKING:
    from ovdeploy.episode import Episode


def vocab_for_baseline(
    baseline: str,
    episode: Episode,
    img_cats: dict[int, set[int]],
    all_cat_ids: list[int],
    freq_cats: list[int],
    oracle_delta: int = 3,
) -> list[int]:
    vs = episode.vocab_size if episode.vocab_size < len(all_cat_ids) else len(all_cat_ids)
    rng = random.Random(episode.seed)

    if baseline == "B0_full":
        return list(all_cat_ids)
    if baseline == "B5_subset" or baseline == "M1_adapter":
        return list(episode.vocab.cat_ids)
    if baseline == "B4_clip":
        # Per-image CLIP top-K in infer.py
        return list(episode.vocab.cat_ids)
    return select_vocab_for_baseline(
        baseline, episode.image_ids, vs, img_cats, all_cat_ids, freq_cats, rng, oracle_delta
    )
