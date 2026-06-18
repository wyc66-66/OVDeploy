"""LVIS co-occurrence prior for hint-free vocabulary recovery."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from robustvocab.paths_util import cooccur_cache_path, load_config


def _build_from_lvis(lvis: dict) -> dict[str, dict[str, float]]:
    """P(c_j | c_i in same image) for category ids."""
    by_img: dict[int, set[int]] = defaultdict(set)
    for ann in lvis["annotations"]:
        by_img[ann["image_id"]].add(ann["category_id"])

    pair_count: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    single_count: dict[int, int] = defaultdict(int)
    for cats in by_img.values():
        clist = sorted(cats)
        for c in clist:
            single_count[c] += 1
        for i, ci in enumerate(clist):
            for cj in clist[i + 1 :]:
                pair_count[ci][cj] += 1
                pair_count[cj][ci] += 1

    prior: dict[str, dict[str, float]] = {}
    for ci, neighbors in pair_count.items():
        denom = max(single_count[ci], 1)
        prior[str(ci)] = {str(cj): cnt / denom for cj, cnt in neighbors.items()}
    return prior


def load_cooccur_prior(lvis: dict, force_rebuild: bool = False) -> dict[str, dict[str, float]]:
    cache = cooccur_cache_path()
    if cache.is_file() and not force_rebuild:
        return json.loads(cache.read_text(encoding="utf-8"))

    prior = _build_from_lvis(lvis)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(prior), encoding="utf-8")
    return prior


def cooccur_score(
    candidate: int,
    core_vocab: Sequence[int],
    prior: dict[str, dict[str, float]],
) -> float:
    if not core_vocab:
        return 0.0
    scores = []
    row = prior.get(str(candidate), {})
    for c in core_vocab:
        scores.append(row.get(str(c), 0.0))
    return sum(scores) / max(len(scores), 1)
