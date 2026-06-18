"""Detector-native vocabulary routing (VocabRouter)."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ovdeploy.vocab import freq_sorted_cat_ids


class VocabRouter:
    """Expand user vocabulary V to V' using detector-native class scores."""

    def __init__(
        self,
        delta: int = 3,
        cand_pool: int = 64,
        missing_class_extra_delta: int = 3,
        use_clip_fallback: bool = False,
    ) -> None:
        self.delta = delta
        self.cand_pool = cand_pool
        self.missing_class_extra_delta = missing_class_extra_delta
        self.use_clip_fallback = use_clip_fallback

    def _candidate_pool(
        self,
        core_vocab: Sequence[int],
        freq_cats: list[int],
        all_cat_ids: list[int],
        b0_hint_classes: Sequence[int] | None = None,
        cand_pool: int | None = None,
    ) -> list[int]:
        core_set = set(core_vocab)
        pool = cand_pool if cand_pool is not None else self.cand_pool
        cand: list[int] = []
        for c in b0_hint_classes or []:
            if c not in core_set and c not in cand:
                cand.append(c)
        for c in freq_cats:
            if c not in core_set and c not in cand:
                cand.append(c)
            if len(cand) >= pool:
                break
        if len(cand) < pool // 2:
            for c in all_cat_ids:
                if c not in core_set and c not in cand:
                    cand.append(c)
                if len(cand) >= pool:
                    break
        return cand

    def route(
        self,
        core_vocab: Sequence[int],
        target_size: int,
        image_rgb: np.ndarray,
        image_id: int,
        backend,
        lvis: dict,
        freq_cats: list[int] | None = None,
        all_cat_ids: list[int] | None = None,
        noise: str = "none",
        b0_hint_classes: Sequence[int] | None = None,
        priority_hint_classes: Sequence[int] | None = None,
        cand_pool_override: int | None = None,
    ) -> list[int]:
        extra = self.missing_class_extra_delta if noise == "missing_class" else 0
        delta = self.delta + extra
        core = list(dict.fromkeys(core_vocab))
        freq_cats = freq_cats or freq_sorted_cat_ids(lvis)
        all_cat_ids = all_cat_ids or [c["id"] for c in lvis["categories"]]
        budget = min(len(core) + delta, len(all_cat_ids))
        budget = max(budget, len(core))

        if len(core) >= budget:
            return core[:budget]

        pool = cand_pool_override if cand_pool_override is not None else self.cand_pool
        merged_hints: list[int] = []
        for c in priority_hint_classes or []:
            if c not in merged_hints:
                merged_hints.append(c)
        for c in b0_hint_classes or []:
            if c not in merged_hints:
                merged_hints.append(c)

        cand = self._candidate_pool(core, freq_cats, all_cat_ids, merged_hints, pool)
        probe_vocab = core + [c for c in cand if c not in set(core)]

        if self.use_clip_fallback or not hasattr(backend, "score_classes"):
            scores = self._clip_scores(image_rgb, probe_vocab, lvis)
        else:
            scores = backend.score_classes(image_rgb, probe_vocab, image_id, lvis)
            priority_set = set(priority_hint_classes or [])
            for c in b0_hint_classes or []:
                if c in scores:
                    scores[c] = max(scores[c], 1.0)
            for c in priority_set:
                if c in scores:
                    scores[c] = max(scores[c], 2.0)

        extras = [c for c in probe_vocab if c not in set(core)]
        ranked = sorted(extras, key=lambda c: -scores.get(c, 0.0))
        need = budget - len(core)
        return core + ranked[:need]

    def _clip_scores(
        self, image_rgb: np.ndarray, cat_ids: list[int], lvis: dict
    ) -> dict[int, float]:
        from ovdeploy.clip_vocab import clip_topk_cat_ids
        from ovdeploy.paths_util import load_class_texts

        _, class_texts_raw = load_class_texts()
        prompts: dict[int, list[str]] = {}
        for i, c in enumerate(sorted(lvis["categories"], key=lambda x: x["id"])):
            if i < len(class_texts_raw):
                t = class_texts_raw[i]
                prompts[c["id"]] = t if isinstance(t, list) else [str(t)]
            else:
                prompts[c["id"]] = [c["name"]]

        freq = freq_sorted_cat_ids(lvis)
        top = clip_topk_cat_ids(
            image_rgb, cat_ids, prompts, len(cat_ids), freq, seed=0
        )
        scores = {c: 0.0 for c in cat_ids}
        for rank, cid in enumerate(reversed(top)):
            scores[cid] = (rank + 1) / max(len(top), 1)
        return scores
