"""VocabRecover: hint-free missing-class recovery via B0 evidence + co-occurrence + two-round probe."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ovdeploy.vocab import freq_sorted_cat_ids
from robustvocab.cooccur import cooccur_score, load_cooccur_prior


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class B0EvidenceRanker:
    """Rank OOV B0 detections as missing-class candidates (deployment-strict)."""

    def __init__(
        self,
        score_thr: float = 0.15,
        top_k: int = 32,
        iou_thr: float = 0.5,
    ) -> None:
        self.score_thr = score_thr
        self.top_k = top_k
        self.iou_thr = iou_thr

    def rank(
        self,
        b0_preds: list[dict],
        user_vocab: Sequence[int],
    ) -> list[tuple[int, float]]:
        core = set(user_vocab)
        oov = [p for p in b0_preds if p.get("category_id", -1) not in core]
        oov.sort(key=lambda p: -float(p["score"]))

        clusters: list[dict] = []
        for p in oov:
            if float(p["score"]) < self.score_thr:
                continue
            cid = int(p["category_id"])
            bbox = p["bbox"]
            merged = False
            for cl in clusters:
                if cl["cid"] == cid and _iou(bbox, cl["bbox"]) >= self.iou_thr:
                    cl["score"] = max(cl["score"], float(p["score"]))
                    merged = True
                    break
            if not merged:
                clusters.append({"cid": cid, "score": float(p["score"]), "bbox": bbox})

        clusters.sort(key=lambda x: -x["score"])
        return [(c["cid"], c["score"]) for c in clusters[: self.top_k]]


class VocabRecover:
    """Expand user vocabulary using B0 evidence, co-occurrence prior, and optional two-round probe."""

    def __init__(
        self,
        delta: int = 8,
        delta_missing: int = 20,
        b0_score_thr: float = 0.15,
        b0_top_k: int = 32,
        cooccur_weight: float = 0.35,
        use_cooccur: bool = True,
        two_round: bool = True,
        round2_top: int = 12,
        cand_pool: int = 128,
    ) -> None:
        self.delta = delta
        self.delta_missing = delta_missing
        self.cooccur_weight = cooccur_weight if use_cooccur else 0.0
        self.use_cooccur = use_cooccur
        self.two_round = two_round
        self.round2_top = round2_top
        self.cand_pool = cand_pool
        self.ranker = B0EvidenceRanker(score_thr=b0_score_thr, top_k=b0_top_k)

    def _combined_scores(
        self,
        ranked_b0: list[tuple[int, float]],
        core_vocab: Sequence[int],
        prior: dict,
        probe_scores: dict[int, float] | None = None,
        noise: str = "none",
    ) -> list[tuple[int, float]]:
        out: list[tuple[int, float]] = []
        for cid, b0s in ranked_b0:
            co = cooccur_score(cid, core_vocab, prior) if self.use_cooccur else 0.0
            ps = (probe_scores or {}).get(cid, 0.0)
            b0_w = 1.5 if noise == "missing_class" else 1.0
            combined = b0_w * b0s + self.cooccur_weight * co + 1.0 * ps
            out.append((cid, combined))
        out.sort(key=lambda x: -x[1])
        return out

    def recover(
        self,
        core_vocab: Sequence[int],
        target_size: int,
        image_rgb: np.ndarray,
        image_id: int,
        backend,
        lvis: dict,
        b0_preds: list[dict] | None = None,
        noise: str = "none",
        freq_cats: list[int] | None = None,
        all_cat_ids: list[int] | None = None,
    ) -> list[int]:
        core = list(dict.fromkeys(core_vocab))
        freq_cats = freq_cats or freq_sorted_cat_ids(lvis)
        all_cat_ids = all_cat_ids or [c["id"] for c in lvis["categories"]]
        extra = self.delta_missing if noise == "missing_class" else self.delta
        if noise == "missing_class":
            budget = min(len(core) + extra, len(all_cat_ids))
        else:
            budget = min(len(core) + extra, target_size, len(all_cat_ids))
        budget = max(budget, len(core))

        if len(core) >= budget:
            return core[:budget]

        prior = load_cooccur_prior(lvis)
        ranked_b0 = self.ranker.rank(b0_preds or [], core)

        if not ranked_b0 and noise == "missing_class":
            for c in freq_cats:
                if c not in set(core):
                    ranked_b0.append((c, 0.01))
                if len(ranked_b0) >= self.cand_pool // 4:
                    break

        cand_ids = [c for c, _ in ranked_b0]
        for c in freq_cats:
            if c not in set(core) and c not in cand_ids:
                cand_ids.append(c)
            if len(cand_ids) >= self.cand_pool:
                break

        probe_vocab = core + [c for c in cand_ids if c not in set(core)]
        round1_scores: dict[int, float] = {}
        if hasattr(backend, "score_classes"):
            round1_scores = backend.score_classes(image_rgb, probe_vocab, image_id, lvis)

        combined = self._combined_scores(ranked_b0, core, prior, round1_scores, noise=noise)

        if self.two_round and hasattr(backend, "score_classes") and combined:
            round2_cands = [c for c, _ in combined[: self.round2_top] if c not in set(core)]
            if round2_cands:
                r2_vocab = core + round2_cands
                r2_scores = backend.score_classes(image_rgb, r2_vocab, image_id, lvis)
                for cid in round2_cands:
                    idx = next((i for i, (c, _) in enumerate(combined) if c == cid), None)
                    if idx is not None:
                        c, old = combined[idx]
                        combined[idx] = (c, old + float(r2_scores.get(cid, 0.0)))
                combined.sort(key=lambda x: -x[1])

        need = budget - len(core)
        extras = []
        seen = set(core)
        for cid, _ in combined:
            if cid in seen:
                continue
            extras.append(cid)
            seen.add(cid)
            if len(extras) >= need:
                break

        if len(extras) < need:
            for c in freq_cats:
                if c not in seen:
                    extras.append(c)
                    seen.add(c)
                if len(extras) >= need:
                    break

        return core + extras[:need]
