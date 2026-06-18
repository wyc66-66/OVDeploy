"""OOV-aware score calibration for full-vocabulary (B0) predictions."""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


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


class OOVGuard:
    """Soft suppression of out-of-vocabulary high-confidence B0 detections."""

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 0.3,
        tau: float = 0.5,
        iou_thr: float = 0.3,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.iou_thr = iou_thr

    def _region_sim_to_vocab(
        self,
        bbox: list[float],
        in_vocab_preds: list[dict],
    ) -> float:
        if not in_vocab_preds:
            return 0.0
        best = 0.0
        for p in in_vocab_preds:
            iou = _iou(bbox, p["bbox"])
            if iou >= self.iou_thr:
                best = max(best, float(p["score"]) * iou)
        return best

    def guard(
        self,
        b0_preds: list[dict],
        user_vocab: Sequence[int],
        in_vocab_preds: list[dict] | None = None,
        hard_mask: bool = False,
    ) -> list[dict]:
        vocab_set = set(user_vocab)
        in_v = [p for p in (in_vocab_preds or b0_preds) if p.get("category_id") in vocab_set]
        out: list[dict] = []

        for p in b0_preds:
            cid = p.get("category_id", -1)
            s = float(p["score"])
            if cid in vocab_set:
                out.append(dict(p))
                continue
            if hard_mask:
                continue
            region_sim = self._region_sim_to_vocab(p["bbox"], in_v)
            factor = _sigmoid(self.alpha * region_sim - self.beta)
            s_adj = s * (1.0 - factor)
            if s_adj >= self.tau:
                q = dict(p)
                q["score"] = s_adj
                out.append(q)
        return out

    def filter_for_eval(
        self,
        preds: list[dict],
        user_vocab: Iterable[int],
        score_thr: float = 0.5,
    ) -> list[dict]:
        vocab_set = set(user_vocab)
        return [
            p
            for p in preds
            if p["score"] >= score_thr and p.get("category_id", -1) in vocab_set
        ]
