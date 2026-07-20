"""Episodic metrics: EpisodicAP v2, OOV-FP, ECE."""
from __future__ import annotations

from typing import Iterable


def _pred_category_id(p: dict) -> int:
    """YOLO/GDINO use category_id; some VLM backends historically used cat_id."""
    if "category_id" in p:
        return int(p["category_id"])
    if "cat_id" in p:
        return int(p["cat_id"])
    return -1


def iou_box(a: list[float], b: list[float]) -> float:
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


def ap_from_pr(precisions: list[float], recalls: list[float]) -> float:
    if not precisions:
        return 0.0
    mrec = [0.0] + sorted(recalls) + [1.0]
    mpre = [0.0] + sorted(precisions, key=lambda x: -x) + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return min(ap * 100.0, 100.0)


def episodic_ap_per_image_v2(
    preds: list[dict],
    gt_boxes: list[list[float]],
    gt_cat_ids: list[int],
    vocab_cat_ids: Iterable[int],
    score_thr: float = 0.05,
    iou_thr: float = 0.5,
) -> float:
    """Greedy IoU matching AP on GT in vocab; capped at 100."""
    vocab_set = set(vocab_cat_ids)
    gt_pairs = [(b, c) for b, c in zip(gt_boxes, gt_cat_ids) if c in vocab_set]
    if not gt_pairs:
        return 0.0

    sorted_preds = sorted(
        [p for p in preds if p["score"] >= score_thr and _pred_category_id(p) in vocab_set],
        key=lambda p: -p["score"],
    )
    if not sorted_preds:
        return 0.0

    matched_gt = [False] * len(gt_pairs)
    tp_flags: list[bool] = []

    for p in sorted_preds:
        pc = _pred_category_id(p)
        best_iou, best_j = 0.0, -1
        for j, (box, gcat) in enumerate(gt_pairs):
            if matched_gt[j] or gcat != pc:
                continue
            iou = iou_box(p["bbox"], box)
            if iou > best_iou:
                best_iou, best_j = iou, j
        is_tp = best_j >= 0 and best_iou >= iou_thr
        if is_tp:
            matched_gt[best_j] = True
        tp_flags.append(is_tp)

    n_gt = len(gt_pairs)
    tp = fp = 0
    precisions, recalls = [], []
    for is_tp in tp_flags:
        if is_tp:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_gt)
    return ap_from_pr(precisions, recalls)


def episodic_ap_per_image(
    preds: list[dict],
    gt_boxes: list[list[float]],
    gt_cat_ids: list[int],
    vocab_cat_ids: Iterable[int],
    score_thr: float = 0.05,
) -> float:
    """Default: v2 greedy AP."""
    return episodic_ap_per_image_v2(
        preds, gt_boxes, gt_cat_ids, vocab_cat_ids, score_thr=score_thr
    )


def oov_fp_rate(
    b0_preds: list[dict],
    vocab_cat_ids: Iterable[int],
    score_thr: float = 0.5,
) -> float:
    """Fraction of high-score B0 preds whose class is outside episode vocab V."""
    vocab_set = set(vocab_cat_ids)
    high = [p for p in b0_preds if p["score"] >= score_thr]
    if not high:
        return 0.0
    oov = sum(1 for p in high if _pred_category_id(p) not in vocab_set)
    return oov / len(high)


def fp_non_gt_rate(
    preds: list[dict],
    vocab_cat_ids: Iterable[int],
    score_thr: float = 0.5,
) -> float:
    vocab_set = set(vocab_cat_ids)
    high = [p for p in preds if p["score"] >= score_thr]
    if not high:
        return 0.0
    bad = sum(1 for p in high if _pred_category_id(p) not in vocab_set)
    return bad / len(high)


def ece_binary(scores: list[float], labels: list[int], n_bins: int = 10) -> float:
    if not scores:
        return 0.0
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for s, y in zip(scores, labels):
        b = min(int(s * n_bins), n_bins - 1)
        bins[b].append((s, y))
    ece = 0.0
    n = len(scores)
    for b in bins:
        if not b:
            continue
        acc = sum(y for _, y in b) / len(b)
        conf = sum(s for s, _ in b) / len(b)
        ece += len(b) / n * abs(acc - conf)
    return ece
