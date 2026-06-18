"""PromptAlign: synonym-robust dual-prompt fusion for subset inference."""
from __future__ import annotations

from typing import Sequence

from ovdeploy.paths_util import cat_id_to_index, load_class_texts
from ovdeploy.vocab import subset_class_texts


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


def _merge_preds(a: list[dict], b: list[dict], iou_thr: float = 0.5) -> list[dict]:
    """Per-class max-score fusion with IoU-based dedup."""
    merged = list(a)
    for pb in b:
        cid_b = pb.get("category_id", -1)
        matched = False
        for i, pa in enumerate(merged):
            if pa.get("category_id") != cid_b:
                continue
            if _iou(pa["bbox"], pb["bbox"]) >= iou_thr:
                if float(pb["score"]) > float(pa["score"]):
                    merged[i] = dict(pb)
                matched = True
                break
        if not matched:
            merged.append(dict(pb))
    merged.sort(key=lambda p: -float(p["score"]))
    return merged


def _score_mass(preds: list[dict]) -> float:
    return float(sum(float(p.get("score", 0.0)) for p in preds))


class PromptAlign:
    """Dual-prompt fusion: user episode prompt + canonical LVIS prompt."""

    def __init__(self, fusion: str = "max_score") -> None:
        self.fusion = fusion

    def build_dual_texts(
        self,
        vocab_ids: Sequence[int],
        episode_prompts: dict[str, list[str]],
        class_texts_raw: list,
        cid2idx: dict[int, int],
    ) -> list:
        canonical = subset_class_texts(class_texts_raw, list(vocab_ids), cid2idx)
        out = []
        for i, cid in enumerate(vocab_ids):
            user = episode_prompts.get(str(cid), [])
            canon = canonical[i] if i < len(canonical) else [str(cid)]
            if isinstance(user, list) and user:
                u = user[0] if user else (canon[0] if isinstance(canon, list) else str(canon))
            else:
                u = canon[0] if isinstance(canon, list) else str(canon)
            c = canon[0] if isinstance(canon, list) else str(canon)
            if u != c:
                out.append([u, c])
            else:
                out.append(canon if isinstance(canon, list) else [str(canon)])
        return out

    def predict(
        self,
        backend,
        image_rgb,
        vocab_ids: list[int],
        image_id: int,
        lvis: dict,
        episode_prompts: dict[str, list[str]],
        use_dual: bool = True,
    ) -> list[dict]:
        if not use_dual or not episode_prompts:
            return backend.predict(image_rgb, vocab_ids, image_id, lvis)

        class_names, class_texts_raw = load_class_texts()
        cid2idx = cat_id_to_index(lvis)

        if hasattr(backend, "predict_with_texts"):
            user_texts = []
            canon_texts = []
            for i, cid in enumerate(vocab_ids):
                ep = episode_prompts.get(str(cid), [])
                if ep:
                    user_texts.append(ep if isinstance(ep, list) else [str(ep)])
                else:
                    dual = self.build_dual_texts(
                        vocab_ids, episode_prompts, class_texts_raw, cid2idx
                    )
                    user_texts.append(dual[i])
                canon = subset_class_texts(class_texts_raw, [cid], cid2idx)[0]
                canon_texts.append(canon if isinstance(canon, list) else [str(canon)])

            preds_user = backend.predict_with_texts(
                image_rgb, vocab_ids, image_id, lvis, user_texts
            )
            preds_canon = backend.predict_with_texts(
                image_rgb, vocab_ids, image_id, lvis, canon_texts
            )
            preds_dual = _merge_preds(preds_user, preds_canon)
            candidates = [
                (preds_user, _score_mass(preds_user)),
                (preds_canon, _score_mass(preds_canon)),
                (preds_dual, _score_mass(preds_dual)),
            ]
            return max(candidates, key=lambda x: x[1])[0]

        return backend.predict(image_rgb, vocab_ids, image_id, lvis)
