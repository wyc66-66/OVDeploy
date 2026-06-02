"""VocabShift between train and eval episodes."""
from __future__ import annotations

from ovdeploy.episode import Episode


def vocab_overlap(a: Episode, b: Episode) -> float:
    sa, sb = set(a.vocab.cat_ids), set(b.vocab.cat_ids)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def vocab_shift_stats(
    train_eps: list[Episode], eval_eps: list[Episode], max_pairs: int = 200
) -> dict:
    overlaps = []
    for te in train_eps[: max_pairs // 2]:
        for ee in eval_eps[: max_pairs // 2]:
            overlaps.append(vocab_overlap(te, ee))
    mean_ov = sum(overlaps) / max(len(overlaps), 1)
    return {
        "mean_vocab_overlap": round(mean_ov, 4),
        "vocab_shift": round(1.0 - mean_ov, 4),
        "n_pairs": len(overlaps),
    }
