"""Smoke test for VocabRecover ranking (no GPU)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from robustvocab.paths_util import load_config

load_config()

from robustvocab.recover import B0EvidenceRanker


def main() -> None:
    ranker = B0EvidenceRanker(score_thr=0.1, top_k=5)
    b0 = [
        {"category_id": 100, "score": 0.8, "bbox": [10, 10, 50, 50]},
        {"category_id": 200, "score": 0.6, "bbox": [100, 100, 40, 40]},
        {"category_id": 100, "score": 0.7, "bbox": [12, 12, 48, 48]},
    ]
    ranked = ranker.rank(b0, user_vocab=[1, 2, 3])
    assert ranked[0][0] == 100
    assert len(ranked) >= 2
    print("VocabRecover smoke OK:", ranked)


if __name__ == "__main__":
    main()
