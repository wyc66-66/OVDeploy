"""Episode dataclass and I/O."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EpisodeVocab:
    cat_ids: list[int]
    prompts: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"cat_ids": self.cat_ids, "prompts": self.prompts}


@dataclass
class Episode:
    episode_id: str
    image_ids: list[int]
    vocab: EpisodeVocab
    vocab_size: int
    noise: str
    split: str
    baseline: str = ""
    seed: int = 42
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["vocab"] = self.vocab.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Episode:
        v = d.get("vocab", {})
        vocab = EpisodeVocab(
            cat_ids=list(v.get("cat_ids", [])),
            prompts={str(k): val for k, val in v.get("prompts", {}).items()},
        )
        return cls(
            episode_id=d["episode_id"],
            image_ids=list(d["image_ids"]),
            vocab=vocab,
            vocab_size=int(d.get("vocab_size", len(vocab.cat_ids))),
            noise=d.get("noise", "none"),
            split=d.get("split", "eval"),
            baseline=d.get("baseline", ""),
            seed=int(d.get("seed", 42)),
            meta=dict(d.get("meta", {})),
        )


def save_episode(ep: Episode, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ep.to_dict(), indent=2), encoding="utf-8")


def load_episode(path: Path) -> Episode:
    return Episode.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_episodes_dir(directory: Path) -> list[Episode]:
    eps = []
    for p in sorted(directory.rglob("*.json")):
        eps.append(load_episode(p))
    return eps
