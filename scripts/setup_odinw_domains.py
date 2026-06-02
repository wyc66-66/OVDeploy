"""Create ODinW domain metadata under data/odinw/."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DOMAINS = {
    "aquarium": {
        "domain": "Aquarium",
        "classes": ["fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray"],
    },
    "packages": {"domain": "Packages", "classes": ["package", "box", "suitcase", "backpack"]},
    "fryingpan": {"domain": "FryingPan", "classes": ["egg", "frying pan", "spatula", "bowl", "pan"]},
}


def main() -> None:
    base = ROOT / "data/odinw"
    base.mkdir(parents=True, exist_ok=True)
    for slug, meta in DOMAINS.items():
        ddir = base / slug
        ddir.mkdir(parents=True, exist_ok=True)
        meta_path = ddir / "domain.json"
        if meta_path.is_file():
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            if existing.get("source"):
                continue
        payload = {**meta, "max_images": 100}
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (base / "README_ODINW.txt").write_text(
        "Domain class lists for OVDeploy cross-domain episodic eval.\n"
        "Images: LVIS minival filtered by domain keyword overlap when no local images.\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(DOMAINS)} domains under {base}")


if __name__ == "__main__":
    main()
