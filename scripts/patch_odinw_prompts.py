"""Patch domain.json prompts for weak ODinW domains (no re-download)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "dod", ROOT / "scripts" / "download_odinw_roboflow.py"
)
_dod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dod)
DOMAINS = _dod.DOMAINS

PATCH_SLUGS = ("packages", "pothole", "shellfish")


def main() -> None:
    for slug in PATCH_SLUGS:
        meta = DOMAINS.get(slug, {})
        prompts = meta.get("prompts")
        if not prompts:
            continue
        dpath = ROOT / "data" / "odinw" / slug / "domain.json"
        if not dpath.is_file():
            print(f"SKIP missing {dpath}")
            continue
        data = json.loads(dpath.read_text(encoding="utf-8"))
        data["prompts"] = prompts
        dpath.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Patched prompts: {slug}")
    print("Done")


if __name__ == "__main__":
    main()
