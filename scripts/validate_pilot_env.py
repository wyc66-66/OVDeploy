#!/usr/bin/env python3
"""Smoke-test nuScenes pilot imports and config (no dataset required)."""
from __future__ import annotations

import sys
from pathlib import Path

from _pilot_layout import ovdeploy_root, pilot_layout

_REPO, _CFG_DIR, _ = pilot_layout(Path(__file__))
sys.path.insert(0, str(_REPO))

from ovdeploy.nuscenes.taxonomy import (  # noqa: E402
    load_pilot_config,
    load_taxonomy,
    resolve_class_map_path,
)


def main() -> int:
    cfg = load_pilot_config(_REPO / "config" / "nuscenes_pilot.yaml")
    tax = load_taxonomy(resolve_class_map_path(cfg))
    assert len(tax.all_cat_ids) == 23, tax.all_cat_ids
    print(f"OK: repo={_REPO}")
    print(f"OK: taxonomy {len(tax.all_cat_ids)} classes, camera={cfg['camera']}")
    print(f"    class_map={resolve_class_map_path(cfg)}")
  # Optional report files (outreach tree)
    pilot = Path(__file__).resolve().parents[1]
    reports = pilot / "reports"
    for name in (
        "REPORT_nuscenes_main.json",
        "REPORT_nuscenes_multicam.json",
        "REPORT_drivevlm_vocab_smoke.json",
        "REPORT_occ3d_subset.json",
        "drivevlm_oov_curve.png",
        "OCC3D_SUBSET_TABLE.md",
    ):
        p = reports / name
        status = "OK" if p.is_file() else "MISSING"
        print(f"    {status}: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
