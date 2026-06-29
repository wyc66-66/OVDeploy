"""Resolve OVDeploy repo root and pilot data paths (dev tree vs ovdeploy-public)."""
from __future__ import annotations

from pathlib import Path


def ovdeploy_root(start: Path | None = None) -> Path:
    here = start or Path(__file__).resolve()
    for anc in list(here.parents):
        if (anc / "ovdeploy").is_dir() and (anc / "config").is_dir():
            return anc
    if len(here.parents) >= 4:
        paper = here.parents[3] / "submission-a"
        if (paper / "ovdeploy").is_dir():
            return paper
    raise RuntimeError("Cannot locate OVDeploy root (ovdeploy/ + config/)")


def pilot_layout(script_file: Path) -> tuple[Path, Path, Path]:
    """Return (ovdeploy_root, config_dir, data_dir)."""
    script = script_file.resolve()
    repo = ovdeploy_root(script)
    pilot_dir = script.parents[1]
    if (pilot_dir / "config").is_dir() and (pilot_dir / "data").is_dir():
        return repo, pilot_dir / "config", pilot_dir / "data"
    return repo, repo / "config", repo / "data"
