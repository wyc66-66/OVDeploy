"""Deployment Scenario Pack schema and loader."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PACK_DIR = ROOT / "config" / "deployment_scenario_packs"


@dataclass
class DeploymentScenarioPack:
    pack_id: str
    name: str
    tier: str
    operator: str
    v_source: str
    b0_meaning: str
    b5_meaning: str
    success_criterion: str
    data_source: str
    vocab_source: str  # lvis | native | human | config
    episode_dir: str
    vocab_sizes: list[int] = field(default_factory=lambda: [10, 30])
    noises: list[str] = field(default_factory=lambda: ["none"])
    baselines: list[str] = field(default_factory=lambda: ["B0_full", "B5_subset"])
    eval_mode: str = "lvis"  # lvis | odinw | nuscenes | vignette | native_coco
    odinw_slug: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml_dict(cls, d: dict) -> DeploymentScenarioPack:
        return cls(
            pack_id=d["pack_id"],
            name=d.get("name", d["pack_id"]),
            tier=d.get("tier", "tier2"),
            operator=d.get("operator", ""),
            v_source=d.get("v_source", ""),
            b0_meaning=d.get("b0_meaning", ""),
            b5_meaning=d.get("b5_meaning", ""),
            success_criterion=d.get("success_criterion", ""),
            data_source=d.get("data_source", ""),
            vocab_source=d.get("vocab_source", "native"),
            episode_dir=d.get("episode_dir", ""),
            vocab_sizes=list(d.get("vocab_sizes", [10, 30])),
            noises=list(d.get("noises", ["none"])),
            baselines=list(d.get("baselines", ["B0_full", "B5_subset"])),
            eval_mode=d.get("eval_mode", "lvis"),
            odinw_slug=d.get("odinw_slug", ""),
            meta=dict(d.get("meta", {})),
        )

    def episodes_path(self) -> Path:
        return ROOT / self.episode_dir

    def vocab_config_path(self) -> Path | None:
        p = self.meta.get("vocab_config")
        if p:
            return ROOT / p
        return None


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        return json.loads(path.read_text(encoding="utf-8"))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_pack(pack_id: str) -> DeploymentScenarioPack:
    pid = pack_id.upper().replace("_", "-")
    if not pid.startswith("DSP-"):
        pid = f"DSP-{pid}"
    path = PACK_DIR / f"{pid.lower()}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Pack config not found: {path}")
    return DeploymentScenarioPack.from_yaml_dict(_load_yaml(path))


def list_packs() -> list[str]:
    if not PACK_DIR.is_dir():
        return []
    return sorted(p.stem.upper().replace("dsp-", "DSP-") for p in PACK_DIR.glob("dsp-*.yaml"))
