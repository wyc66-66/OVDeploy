"""Light C1 hyperparameter sweep on dev_v10_s42_missing_class."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PATHS = ROOT / "config" / "paths.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    if not args.gpu:
        raise SystemExit("GPU required")

    best_gain = -1.0
    best = (25, 0.5)
    for delta in (25, 30, 35):
        for co in (0.5, 0.6):
            overrides = json.dumps({"recover_delta_missing": delta, "cooccur_weight": co})
            out = ROOT / f"reports/REPORT_RV_c1_sweep_d{delta}_c{co}.json"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/rv/run_robustvocab_eval.py",
                    "--gpu",
                    "--config-key",
                    "dev_v10_s42_missing_class",
                    "--max-episodes",
                    "10",
                    "--methods",
                    "B5_subset,RV_recover,RV_full",
                    "--recover-overrides",
                    overrides,
                    "--report",
                    str(out.relative_to(ROOT)),
                ],
                cwd=ROOT,
                check=True,
            )
            data = json.loads(out.read_text(encoding="utf-8"))
            rows = data.get("rows", [])
            b5 = next((r["EpisodicAP_mean"] for r in rows if r["method"] == "B5_subset"), 0.0)
            rv = max(
                (
                    r["EpisodicAP_mean"]
                    for r in rows
                    if r["method"] in ("RV_recover", "RV_full")
                ),
                default=0.0,
            )
            gain = (rv - b5) / max(b5, 1e-6)
            print(f"delta={delta} co={co} rel_gain={gain:.4f}")
            if gain > best_gain:
                best_gain = gain
                best = (delta, co)

    text = PATHS.read_text(encoding="utf-8")
    text = re.sub(
        r"recover_delta_missing:\s*\d+",
        f"recover_delta_missing: {best[0]}",
        text,
        count=1,
    )
    text = re.sub(
        r"cooccur_weight:\s*[\d.]+",
        f"cooccur_weight: {best[1]}",
        text,
        count=1,
    )
    PATHS.write_text(text, encoding="utf-8")
    print(f"Updated paths.yaml: recover_delta_missing={best[0]} cooccur_weight={best[1]} gain={best_gain:.4f}")


if __name__ == "__main__":
    main()
