#!/usr/bin/env python3
"""Download YOLO-World v2-M checkpoint to YOLO-World/weights/."""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

URL = (
    "https://hf-mirror.com/wondervictor/YOLO-World/resolve/main/"
    "yolo_world_v2_m_obj365v1_goldg_pretrain-c6237d5b.pth"
)
FNAME = "yolo_world_v2_m_obj365v1_goldg_pretrain-c6237d5b.pth"


def main() -> None:
    from ovdeploy.paths_util import load_paths

    cfg = load_paths()
    dest = cfg["_yolo"] / "weights" / FNAME
    if dest.is_file() and dest.stat().st_size > 300_000_000:
        print(f"SKIP: already at {dest}")
        return

    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    url = URL.replace("https://hf-mirror.com", endpoint.rstrip("/"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "OVDeploy/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = resp.read()
    dest.write_bytes(data)
    print(f"OK {len(data)} bytes -> {dest}")


if __name__ == "__main__":
    main()
