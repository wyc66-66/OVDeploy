#!/usr/bin/env python3
"""Verify or stage DetCLIPv2-T checkpoint paths (manual copy or URL download)."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HUNT_LOG = ROOT / "reports" / "detclip_v2_hunt_log.json"


def _download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "OVDeploy-DetCLIP/1.0"})
    print(f"Downloading {url} -> {dest}", flush=True)
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    dest.write_bytes(data)
    print(f"OK {len(data)} bytes", flush=True)


def _first_hunt_url(log_path: Path) -> str | None:
    if not log_path.is_file():
        return None
    log = json.loads(log_path.read_text(encoding="utf-8"))
    urls = log.get("download_urls") or []
    for u in urls:
        low = u.lower()
        if ".pth" in low or "drive.google" in low or "huggingface" in low:
            return u
    return urls[0] if urls else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        help="Source .pth from authors; copied to weights/detclipv2_swin_t/",
    )
    parser.add_argument(
        "--config",
        help="Source MMDet config .py; copied to third_party/DetCLIPv2/configs/",
    )
    parser.add_argument("--from-url", help="HTTP(S) direct link to .pth checkpoint")
    parser.add_argument(
        "--from-hunt-log",
        action="store_true",
        help="Download first URL from reports/detclip_v2_hunt_log.json",
    )
    parser.add_argument("--hunt-log", default=str(HUNT_LOG))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    from ovdeploy.backends.detclip import checkpoint_ready
    from ovdeploy.paths_util import load_paths

    cfg = load_paths()
    dc = cfg.get("detclip_v2", {})
    ckpt_dst = ROOT / dc.get("checkpoint", "weights/detclipv2_swin_t/detclipv2_swin_t.pth")
    cfg_dst = ROOT / dc.get("config", "third_party/DetCLIPv2/configs/detclipv2_swin_t_lvis.py")

    if args.from_hunt_log:
        url = _first_hunt_url(Path(args.hunt_log))
        if not url:
            print("No download URL in hunt log", file=sys.stderr)
            sys.exit(1)
        args.from_url = url

    if args.from_url:
        try:
            _download_url(args.from_url, ckpt_dst)
        except Exception as e:
            print(f"Download failed: {e}", file=sys.stderr)
            sys.exit(1)

    if args.checkpoint:
        src = Path(args.checkpoint)
        if not src.is_file():
            print(f"ERROR: checkpoint not found: {src}", file=sys.stderr)
            sys.exit(1)
        ckpt_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, ckpt_dst)
        print(f"Copied checkpoint -> {ckpt_dst}")

    if args.config:
        src = Path(args.config)
        if not src.is_file():
            print(f"ERROR: config not found: {src}", file=sys.stderr)
            sys.exit(1)
        cfg_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, cfg_dst)
        print(f"Copied config -> {cfg_dst}")

    ok, msg = checkpoint_ready(cfg)
    if ok:
        print(f"DetCLIPv2 ready: {msg}")
        sys.exit(0)

    print("DetCLIPv2 NOT ready (checkpoint hunt No-Go).", file=sys.stderr)
    print(msg, file=sys.stderr)
    print("See docs/DETCLIP_V2_CHECKPOINT_HUNT.md", file=sys.stderr)
    sys.exit(2 if args.verify_only else 1)


if __name__ == "__main__":
    main()
