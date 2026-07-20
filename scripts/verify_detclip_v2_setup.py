#!/usr/bin/env python3
"""Smoke-test DetCLIPv2-T setup (single LVIS image, small vocab)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from ovdeploy.backends.detclip import checkpoint_ready, DetclipCheckpointError
    from ovdeploy.paths_util import load_paths, load_lvis_minival, load_class_texts, cat_id_to_index

    cfg = load_paths()
    ok, msg = checkpoint_ready(cfg)
    if not ok:
        print(f"BLOCKED: {msg}")
        print("See docs/DETCLIP_V2_CHECKPOINT_HUNT.md")
        sys.exit(2)

    device = args.device
    if args.gpu:
        import torch

        if not torch.cuda.is_available():
            print("CUDA not available")
            sys.exit(1)
    else:
        device = "cpu"

    lvis = load_lvis_minival()
    class_texts = load_class_texts(cfg)
    cid2idx = cat_id_to_index(lvis)
    cat_ids = [c["id"] for c in lvis["categories"][:10]]
    texts = []
    for c in cat_ids:
        raw = class_texts[cid2idx[c]]
        texts.append(str(raw[0]) if isinstance(raw, list) and raw else str(raw))

    img_id = lvis["images"][0]["id"]
    file_name = lvis["images"][0]["file_name"]
    from ovdeploy.backends.detclip import DetclipV2Backend
    import cv2

    backend = DetclipV2Backend(device=device)
    p = backend.image_path(file_name)
    bgr = cv2.imread(str(p))
    if bgr is None:
        print(f"Cannot read image: {p}")
        sys.exit(1)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    try:
        preds = backend.predict(
            rgb,
            texts,
            cat_ids,
            img_id,
            class_texts_raw=class_texts,
            cid2idx=cid2idx,
        )
    except DetclipCheckpointError as e:
        print(f"BLOCKED: {e}")
        sys.exit(2)
    except Exception as e:
        print(f"FAIL: {e}")
        sys.exit(1)

    print(f"PASS: {len(preds)} preds on {file_name}")
    if preds:
        print(f"  sample: score={preds[0]['score']:.3f} cat={preds[0]['category_id']}")
    sys.exit(0)


if __name__ == "__main__":
    main()
