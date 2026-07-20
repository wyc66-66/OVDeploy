"""Quick smoke test for YOLO-S / YOLO-M backends."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ovdeploy.infer import smoke_two_images

def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="yolo")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--max-images", type=int, default=2)
    args = p.parse_args()
    eps = sorted((ROOT / "data/episodes/dev/dev_v10_s42_none").glob("*.json"))
    if not eps:
        sys.exit("no episodes")
    from ovdeploy.episode import load_episode
    from ovdeploy.infer import run_episode_infer
    from ovdeploy.paths_util import load_lvis_minival

    ep = load_episode(eps[0])
    lvis = load_lvis_minival()
    device = "cuda:0"
    if args.gpu:
        try:
            import torch

            if not torch.cuda.is_available():
                print("WARN: --gpu set but CUDA unavailable; use WSL bash scripts/wsl_preflight_gpu.sh")
                sys.exit(1)
        except ImportError:
            sys.exit(1)
    else:
        print("WARN: running without --gpu is for smoke only; not valid for main reports")
        device = "cpu"
    r = run_episode_infer(
        ep, "B5_subset", lvis, max_images=args.max_images, device=device, backbone=args.backbone
    )
    print(r)

if __name__ == "__main__":
    main()
