"""Download HuggingFace zero-shot detection model (OWL-ViT) to local weights/."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ovdeploy.paths_util import load_paths


def _has_model_files(d: Path) -> bool:
    if not d.is_dir():
        return False
    names = {p.name for p in d.iterdir()}
    return bool(names & {"config.json", "model.safetensors", "pytorch_model.bin"})


def download(model_id: str, local_dir: Path, endpoint: str | None) -> Path:
    if _has_model_files(local_dir):
        print(f"SKIP download: model already at {local_dir}")
        return local_dir

    local_dir.mkdir(parents=True, exist_ok=True)
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
        os.environ["HUGGINGFACE_HUB_ENDPOINT"] = endpoint

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        print(f"OK snapshot_download -> {local_dir}")
        return local_dir
    except Exception as e:
        print(f"snapshot_download failed: {e}")

    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "huggingface_hub.cli.huggingface_cli",
        "download",
        model_id,
        "--local-dir",
        str(local_dir),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0 and _has_model_files(local_dir):
        print(f"OK huggingface-cli -> {local_dir}")
        return local_dir
    print(r.stdout)
    print(r.stderr)
    raise RuntimeError(
        f"Could not download {model_id}.\n"
        f"Manual: browser https://huggingface.co/{model_id} -> extract to {local_dir}\n"
        f"Or WSL: HF_ENDPOINT=https://hf-mirror.com python scripts/download_hf_model.py"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default=None)
    p.add_argument("--local-dir", default=None)
    p.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
    )
    args = p.parse_args()

    cfg = load_paths()
    glip = cfg.get("glip", {})
    model_id = args.model_id or glip.get("model_id", "google/owlvit-base-patch32")
    rel = args.local_dir or glip.get("local_dir", "weights/owlvit-base-patch32")
    local_dir = Path(rel) if Path(rel).is_absolute() else ROOT / rel
    download(model_id, local_dir, args.endpoint)


if __name__ == "__main__":
    main()
