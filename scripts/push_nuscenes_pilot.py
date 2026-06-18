"""Create/push nuscenes-pilot branch via GitHub Git Data API."""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path


def gh_api(method: str, endpoint: str, payload: dict | None = None) -> dict:
    cmd = ["gh", "api", "-X", method, endpoint]
    if payload is not None:
        cmd.extend(["--input", "-"])
        proc = subprocess.run(
            cmd,
            input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            capture_output=True,
            check=False,
        )
    else:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"gh api failed ({proc.returncode}): {proc.stderr.decode() or proc.stdout.decode()}"
        )
    out = proc.stdout.decode("utf-8").strip()
    return json.loads(out) if out else {}


def gh_query(endpoint: str, field: str) -> str:
    proc = subprocess.run(
        ["gh", "api", endpoint, "-q", field],
        capture_output=True,
        check=True,
        text=True,
    )
    return proc.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="wyc66-66/OVDeploy")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--branch", default="nuscenes-pilot")
    parser.add_argument(
        "--message",
        default="Add nuScenes-OVDeploy pilot for VCAD deployment metrics",
    )
    args = parser.parse_args()

    repo = args.repo
    parent_sha = gh_query(f"repos/{repo}/commits/main", ".sha")
    base_tree = gh_query(f"repos/{repo}/git/commits/{parent_sha}", ".tree.sha")
    print(f"Parent: {parent_sha}  Base tree: {base_tree}")

    rel_paths = [
        "README.md",
        "config/nuscenes_class_map.yaml",
        "config/nuscenes_pilot.yaml",
        "docs/NUSCENES_PILOT.md",
        "docs/NUSCENES_PILOT_SUMMARY.md",
        "ovdeploy/nuscenes/__init__.py",
        "ovdeploy/nuscenes/gt.py",
        "ovdeploy/nuscenes/infer.py",
        "ovdeploy/nuscenes/taxonomy.py",
        "reports/REPORT_nuscenes_main.json",
        "scripts/build_nuscenes_episodes.py",
        "scripts/plot_nuscenes_pilot.py",
        "scripts/run_nuscenes_eval.py",
        "scripts/wsl_run_nuscenes_pilot.sh",
        "scripts/wsl_run_nuscenes_sweep.sh",
    ]

    tree_items = []
    for rel in rel_paths:
        full = args.base_dir / rel
        if not full.exists():
            raise FileNotFoundError(full)
        blob = gh_api(
            "POST",
            f"repos/{repo}/git/blobs",
            {
                "content": base64.b64encode(full.read_bytes()).decode("ascii"),
                "encoding": "base64",
            },
        )
        print(f"Blob {rel}: {blob['sha'][:8]}")
        tree_items.append(
            {"path": rel, "mode": "100644", "type": "blob", "sha": blob["sha"]}
        )

    tree = gh_api(
        "POST",
        f"repos/{repo}/git/trees",
        {"base_tree": base_tree, "tree": tree_items},
    )
    print(f"New tree: {tree['sha']}")

    commit = gh_api(
        "POST",
        f"repos/{repo}/git/commits",
        {"message": args.message, "tree": tree["sha"], "parents": [parent_sha]},
    )
    print(f"New commit: {commit['sha']}")

    ref = f"heads/{args.branch}"
    try:
        gh_api("PATCH", f"repos/{repo}/git/refs/{ref}", {"sha": commit["sha"], "force": True})
        print(f"Updated {args.branch} -> {commit['sha']}")
    except RuntimeError:
        gh_api("POST", f"repos/{repo}/git/refs", {"ref": f"refs/{ref}", "sha": commit["sha"]})
        print(f"Created {args.branch} -> {commit['sha']}")

    print(f"https://github.com/{repo}/tree/{args.branch}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
