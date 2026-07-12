#!/usr/bin/env python3
"""Automated DetCLIPv2-T checkpoint hunt across public channels + local scan."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HUNT_LOG = ROOT / "reports" / "detclip_v2_hunt_log.json"
URL_RE = re.compile(
    r"https?://[^\s\"'<>]+(?:github\.com[^\s\"'<>]*|drive\.google[^\s\"'<>]*|"
    r"pan\.baidu[^\s\"'<>]*|openmmlab[^\s\"'<>]*|huggingface\.co[^\s\"'<>]*|"
    r"modelscope\.cn[^\s\"'<>]*|\.pth[^\s\"'<>]*)",
    re.IGNORECASE,
)
PTH_NAME_RE = re.compile(r"detclip|DetCLIP", re.IGNORECASE)

HF_QUERIES = [
    "DetCLIP",
    "DetCLIPv2",
    "detclipv2",
    "detclip_v2",
    "lewei yao detclip",
]
GITHUB_QUERIES = [
    "detclipv2",
    "DetCLIPv2",
    "detclip lvis",
    "leweiyao detclip",
    "DetCLIP swin",
]
MODELSCOPE_QUERIES = ["DetCLIP", "DetCLIPv2", "detclipv2"]
OPENXLAB_QUERIES = ["DetCLIP", "DetCLIPv2"]
PWC_SLUGS = [
    "detclip-v2-scalable-open-vocabulary-object-detection-pre-training-via-word-region-alignment",
    "detclip-v3-towards-versatile-generative-open-vocabulary-object-detection",
]
PDF_SOURCES = [
    "https://arxiv.org/pdf/2304.04514.pdf",
    "https://openaccess.thecvf.com/content/CVPR2023/papers/Yao_DetCLIPv2_Scalable_Open-Vocabulary_Object_Detection_Pre-Training_via_Word-Region_Alignment_CVPR_2023_paper.pdf",
    "https://arxiv.org/pdf/2404.09216.pdf",
]
AWESOME_OVD_RAW = (
    "https://raw.githubusercontent.com/witnessai/Awesome-Open-Vocabulary-Object-Detection/main/README.md"
)


def _http_get(url: str, timeout: int = 25, headers: dict | None = None) -> tuple[int, str]:
    hdrs = {
        "User-Agent": "OVDeploy-DetCLIP-Hunt/1.0",
        "Accept": "application/json,text/html,*/*",
    }
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            try:
                return resp.status, data.decode("utf-8", errors="replace")
            except Exception:
                return resp.status, data.decode("latin-1", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        return e.code, body
    except Exception as e:
        return 0, str(e)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _channel(name: str) -> dict[str, Any]:
    return {"channel": name, "status": "pending", "hits": [], "error": None}


def hunt_huggingface(endpoint: str | None) -> dict[str, Any]:
    ch = _channel("huggingface")
    base = (endpoint or os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
    hits: list[dict] = []
    for q in HF_QUERIES:
        url = f"{base}/api/models?search={urllib.parse.quote(q)}&limit=20"
        code, body = _http_get(url)
        if code != 200:
            ch["error"] = f"HF API {code} for {q}"
            continue
        try:
            models = json.loads(body)
        except json.JSONDecodeError:
            continue
        for m in models:
            mid = m.get("modelId") or m.get("id") or ""
            if not mid or "detclip" not in mid.lower():
                continue
            hits.append(
                {
                    "type": "hf_model",
                    "id": mid,
                    "url": f"{base}/{mid}",
                    "query": q,
                }
            )
    ch["hits"] = hits
    ch["status"] = "ok" if hits else "empty"
    return ch


def hunt_modelscope() -> dict[str, Any]:
    ch = _channel("modelscope")
    hits: list[dict] = []
    for q in MODELSCOPE_QUERIES:
        url = (
            "https://www.modelscope.cn/api/v1/dolphin/models?"
            f"PageSize=20&PageNumber=1&Search={urllib.parse.quote(q)}"
        )
        code, body = _http_get(url)
        if code != 200:
            ch["error"] = f"ModelScope API {code}"
            continue
        try:
            data = json.loads(body)
            items = data.get("Data", {}).get("Models") or data.get("data", {}).get("models") or []
        except json.JSONDecodeError:
            items = []
        for m in items:
            name = m.get("Name") or m.get("name") or ""
            if name:
                hits.append(
                    {
                        "type": "modelscope",
                        "id": name,
                        "url": f"https://www.modelscope.cn/models/{name}",
                        "query": q,
                    }
                )
    ch["hits"] = hits
    ch["status"] = "ok" if hits else "empty"
    return ch


def hunt_openxlab() -> dict[str, Any]:
    ch = _channel("openxlab")
    hits: list[dict] = []
    for q in OPENXLAB_QUERIES:
        url = f"https://openxlab.org.cn/models/api/v1/models?keyword={urllib.parse.quote(q)}&page=1&page_size=20"
        code, body = _http_get(url)
        if code != 200:
            ch["error"] = f"OpenXLab API {code}"
            continue
        try:
            data = json.loads(body)
            items = data.get("data", {}).get("list") or data.get("data") or []
        except json.JSONDecodeError:
            items = []
        if isinstance(items, dict):
            items = items.get("list") or []
        for m in items:
            name = m.get("name") or m.get("model_name") or ""
            if name:
                hits.append(
                    {
                        "type": "openxlab",
                        "id": name,
                        "url": f"https://openxlab.org.cn/models/{name}",
                        "query": q,
                    }
                )
    ch["hits"] = hits
    ch["status"] = "ok" if hits else "empty"
    return ch


def hunt_github() -> dict[str, Any]:
    ch = _channel("github")
    hits: list[dict] = []
    for q in GITHUB_QUERIES:
        url = (
            "https://api.github.com/search/repositories?"
            f"q={urllib.parse.quote(q)}&sort=updated&per_page=15"
        )
        code, body = _http_get(url)
        if code != 200:
            ch["error"] = f"GitHub API {code}"
            continue
        try:
            repos = json.loads(body).get("items") or []
        except json.JSONDecodeError:
            repos = []
        for r in repos:
            full = r.get("full_name") or ""
            if not full:
                continue
            hits.append(
                {
                    "type": "github_repo",
                    "id": full,
                    "url": r.get("html_url") or f"https://github.com/{full}",
                    "query": q,
                    "description": (r.get("description") or "")[:200],
                }
            )
    ch["hits"] = hits
    ch["status"] = "ok" if hits else "empty"
    return ch


def hunt_papers_with_code() -> dict[str, Any]:
    ch = _channel("papers_with_code")
    hits: list[dict] = []
    for slug in PWC_SLUGS:
        url = f"https://paperswithcode.com/paper/{slug}"
        code, body = _http_get(url)
        if code != 200:
            continue
        for m in URL_RE.findall(body):
            hits.append({"type": "pwc_url", "url": m.rstrip(").,;"), "paper": slug})
        if "Official implementation" in body or "github.com" in body:
            hits.append({"type": "pwc_page", "url": url, "paper": slug})
    ch["hits"] = hits
    ch["status"] = "ok" if hits else "empty"
    return ch


def hunt_pdf_urls() -> dict[str, Any]:
    ch = _channel("pdf_url_extract")
    hits: list[dict] = []
    for pdf_url in PDF_SOURCES:
        code, body = _http_get(pdf_url, timeout=40)
        if code != 200:
            hits.append({"type": "pdf_fetch_failed", "url": pdf_url, "code": code})
            continue
        for m in URL_RE.findall(body):
            clean = m.rstrip(").,;")
            hits.append({"type": "pdf_url", "source": pdf_url, "url": clean})
    ch["hits"] = hits
    ch["status"] = "ok" if any(h.get("type") == "pdf_url" for h in hits) else "empty"
    return ch


def hunt_awesome_ovd() -> dict[str, Any]:
    ch = _channel("awesome_ovd_readme")
    code, body = _http_get(AWESOME_OVD_RAW)
    hits: list[dict] = []
    if code == 200:
        for m in URL_RE.findall(body):
            if "detclip" in m.lower() or "DetCLIP" in m:
                hits.append({"type": "awesome_ovd_url", "url": m.rstrip(").,;")})
        if "DetCLIPv2" in body and not hits:
            hits.append({"type": "awesome_ovd_mention", "url": AWESOME_OVD_RAW})
    else:
        ch["error"] = f"fetch {code}"
    ch["hits"] = hits
    ch["status"] = "ok" if hits else "empty"
    return ch


def _scan_dir(root: Path, max_files: int = 800) -> list[dict]:
    found: list[dict] = []
    if not root.exists():
        return found
    skip_parts = {
        "docs",
        "reports",
        "scripts",
        "paper",
        ".git",
        "ovdeploy-public",
        "node_modules",
    }
    count = 0
    try:
        for p in root.rglob("*"):
            if count >= max_files:
                break
            if not p.is_file():
                continue
            parts = {x.lower() for x in p.parts}
            if parts & skip_parts:
                continue
            name = p.name
            low_path = str(p).lower()
            is_ckpt = name.endswith(".pth") and (
                PTH_NAME_RE.search(name) or "detclip" in low_path
            )
            is_cfg = name.endswith(".py") and "detclip" in low_path and "config" in low_path
            if not (is_ckpt or is_cfg):
                continue
            try:
                size = p.stat().st_size
                sha = _sha256_file(p) if size < 2_000_000_000 else "skipped_large"
            except OSError:
                size, sha = -1, "unreadable"
            found.append(
                {
                    "type": "local_file",
                    "path": str(p),
                    "size_bytes": size,
                    "sha256": sha,
                }
            )
            count += 1
    except (OSError, PermissionError):
        pass
    return found


def hunt_local() -> dict[str, Any]:
    ch = _channel("local_scan")
    roots = [
        ROOT,
        Path(r"d:\ccfa"),
        Path.home() / "Downloads",
        Path(r"\\wsl.localhost\Ubuntu\home\a"),
        Path(r"\\wsl$\Ubuntu\home\a"),
        Path("/home/a"),
        Path.home() / "miniconda3",
    ]
    hits: list[dict] = []
    for r in roots:
        hits.extend(_scan_dir(r))
    # Also check configured checkpoint path
    try:
        from ovdeploy.backends.detclip import checkpoint_ready
        from ovdeploy.paths_util import load_paths

        ok, msg = checkpoint_ready(load_paths())
        if ok:
            p = Path(msg)
            hits.append(
                {
                    "type": "configured_checkpoint",
                    "path": str(p),
                    "size_bytes": p.stat().st_size,
                    "sha256": _sha256_file(p),
                }
            )
    except Exception:
        pass
    ch["hits"] = hits
    ch["status"] = "found" if hits else "empty"
    return ch


def hunt_configured_paths() -> dict[str, Any]:
    ch = _channel("configured_paths")
    hits: list[dict] = []
    try:
        from ovdeploy.backends.detclip import _resolve_paths
        from ovdeploy.paths_util import load_paths

        ckpt, mcfg, third = _resolve_paths(load_paths())
        for label, p in [("checkpoint", ckpt), ("config", mcfg)]:
            hits.append(
                {
                    "type": label,
                    "path": str(p),
                    "exists": p.is_file(),
                }
            )
        if third.is_dir():
            for p in third.rglob("*.pth"):
                hits.append({"type": "third_party_pth", "path": str(p)})
    except Exception as e:
        ch["error"] = str(e)
    ch["hits"] = hits
    ch["status"] = "ready" if any(
        h.get("exists") for h in hits if h.get("type") in ("checkpoint", "config")
    ) else "missing"
    return ch


def pick_download_urls(channels: list[dict]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    skip_hosts = ("discuss.huggingface.co", "cdn-avatars.huggingface.co", "cdn-uploads.huggingface.co")
    for ch in channels:
        for h in ch.get("hits") or []:
            u = h.get("url") or ""
            if not u or u in seen:
                continue
            low = u.lower()
            if any(s in low for s in skip_hosts):
                continue
            if not (
                ".pth" in low
                or "drive.google" in low
                or ("huggingface.co" in low and "/models/" in low)
                or ("modelscope.cn" in low and "/models/" in low)
                or "openmmlab" in low
                or "pan.baidu" in low
            ):
                continue
            if "detclip" not in low and ".pth" not in low and "drive.google" not in low:
                continue
            seen.add(u)
            urls.append(u)
    return urls


def build_log(channels: list[dict]) -> dict[str, Any]:
    local = next((c for c in channels if c["channel"] == "local_scan"), {})
    configured = next((c for c in channels if c["channel"] == "configured_paths"), {})
    local_hits = local.get("hits") or []
    cfg_ready = configured.get("status") == "ready"
    download_urls = pick_download_urls(channels)

    if cfg_ready:
        verdict = "FOUND"
        reason = "checkpoint and config on disk"
    elif local_hits:
        verdict = "PARTIAL"
        reason = "local detclip checkpoint/config candidate; verify paths"
    elif download_urls:
        verdict = "URL_CANDIDATES"
        reason = f"{len(download_urls)} download URL(s) to try"
    else:
        verdict = "NOT_FOUND"
        reason = "no public checkpoint; author contact recommended"

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "reason": reason,
        "download_urls": download_urls,
        "local_hits": local_hits,
        "channels": channels,
        "author_contact": "docs/templates/detclip_author_request_en.md",
    }


def main() -> None:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
    )
    parser.add_argument("--out", default=str(HUNT_LOG))
    args = parser.parse_args()

    print("DetCLIPv2 checkpoint hunt...", flush=True)
    channels = [
        hunt_huggingface(args.hf_endpoint),
        hunt_modelscope(),
        hunt_openxlab(),
        hunt_github(),
        hunt_papers_with_code(),
        hunt_pdf_urls(),
        hunt_awesome_ovd(),
        hunt_local(),
        hunt_configured_paths(),
    ]
    log = build_log(channels)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"Verdict: {log['verdict']} — {log['reason']}")
    print(f"Wrote {out}")
    if log["download_urls"]:
        print("Download URL candidates:")
        for u in log["download_urls"][:5]:
            print(f"  {u}")
    if log["local_hits"]:
        print("Local hits:")
        for h in log["local_hits"][:5]:
            print(f"  {h.get('path', h)}")

    sys.exit(0 if log["verdict"] == "FOUND" else 1)


if __name__ == "__main__":
    main()
