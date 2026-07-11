"""Qualitative OOV-FP figure from B0 cache + episode JSON."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _resolve_val_image(yolo: Path, cfg: dict, file_name: str) -> Path | None:
    fn = file_name.replace("\\", "/")
    img_dir = yolo / cfg["data"]["val2017_dir"]
    candidates = (
        img_dir / Path(fn).name,
        yolo / "data/coco" / fn,
        img_dir / fn,
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def _try_panel(
    iid: int,
    vocab: set[int],
    yolo: Path,
    cfg: dict,
    id_to_im: dict,
    gt_by_img: dict,
    axes,
    shown: int,
    patches,
    load_b0_preds,
    imread_bgr,
) -> int:
    preds = load_b0_preds(iid, "yolo")
    if not preds:
        return shown
    oov = [
        p
        for p in preds
        if p.get("score", 0) >= 0.5 and p.get("category_id") not in vocab
    ]
    if len(oov) < 2:
        return shown
    im = id_to_im.get(iid)
    if not im:
        return shown
    path = _resolve_val_image(yolo, cfg, im["file_name"])
    if path is None:
        return shown
    bgr = imread_bgr(path)
    if bgr is None:
        return shown
    rgb = bgr[:, :, [2, 1, 0]]
    ax = axes[shown]
    ax.imshow(rgb)
    for g in gt_by_img.get(iid, []):
        if g["category_id"] in vocab:
            x, y, w, h = g["bbox"]
            ax.add_patch(
                patches.Rectangle(
                    (x, y), w, h, linewidth=1.5, edgecolor="#55A868", facecolor="none"
                )
            )
    for p in oov[:8]:
        x1, y1, x2, y2 = p["bbox"]
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=1.2,
                edgecolor="#C44E52",
                facecolor="none",
            )
        )
    ax.set_title(f"img {iid} (|V|={len(vocab)})", fontsize=8)
    return shown + 1


def main() -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("requires matplotlib")
        return

    from ovdeploy.b0_cache import load_b0_preds
    from ovdeploy.episode import load_episode
    from ovdeploy.image_io import imread_bgr
    from ovdeploy.paths_util import load_lvis_minival, load_paths

    cfg = load_paths()
    yolo = cfg["_yolo"]
    lvis = load_lvis_minival()
    id_to_im = {im["id"]: im for im in lvis["images"]}
    gt_by_img: dict[int, list] = {}
    for a in lvis["annotations"]:
        gt_by_img.setdefault(a["image_id"], []).append(a)

    ep_dir = ROOT / "data/episodes/dev/dev_v10_s42_none"
    eps = sorted(ep_dir.glob("*.json"))
    if not eps:
        print("no dev_v10 episodes")
        return

    fig_dir = ROOT / "paper/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6))
    for ax in axes:
        ax.axis("off")

    shown = 0
    seen_iids: set[int] = set()
    for ep_path in eps:
        if shown >= 3:
            break
        ep = load_episode(ep_path)
        vocab = set(ep.vocab.cat_ids)
        for iid in ep.image_ids:
            if shown >= 3 or iid in seen_iids:
                continue
            seen_iids.add(iid)
            shown = _try_panel(
                iid,
                vocab,
                yolo,
                cfg,
                id_to_im,
                gt_by_img,
                axes,
                shown,
                patches,
                load_b0_preds,
                imread_bgr,
            )

    if shown == 0:
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4)
        ax.axis("off")
        ax.text(
            5,
            3.2,
            "OOV-FP qualitative (schema)",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
        ax.add_patch(
            patches.Rectangle((0.5, 0.5), 4, 2.2, linewidth=1.5, edgecolor="gray", facecolor="#f3f4f6")
        )
        ax.add_patch(
            patches.Rectangle((1, 1.2), 1.2, 0.9, linewidth=1.5, edgecolor="#55A868", facecolor="none")
        )
        ax.add_patch(
            patches.Rectangle((2.5, 1.0), 1.0, 0.7, linewidth=1.2, edgecolor="#C44E52", facecolor="none")
        )
        ax.add_patch(
            patches.Rectangle((3.0, 1.9), 0.8, 0.5, linewidth=1.2, edgecolor="#C44E52", facecolor="none")
        )
        ax.text(5, 0.2, "Green = in-vocab GT; Red = B0 preds outside $V_e$", ha="center", fontsize=8)
    else:
        fig.suptitle("Deployment gap: B0 OOV (red) vs in-vocab GT (green)", fontsize=9)
    plt.tight_layout()
    out = fig_dir / "fig_oov_qualitative.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out} ({shown} panels)")

    _write_triptych_schema(fig_dir, patches)


def _write_triptych_schema(fig_dir: Path, patches) -> None:
    """Schematic B0 / B5 / VG_full comparison for advisor narrative."""
    import json
    import matplotlib.pyplot as plt

    vg_report = ROOT.parent / "submission-b" / "reports" / "REPORT_VG_dev_main.json"
    oov_b0, oov_vg = "66%", "0.5%"
    if vg_report.is_file():
        data = json.loads(vg_report.read_text(encoding="utf-8"))
        for r in data.get("rows", []):
            if r.get("method") == "B5_subset" and r.get("config") == "dev_v10_s42_none":
                oov_b0 = f"{r.get('OOV_FP_mean', 0.664) * 100:.1f}%"
            if r.get("method") == "VG_full" and r.get("config") == "dev_v10_s42_none":
                oov_vg = f"{r.get('OOV_FP_mean', 0.005) * 100:.1f}%"

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.8))
    titles = [
        f"B0 full-vocab\nOOV-FP ≈ {oov_b0}",
        "B5 subset\nEpiAP ↑, OOV unchanged",
        f"VG Router+Guard\nguarded B0 OOV ≈ {oov_vg}",
    ]
    colors = ["#C44E52", "#4C72B0", "#55A868"]
    for ax, title, color in zip(axes, titles, colors):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.add_patch(patches.Rectangle((1, 2), 8, 6, linewidth=1.2, edgecolor="gray", facecolor="#f8f8f8"))
        ax.add_patch(patches.Rectangle((2, 4), 2, 2, linewidth=1.5, edgecolor="#55A868", facecolor="none"))
        ax.add_patch(patches.Rectangle((5, 3.5), 2.5, 1.8, linewidth=1.2, edgecolor=color, facecolor="none"))
        if "Guard" in title:
            ax.text(6.2, 3.2, "×", color=color, fontsize=14, fontweight="bold")
        ax.set_title(title, fontsize=9)
    fig.suptitle("Same episode protocol: audit path closes OOV without redefining metrics", fontsize=9)
    plt.tight_layout()
    out = fig_dir / "fig_deploy_triptych.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
