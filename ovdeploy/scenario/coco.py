"""Unified COCO loader for DSP domains (ODinW, VisDrone, SKU-110K)."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_slug(slug: str) -> str:
    return slug.lower().strip()


def load_scenario_coco(slug: str) -> dict[str, Any]:
    s = resolve_slug(slug)
    if s == "visdrone":
        from ovdeploy.visdrone.loader import load_visdrone_coco

        return load_visdrone_coco()
    if s in ("sku110k", "sku-110k"):
        from ovdeploy.sku110k.loader import load_sku110k_coco

        return load_sku110k_coco()
    from ovdeploy.odinw_loader import load_odinw_coco

    return load_odinw_coco(s)


def load_scenario_meta(slug: str) -> dict[str, Any]:
    s = resolve_slug(slug)
    if s == "visdrone":
        from ovdeploy.visdrone.loader import load_domain_meta

        return load_domain_meta()
    if s in ("sku110k", "sku-110k"):
        from ovdeploy.sku110k.loader import load_domain_meta

        return load_domain_meta()
    from ovdeploy.odinw_loader import load_domain_meta

    return load_domain_meta(s)


def scenario_image_path(slug: str, file_name: str) -> Path:
    s = resolve_slug(slug)
    if s == "visdrone":
        from ovdeploy.visdrone.loader import image_path

        return image_path(file_name)
    if s in ("sku110k", "sku-110k"):
        from ovdeploy.sku110k.loader import image_path

        return image_path(file_name)
    from ovdeploy.odinw_loader import image_path

    return image_path(s, file_name)
