"""Ground truth: project nuScenes 3D boxes to CAM_FRONT 2D [x,y,w,h]."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ovdeploy.nuscenes.taxonomy import NuScenesTaxonomy, load_taxonomy


def sample_token_to_image_id(sample_token: str) -> int:
    h = hashlib.sha256(sample_token.encode()).hexdigest()[:12]
    return int(h, 16) % 1_000_000_000


def project_box_to_image(nusc: Any, ann_token: str, cam_sd_token: str) -> list[float] | None:
    from nuscenes.utils.geometry_utils import box_in_image, view_points
    from pyquaternion import Quaternion

    cam_sd = nusc.get("sample_data", cam_sd_token)
    cs_record = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", cam_sd["ego_pose_token"])
    cam_intrinsic = np.array(cs_record["camera_intrinsic"])
    imsize = (int(cam_sd["width"]), int(cam_sd["height"]))

    box = nusc.get_box(ann_token)
    box.translate(-np.array(pose_record["translation"]))
    box.rotate(Quaternion(pose_record["rotation"]).inverse)
    box.translate(-np.array(cs_record["translation"]))
    box.rotate(Quaternion(cs_record["rotation"]).inverse)

    if not box_in_image(box, cam_intrinsic, imsize, vis_level=0):
        return None

    corners = box.corners()
    in_front = np.argwhere(corners[2, :] > 0.1).flatten()
    if len(in_front) == 0:
        return None
    corners = corners[:, in_front]
    corners_2d = view_points(corners, cam_intrinsic, normalize=True)[:2, :]
    xmin = float(np.min(corners_2d[0, :]))
    xmax = float(np.max(corners_2d[0, :]))
    ymin = float(np.min(corners_2d[1, :]))
    ymax = float(np.max(corners_2d[1, :]))
    w, h = imsize
    xmin = max(0.0, xmin)
    ymin = max(0.0, ymin)
    xmax = min(float(w), xmax)
    ymax = min(float(h), ymax)
    bw = xmax - xmin
    bh = ymax - ymin
    if bw < 2 or bh < 2:
        return None
    return [xmin, ymin, bw, bh]


class NuScenesGT:
    """Index GT boxes by image_id (derived from sample_token)."""

    def __init__(
        self,
        nuscenes_root: Path,
        version: str = "v1.0-mini",
        camera: str = "CAM_FRONT",
        taxonomy: NuScenesTaxonomy | None = None,
        verbose: bool = False,
    ):
        from nuscenes.nuscenes import NuScenes

        self.nusc = NuScenes(version=version, dataroot=str(nuscenes_root), verbose=verbose)
        self.camera = camera
        self.taxonomy = taxonomy or load_taxonomy()
        self.gt_by_image_id: dict[int, dict[str, list]] = defaultdict(
            lambda: {"boxes": [], "cat_ids": []}
        )
        self.sample_token_to_image_id: dict[str, int] = {}
        self.image_id_to_sample_token: dict[int, str] = {}
        self.image_id_to_path: dict[int, str] = {}
        self._build_index()

    def _build_index(self) -> None:
        for sample in self.nusc.sample:
            cam_token = sample["data"].get(self.camera)
            if not cam_token:
                continue
            iid = sample_token_to_image_id(sample["token"])
            self.sample_token_to_image_id[sample["token"]] = iid
            self.image_id_to_sample_token[iid] = sample["token"]
            sd = self.nusc.get("sample_data", cam_token)
            self.image_id_to_path[iid] = str(
                Path(self.nusc.dataroot) / sd["filename"]
            )

            for ann_token in sample["anns"]:
                ann = self.nusc.get("sample_annotation", ann_token)
                if "category_name" in ann:
                    cat_name = ann["category_name"]
                else:
                    cat = self.nusc.get("category", ann["category_token"])
                    cat_name = cat["name"]
                cid = self.taxonomy.map_category_name(cat_name)
                if cid is None:
                    continue
                bbox = project_box_to_image(self.nusc, ann_token, cam_token)
                if bbox is None:
                    continue
                self.gt_by_image_id[iid]["boxes"].append(bbox)
                self.gt_by_image_id[iid]["cat_ids"].append(cid)

    def image_gt_cat_ids(self) -> dict[int, set[int]]:
        out: dict[int, set[int]] = {}
        for iid, gt in self.gt_by_image_id.items():
            out[iid] = set(gt["cat_ids"])
        return out

    def cam_front_samples_by_scene(self) -> dict[str, list[tuple[str, str, str]]]:
        """scene_name -> [(sample_token, cam_sd_token, file_path), ...]"""
        out: dict[str, list[tuple[str, str, str]]] = {}
        for scene in self.nusc.scene:
            name = scene["name"]
            rows: list[tuple[str, str, str]] = []
            token = scene["first_sample_token"]
            while token:
                sample = self.nusc.get("sample", token)
                cam_token = sample["data"].get(self.camera)
                if cam_token:
                    sd = self.nusc.get("sample_data", cam_token)
                    path = str(Path(self.nusc.dataroot) / sd["filename"])
                    rows.append((sample["token"], cam_token, path))
                if token == scene["last_sample_token"]:
                    break
                token = sample["next"]
            if rows:
                out[name] = rows
        return out

    def get_gt(self, image_id: int) -> dict[str, list]:
        return self.gt_by_image_id.get(image_id, {"boxes": [], "cat_ids": []})
