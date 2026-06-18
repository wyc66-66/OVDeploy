"""Extended YOLO backend wrapper with custom class texts for PromptAlign."""
from __future__ import annotations

from typing import Any

import numpy as np

from ovdeploy.paths_util import cat_id_to_index, load_class_texts


def get_backend(device: str = "cuda:0"):
    from vocabguard.backend import get_extended_backend

    backend = get_extended_backend("yolo", device=device)
    return YoloPromptBackend(backend)


class YoloPromptBackend:
    """Wraps YoloWorldExtended with predict_with_texts."""

    name = "yolo_prompt"

    def __init__(self, inner) -> None:
        self._inner = inner
        self.device = inner.device

    def image_path(self, file_name: str):
        return self._inner.image_path(file_name)

    def predict(
        self,
        image_rgb: np.ndarray,
        vocab_ids: list[int],
        image_id: int,
        lvis: dict,
        **kwargs,
    ) -> list[dict]:
        return self._inner.predict(image_rgb, vocab_ids, image_id, lvis, **kwargs)

    def predict_with_texts(
        self,
        image_rgb: np.ndarray,
        vocab_ids: list[int],
        image_id: int,
        lvis: dict,
        texts_sub: list,
    ) -> list[dict]:
        import torch

        from ovdeploy.paths_util import load_paths
        from ovdeploy.vocab import subset_class_texts

        class_names, class_texts_raw = load_class_texts()
        cid2idx = cat_id_to_index(lvis)
        sub_names = [
            class_names[cid2idx[c]] if c in cid2idx else str(c) for c in vocab_ids
        ]

        model, test_pipeline = self._inner._init_model()
        self._inner._FEAT_HOOK = getattr(self._inner, "_FEAT_HOOK", None)
        from vocabguard.backend import yolo_ext as ymod

        ymod._FEAT_HOOK["feat"].clear()
        self._inner._register_feat_hook(model)

        data_info = dict(img=image_rgb, img_id=image_id, texts=texts_sub)
        data_info = test_pipeline(data_info)
        data_batch = dict(
            inputs=data_info["inputs"].unsqueeze(0),
            data_samples=[data_info["data_samples"]],
        )
        with torch.no_grad():
            out = model.test_step(data_batch)[0]
        pred = out.pred_instances

        preds = []
        for b, s, lb in zip(
            pred.bboxes.cpu().numpy(), pred.scores.cpu().numpy(), pred.labels.cpu().numpy()
        ):
            li = int(lb)
            cid = vocab_ids[li] if li < len(vocab_ids) else -1
            x1, y1, x2, y2 = [float(v) for v in b]
            bbox_xywh = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
            preds.append(
                {
                    "bbox": bbox_xywh,
                    "score": float(s),
                    "label_idx": li,
                    "category_id": cid,
                    "category_name": sub_names[li] if li < len(sub_names) else "",
                }
            )
        return preds

    def score_classes(
        self,
        image_rgb: np.ndarray,
        cat_ids: list[int],
        image_id: int,
        lvis: dict,
    ) -> dict[int, float]:
        return self._inner.score_classes(image_rgb, cat_ids, image_id, lvis)

    def encode_image(self, image_rgb: np.ndarray, image_id: int, lvis: dict) -> np.ndarray:
        return self._inner.encode_image(image_rgb, image_id, lvis)
