"""GLIP-T backend: native Microsoft GLIP (.pth) or Grounding-DINO transformers fallback."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ovdeploy.paths_util import load_paths, load_class_texts, cat_id_to_index
from ovdeploy.vocab import subset_class_texts

_MODEL_CACHE: dict[tuple[str, str], tuple[Any, ...]] = {}


def _gdino_prepare_vision_bundle(inner_model, pixel_values, pixel_mask):
    """Run Swin backbone + vision proj once; reuse across text chunks."""
    import torch
    import torch.nn.functional as F

    batch_size, _, height, width = pixel_values.shape
    device = pixel_values.device
    if pixel_mask is None:
        pixel_mask = torch.ones((batch_size, height, width), dtype=torch.long, device=device)

    vision_features, position_embeddings_list = inner_model.backbone(pixel_values, pixel_mask)
    feature_maps = []
    masks = []
    for level, (source, mask) in enumerate(vision_features):
        feature_maps.append(inner_model.input_proj_vision[level](source))
        masks.append(mask)

    if inner_model.config.num_feature_levels > len(feature_maps):
        _len_sources = len(feature_maps)
        for level in range(_len_sources, inner_model.config.num_feature_levels):
            if level == _len_sources:
                source = inner_model.input_proj_vision[level](vision_features[-1][0])
            else:
                source = inner_model.input_proj_vision[level](feature_maps[-1])
            mask = F.interpolate(pixel_mask[None].float(), size=source.shape[-2:]).to(torch.bool)[0]
            pos_l = inner_model.backbone.position_embedding(source, mask).to(source.dtype)
            feature_maps.append(source)
            masks.append(mask)
            position_embeddings_list.append(pos_l)

    source_flatten = []
    mask_flatten = []
    lvl_pos_embed_flatten = []
    spatial_shapes = []
    for level, (source, mask, pos_embed) in enumerate(
        zip(feature_maps, masks, position_embeddings_list)
    ):
        _, _, h, w = source.shape
        spatial_shapes.append((h, w))
        source_flatten.append(source.flatten(2).transpose(1, 2))
        mask_flatten.append(mask.flatten(1))
        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        lvl_pos_embed = pos_embed + inner_model.level_embed[level].view(1, 1, -1)
        lvl_pos_embed_flatten.append(lvl_pos_embed)
    source_flatten = torch.cat(source_flatten, 1)
    mask_flatten = torch.cat(mask_flatten, 1)
    lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
    spatial_shapes_t = torch.as_tensor(
        spatial_shapes, dtype=torch.long, device=source_flatten.device
    )
    level_start_index = torch.cat(
        (spatial_shapes_t.new_zeros((1,)), spatial_shapes_t.prod(1).cumsum(0)[:-1])
    )
    valid_ratios = torch.stack([inner_model.get_valid_ratio(m) for m in masks], 1).float()

    query_embeds = None
    if inner_model.config.embedding_init_target or inner_model.config.two_stage:
        query_embeds = inner_model.query_position_embeddings.weight

    return {
        "source_flatten": source_flatten,
        "mask_flatten": mask_flatten,
        "lvl_pos_embed_flatten": lvl_pos_embed_flatten,
        "spatial_shapes": spatial_shapes_t,
        "level_start_index": level_start_index,
        "valid_ratios": valid_ratios,
        "query_embeds": query_embeds,
        "batch_size": batch_size,
    }


def _gdino_forward_chunk(det_model, inner_model, vision_bundle, text_inputs, dev, use_autocast, text_bundle=None):
    """Encoder+decoder+heads for one text chunk with cached vision features."""
    import torch
    from transformers.models.grounding_dino.modeling_grounding_dino import (
        generate_masks_with_special_tokens_and_transfer_map,
    )

    if text_bundle is not None:
        input_ids = text_bundle["input_ids"]
        token_type_ids = text_bundle["token_type_ids"]
        text_self_attention_masks = text_bundle["text_self_attention_masks"]
        position_ids = text_bundle["position_ids"]
        text_token_mask = text_bundle["text_token_mask"]
    else:
        input_ids = text_inputs["input_ids"].to(dev)
        attention_mask = text_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(dev)
        token_type_ids = text_inputs.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        else:
            token_type_ids = token_type_ids.to(dev)

        text_self_attention_masks, position_ids = generate_masks_with_special_tokens_and_transfer_map(
            input_ids
        )
        text_token_mask = attention_mask.bool()
        max_text_len = inner_model.config.max_text_len
        if text_self_attention_masks.shape[1] > max_text_len:
            text_self_attention_masks = text_self_attention_masks[:, :max_text_len, :max_text_len]
            position_ids = position_ids[:, :max_text_len]
            input_ids = input_ids[:, :max_text_len]
            token_type_ids = token_type_ids[:, :max_text_len]
            text_token_mask = text_token_mask[:, :max_text_len]

    cached_text_features = (
        text_bundle.get("text_features") if text_bundle is not None else None
    )

    def _run():
        if cached_text_features is not None:
            text_features = cached_text_features
        else:
            text_outputs = inner_model.text_backbone(
                input_ids,
                text_self_attention_masks,
                token_type_ids,
                position_ids,
                return_dict=True,
            )
            text_features = inner_model.text_projection(text_outputs.last_hidden_state)
        encoder_outputs = inner_model.encoder(
            vision_features=vision_bundle["source_flatten"],
            vision_attention_mask=~vision_bundle["mask_flatten"],
            vision_position_embedding=vision_bundle["lvl_pos_embed_flatten"],
            spatial_shapes=vision_bundle["spatial_shapes"],
            level_start_index=vision_bundle["level_start_index"],
            valid_ratios=vision_bundle["valid_ratios"],
            text_features=text_features,
            text_attention_mask=~text_token_mask,
            text_position_embedding=None,
            text_self_attention_masks=~text_self_attention_masks,
            text_position_ids=position_ids,
            return_dict=True,
        )
        batch_size = vision_bundle["batch_size"]
        query_embeds = vision_bundle["query_embeds"]
        mask_flatten = vision_bundle["mask_flatten"]
        spatial_shapes = vision_bundle["spatial_shapes"]

        if inner_model.config.two_stage:
            object_query_embedding, output_proposals = inner_model.generate_encoder_output_proposals(
                encoder_outputs.last_hidden_state_vision,
                ~mask_flatten,
                spatial_shapes,
            )
            enc_outputs_class = inner_model.encoder_output_class_embed(
                object_query_embedding,
                encoder_outputs.last_hidden_state_text,
                text_token_mask,
            )
            delta_bbox = inner_model.encoder_output_bbox_embed(object_query_embedding)
            enc_outputs_coord_logits = delta_bbox + output_proposals
            topk = inner_model.config.num_queries
            topk_logits = enc_outputs_class.max(-1)[0]
            topk_proposals = torch.topk(topk_logits, topk, dim=1)[1]
            topk_coords_logits = torch.gather(
                enc_outputs_coord_logits,
                1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
            )
            reference_points = topk_coords_logits.detach().sigmoid()
            init_reference_points = reference_points
            if query_embeds is not None:
                target = query_embeds.unsqueeze(0).repeat(batch_size, 1, 1)
            else:
                target = torch.gather(
                    object_query_embedding,
                    1,
                    topk_proposals.unsqueeze(-1).repeat(1, 1, inner_model.d_model),
                ).detach()
        else:
            target = query_embeds.unsqueeze(0).repeat(batch_size, 1, 1)
            reference_points = (
                inner_model.reference_points.weight.unsqueeze(0).repeat(batch_size, 1, 1).sigmoid()
            )
            init_reference_points = reference_points

        decoder_outputs = inner_model.decoder(
            inputs_embeds=target,
            vision_encoder_hidden_states=encoder_outputs.last_hidden_state_vision,
            vision_encoder_attention_mask=mask_flatten,
            text_encoder_hidden_states=encoder_outputs.last_hidden_state_text,
            text_encoder_attention_mask=~text_token_mask,
            reference_points=reference_points,
            spatial_shapes=vision_bundle["spatial_shapes"],
            level_start_index=vision_bundle["level_start_index"],
            valid_ratios=vision_bundle["valid_ratios"],
            self_attn_mask=None,
            return_dict=True,
        )
        hidden_states = decoder_outputs.intermediate_hidden_states
        inter_references_points = decoder_outputs.intermediate_reference_points
        outputs_classes = []
        outputs_coords = []
        num_levels = hidden_states.shape[1]
        for level in range(num_levels):
            reference = init_reference_points if level == 0 else inter_references_points[:, level - 1]
            reference = torch.special.logit(reference, eps=1e-5)
            outputs_class = det_model.class_embed[level](
                vision_hidden_state=hidden_states[:, level],
                text_hidden_state=encoder_outputs.last_hidden_state_text,
                text_token_mask=text_token_mask,
            )
            delta_bbox = det_model.bbox_embed[level](hidden_states[:, level])
            if reference.shape[-1] == 4:
                outputs_coord_logits = delta_bbox + reference
            else:
                delta_bbox[..., :2] += reference
                outputs_coord_logits = delta_bbox
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord_logits.sigmoid())
        logits = torch.stack(outputs_classes)[-1]
        pred_boxes = torch.stack(outputs_coords)[-1]
        return logits, pred_boxes, input_ids

    with torch.inference_mode():
        if use_autocast and dev.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return _run()
        return _run()


def _box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter + 1e-6
    return inter / union


def _flat_text(t: Any) -> str:
    if isinstance(t, list) and t:
        return str(t[0])
    return str(t)


def _grounding_caption(texts: list) -> str:
    parts = [_flat_text(t).strip().rstrip(".") for t in texts if _flat_text(t).strip()]
    if not parts:
        return "object."
    return ". ".join(parts) + "."


def _gdino_token_len(processor, caption: str) -> int:
    ids = processor.tokenizer(caption, add_special_tokens=True, return_tensors="pt")["input_ids"]
    return int(ids.shape[1])


def _gdino_pack_text_chunks(
    processor,
    texts_sub: list,
    vocab_ids: list[int],
    sub_names: list,
    *,
    max_class_chunk: int,
    max_tokens: int,
) -> list[dict]:
    """Pack class captions into chunks that fit GDINO max_text_len (token budget)."""
    chunks: list[dict] = []
    n = len(texts_sub)
    start = 0
    while start < n:
        hi = min(max_class_chunk, n - start)
        lo, best = 1, 1
        while lo <= hi:
            mid = (lo + hi) // 2
            caption = _grounding_caption(texts_sub[start : start + mid])
            if _gdino_token_len(processor, caption) <= max_tokens:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        chunk_texts = texts_sub[start : start + best]
        chunk_ids = vocab_ids[start : start + best]
        chunk_names = sub_names[start : start + best]
        caption = _grounding_caption(chunk_texts)
        text_inputs = processor(
            text=caption,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
        )
        chunks.append(
            {
                "start": start,
                "chunk_texts": chunk_texts,
                "chunk_ids": chunk_ids,
                "chunk_names": chunk_names,
                "text_inputs": text_inputs,
            }
        )
        start += best
    return chunks


def _gdino_prepare_text_bundle(
    text_inputs, inner_model, dev: str, *, use_autocast: bool = False
) -> dict:
    """Precompute text masks/tensors and BERT features on GPU (once per chunk)."""
    import torch
    from transformers.models.grounding_dino.modeling_grounding_dino import (
        generate_masks_with_special_tokens_and_transfer_map,
    )

    input_ids = text_inputs["input_ids"].to(dev, non_blocking=True)
    attention_mask = text_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(dev, non_blocking=True)
    token_type_ids = text_inputs.get("token_type_ids")
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids)
    else:
        token_type_ids = token_type_ids.to(dev, non_blocking=True)

    text_self_attention_masks, position_ids = generate_masks_with_special_tokens_and_transfer_map(
        input_ids
    )
    text_token_mask = attention_mask.bool()
    max_text_len = inner_model.config.max_text_len
    if text_self_attention_masks.shape[1] > max_text_len:
        text_self_attention_masks = text_self_attention_masks[:, :max_text_len, :max_text_len]
        position_ids = position_ids[:, :max_text_len]
        input_ids = input_ids[:, :max_text_len]
        token_type_ids = token_type_ids[:, :max_text_len]
        text_token_mask = text_token_mask[:, :max_text_len]

    def _encode_text():
        text_outputs = inner_model.text_backbone(
            input_ids,
            text_self_attention_masks,
            token_type_ids,
            position_ids,
            return_dict=True,
        )
        return inner_model.text_projection(text_outputs.last_hidden_state)

    with torch.inference_mode():
        if use_autocast and dev.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                text_features = _encode_text()
        else:
            text_features = _encode_text()

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "text_self_attention_masks": text_self_attention_masks,
        "position_ids": position_ids,
        "text_token_mask": text_token_mask,
        "text_features": text_features,
    }


def _label_to_index(label: Any, chunk_texts: list, chunk_names: list) -> int | None:
    if isinstance(label, (int, np.integer)):
        li = int(label)
        return li if 0 <= li < len(chunk_texts) else None
    text = str(label).strip().lower()
    if not text:
        return None
    flat = [_flat_text(t).strip().lower() for t in chunk_texts]
    names = [str(n).strip().lower() for n in chunk_names]
    for i, (t, n) in enumerate(zip(flat, names)):
        for cand in (t, n):
            if not cand:
                continue
            if cand == text or cand in text or text in cand:
                return i
    return None


def _result_arrays(results: dict) -> tuple[np.ndarray, np.ndarray, list]:
    boxes = results["boxes"]
    scores = results["scores"]
    labels = results["labels"]
    if hasattr(boxes, "cpu"):
        boxes = boxes.cpu().numpy()
    else:
        boxes = np.asarray(boxes)
    if hasattr(scores, "cpu"):
        scores = scores.cpu().numpy()
    else:
        scores = np.asarray(scores)
    if hasattr(labels, "cpu"):
        labels = labels.cpu().numpy().tolist()
    elif not isinstance(labels, list):
        labels = list(labels)
    return boxes, scores, labels


class GlipBackend:
    name = "glip"

    def __init__(
        self,
        device: str = "cuda:0",
        model_id: str | None = None,
        local_dir: str | None = None,
        chunk_size: int | None = None,
        score_thresh: float | None = None,
        max_text_tokens: int | None = None,
        b0_short_captions: bool | None = None,
        b0_short_vocab_threshold: int = 500,
        b0_image_short_edge: int | None = None,
    ) -> None:
        self.device = device
        self.cfg = load_paths()
        glip_cfg = self.cfg.get("glip", {})
        self.backend = glip_cfg.get("backend", "transformers")
        self.model_id = model_id or glip_cfg.get(
            "glip_tiny_model_id", "IDEA-Research/grounding-dino-tiny"
        )
        local = local_dir or glip_cfg.get("glip_tiny_local_dir")
        self.local_dir = None
        if local:
            p = Path(local)
            if not p.is_absolute():
                p = Path(self.cfg.get("_root", Path(__file__).resolve().parents[2])) / local
            if p.is_dir() and (p / "config.json").is_file():
                self.local_dir = str(p)
        native_w = glip_cfg.get("native_weight")
        self.native_weight = None
        if native_w:
            wp = Path(native_w)
            if not wp.is_absolute():
                wp = Path(self.cfg.get("_root", Path(__file__).resolve().parents[2])) / native_w
            if wp.is_file() and wp.stat().st_size > 1_000_000:
                self.native_weight = str(wp)
        self.chunk_size = int(
            chunk_size if chunk_size is not None else glip_cfg.get("chunk_size", 40)
        )
        self.score_thresh = float(
            score_thresh if score_thresh is not None else glip_cfg.get("score_thresh", 0.05)
        )
        gdino_cfg = self.cfg.get("gdino_base", {})
        self.max_text_tokens = int(
            max_text_tokens
            if max_text_tokens is not None
            else gdino_cfg.get("max_text_tokens", 240)
        )
        self.b0_short_captions = (
            b0_short_captions
            if b0_short_captions is not None
            else bool(gdino_cfg.get("b0_short_captions", False))
        )
        self.b0_short_vocab_threshold = int(
            gdino_cfg.get("b0_short_vocab_threshold", b0_short_vocab_threshold)
        )
        edge = b0_image_short_edge
        if edge is None:
            edge = gdino_cfg.get("b0_image_short_edge")
        self.b0_image_short_edge = int(edge) if edge else None
        self._last_gdino_n_chunks: int = 0
        self._use_autocast = self.device.startswith("cuda")
        self.yolo: Path = self.cfg["_yolo"]
        self._use_grounding_dino = False
        self._gdino_text_chunk_cache: dict[tuple[int, int, int, bool], list[dict]] = {}
        self._gdino_vision_image_id: int | None = None
        self._gdino_vision_bundle: dict | None = None

    def _model_source(self) -> str:
        return self.local_dir or self.model_id

    def _resolve_dev(self) -> str:
        import torch

        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"Requested {self.device} but torch.cuda.is_available() is False"
                )
            return self.device
        return "cpu"

    def _init_transformers(self):
        src = self._model_source()
        key = ("transformers", src, self.device)
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]

        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        processor = AutoProcessor.from_pretrained(src)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(src)
        proc_name = type(processor).__name__
        if not hasattr(processor, "post_process_grounded_object_detection"):
            raise RuntimeError(
                f"GLIP backend needs grounded post-process; got {proc_name} from {src}. "
                "Install transformers>=4.45 and download IDEA-Research/grounding-dino-tiny."
            )
        self._use_grounding_dino = "GroundingDino" in proc_name
        dev = self._resolve_dev()
        model = model.to(dev)
        model.eval()
        if dev.startswith("cuda"):
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        _MODEL_CACHE[key] = (model, processor, dev, self._use_grounding_dino)
        return model, processor, dev, self._use_grounding_dino

    def _init_model(self):
        if self.backend == "native" and self.native_weight:
            raise RuntimeError(
                "Native GLIP backend not wired yet; set glip.backend=transformers "
                f"(weight present: {self.native_weight})"
            )
        return self._init_transformers()

    def image_path(self, file_name: str) -> Path:
        from ovdeploy.paths_util import resolve_val2017_image

        return resolve_val2017_image(self.cfg, file_name)

    def _gdino_text_chunks(
        self,
        processor,
        texts_sub: list,
        vocab_ids: list[int],
        sub_names: list,
    ) -> list[dict]:
        use_short = (
            self.b0_short_captions and len(vocab_ids) >= self.b0_short_vocab_threshold
        )
        caption_texts = sub_names if use_short else texts_sub
        cache_key = (self.chunk_size, self.max_text_tokens, len(vocab_ids), use_short)
        if cache_key not in self._gdino_text_chunk_cache:
            chunks = _gdino_pack_text_chunks(
                processor,
                caption_texts,
                vocab_ids,
                sub_names,
                max_class_chunk=self.chunk_size,
                max_tokens=self.max_text_tokens,
            )
            self._gdino_text_chunk_cache[cache_key] = chunks
        chunks = self._gdino_text_chunk_cache[cache_key]
        if chunks and (
            "text_bundle" not in chunks[0]
            or "text_features" not in chunks[0].get("text_bundle", {})
        ):
            model, _, dev, _ = self._init_model()
            inner = model.model
            for chunk in chunks:
                chunk["text_bundle"] = _gdino_prepare_text_bundle(
                    chunk["text_inputs"],
                    inner,
                    dev,
                    use_autocast=self._use_autocast,
                )
        return chunks

    def warm_gdino_full_vocab(
        self,
        class_names: list[str],
        class_texts_raw: list,
        all_cat_ids: list[int],
        cid2idx: dict[int, int],
    ) -> None:
        """Preload model + GPU text bundles for B0 full-vocab (call once before B0 cache loop)."""
        model, processor, dev, use_gdino = self._init_model()
        if not use_gdino:
            return
        texts_sub = subset_class_texts(class_texts_raw, all_cat_ids, cid2idx)
        sub_names = [
            class_names[cid2idx[c]] if c in cid2idx else str(c) for c in all_cat_ids
        ]
        chunks = self._gdino_text_chunks(processor, texts_sub, all_cat_ids, sub_names)
        self._last_gdino_n_chunks = len(chunks)
        print(
            f"GDINO B0 warm: {len(chunks)} chunks, short_captions={self.b0_short_captions}, "
            f"device={dev}",
            flush=True,
        )

    def _predict_gdino(
        self,
        model,
        processor,
        dev: str,
        pil,
        texts_sub: list,
        vocab_ids: list[int],
        sub_names: list,
        image_id: int,
    ) -> list[dict]:
        import torch
        from PIL import Image

        if (
            self.b0_image_short_edge
            and len(vocab_ids) >= self.b0_short_vocab_threshold
        ):
            w, h = pil.size
            short = min(w, h)
            if short > self.b0_image_short_edge:
                scale = self.b0_image_short_edge / short
                pil = pil.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

        img_inputs = processor(images=pil, return_tensors="pt")
        pixel_values = img_inputs["pixel_values"].to(dev)
        pixel_mask = img_inputs.get("pixel_mask")
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(dev)

        inner = model.model
        if self._gdino_vision_image_id != image_id or self._gdino_vision_bundle is None:
            with torch.inference_mode():
                if self._use_autocast and dev.startswith("cuda"):
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        self._gdino_vision_bundle = _gdino_prepare_vision_bundle(
                            inner, pixel_values, pixel_mask
                        )
                else:
                    self._gdino_vision_bundle = _gdino_prepare_vision_bundle(
                        inner, pixel_values, pixel_mask
                    )
            self._gdino_vision_image_id = image_id

        chunks = self._gdino_text_chunks(processor, texts_sub, vocab_ids, sub_names)
        self._last_gdino_n_chunks = len(chunks)
        all_preds: list[dict] = []
        with torch.inference_mode():
            for chunk in chunks:
                start = chunk["start"]
                chunk_texts = chunk["chunk_texts"]
                chunk_ids = chunk["chunk_ids"]
                chunk_names = chunk["chunk_names"]
                logits, pred_boxes, input_ids = _gdino_forward_chunk(
                    model,
                    inner,
                    self._gdino_vision_bundle,
                    chunk["text_inputs"],
                    dev,
                    self._use_autocast,
                    text_bundle=chunk.get("text_bundle"),
                )
                outputs_obj = type("Out", (), {"logits": logits, "pred_boxes": pred_boxes})()
                results = processor.post_process_grounded_object_detection(
                    outputs_obj,
                    input_ids,
                    box_threshold=self.score_thresh,
                    text_threshold=self.score_thresh,
                    target_sizes=[pil.size[::-1]],
                )[0]
                boxes, scores, labels = _result_arrays(results)
                for box, score, label_idx in zip(boxes, scores, labels):
                    li = _label_to_index(label_idx, chunk_texts, chunk_names)
                    if li is None:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in box]
                    bbox_xywh = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
                    all_preds.append(
                        {
                            "bbox": bbox_xywh,
                            "score": float(score),
                            "label_idx": start + li,
                            "category_id": chunk_ids[li],
                            "category_name": chunk_names[li],
                        }
                    )
        return all_preds

    def predict(
        self,
        image_rgb: np.ndarray,
        texts: list[str],
        vocab_ids: list[int],
        image_id: int,
        class_names: list[str] | None = None,
        class_texts_raw: list | None = None,
        cid2idx: dict[int, int] | None = None,
    ) -> list[dict]:
        import torch
        from PIL import Image

        if class_texts_raw is not None and cid2idx is not None:
            texts_sub = subset_class_texts(class_texts_raw, vocab_ids, cid2idx)
            sub_names = [
                class_names[cid2idx[c]] if class_names and c in cid2idx else str(c)
                for c in vocab_ids
            ]
        else:
            texts_sub = texts
            sub_names = texts

        model, processor, dev, use_gdino = self._init_model()
        pil = Image.fromarray(image_rgb)
        h, w = image_rgb.shape[:2]

        if use_gdino:
            all_preds = self._predict_gdino(
                model, processor, dev, pil, texts_sub, vocab_ids, sub_names, image_id
            )
        else:
            all_preds = []
            for start in range(0, len(texts_sub), self.chunk_size):
                chunk_texts = texts_sub[start : start + self.chunk_size]
                chunk_ids = vocab_ids[start : start + self.chunk_size]
                chunk_names = sub_names[start : start + self.chunk_size]
                if not chunk_texts:
                    continue

                flat_texts = [_flat_text(t) for t in chunk_texts]
                inputs = processor(text=[flat_texts], images=pil, return_tensors="pt")
                inputs = {k: v.to(dev) for k, v in inputs.items()}
                with torch.no_grad():
                    if self._use_autocast and dev.startswith("cuda"):
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = model(**inputs)
                    else:
                        outputs = model(**inputs)

                target_sizes = torch.tensor([[h, w]], device=dev)
                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    box_threshold=self.score_thresh,
                    text_threshold=self.score_thresh,
                    target_sizes=target_sizes,
                )[0]

                boxes, scores, labels = _result_arrays(results)
                for box, score, label_idx in zip(boxes, scores, labels):
                    li = _label_to_index(label_idx, chunk_texts, chunk_names)
                    if li is None:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in box]
                    bbox_xywh = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
                    all_preds.append(
                        {
                            "bbox": bbox_xywh,
                            "score": float(score),
                            "label_idx": start + li,
                            "category_id": chunk_ids[li],
                            "category_name": chunk_names[li],
                        }
                    )

        all_preds.sort(key=lambda x: x["score"], reverse=True)
        kept: list[dict] = []
        for p in all_preds:
            dup = False
            for k in kept:
                if k["category_id"] == p["category_id"] and _box_iou_xyxy(
                    np.array(k["bbox"]), np.array(p["bbox"])
                ) > 0.5:
                    dup = True
                    break
            if not dup:
                kept.append(p)
        return kept
