"""CLIP top-K vocabulary for B4_clip (per-image)."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _load_clip():
    try:
        import open_clip
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        tok = open_clip.get_tokenizer("ViT-B-32")
        model.eval()
        return model, preprocess, tok, torch
    except Exception:
        pass
    try:
        import clip
        import torch

        model, preprocess = clip.load("ViT-B/32", device="cpu")
        return model, preprocess, None, torch
    except Exception:
        return None


def clip_topk_cat_ids(
    image_rgb: np.ndarray,
    cat_ids: Sequence[int],
    prompts_per_cat: dict[int, list[str]],
    k: int,
    freq_fallback: list[int],
    seed: int = 42,
) -> list[int]:
    k = min(k, len(cat_ids))
    if k <= 0:
        return []

    try:
        from PIL import Image
    except ImportError:
        return freq_fallback[:k]

    loaded = _load_clip()
    if loaded is None:
        return freq_fallback[:k]

    model, preprocess, tokenizer, torch = loaded
    device = next(model.parameters()).device
    pil = Image.fromarray(image_rgb.astype(np.uint8))
    image_input = preprocess(pil).unsqueeze(0).to(device)

    texts, order = [], []
    for cid in cat_ids:
        pr = prompts_per_cat.get(cid, [str(cid)])
        texts.append(pr[0] if pr else str(cid))
        order.append(cid)

    with torch.no_grad():
        if tokenizer is not None:
            text_tokens = tokenizer(texts).to(device)
            img_f = model.encode_image(image_input)
            txt_f = model.encode_text(text_tokens)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
            sims = (img_f @ txt_f.T).squeeze(0).cpu().numpy()
        else:
            import clip as clip_pkg

            text_tokens = clip_pkg.tokenize(texts, truncate=True).to(device)
            logits, _ = model(image_input, text_tokens)
            sims = logits.squeeze(0).softmax(dim=-1).cpu().numpy()

    idx = np.argsort(-sims)[:k]
    return [order[int(i)] for i in idx]
