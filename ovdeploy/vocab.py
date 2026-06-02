"""Prompt tables and vocabulary noise."""
from __future__ import annotations

import random
from typing import Iterable

SYNONYM_MAP = {
    "cat": ["feline", "domestic cat"],
    "dog": ["canine", "domestic dog"],
    "car": ["automobile", "motor vehicle"],
    "person": ["human", "individual"],
    "chair": ["seat", "armchair"],
    "cup": ["mug", "drinking cup"],
    "bottle": ["container", "glass bottle"],
    "book": ["volume", "textbook"],
    "phone": ["cellphone", "mobile phone"],
}


def build_prompts_for_cat(cat_name: str, template: str | list) -> list[str]:
    if isinstance(template, list):
        base = template[0] if template else cat_name
    else:
        base = str(template)
    prompts = [base]
    if cat_name in SYNONYM_MAP:
        prompts.extend(SYNONYM_MAP[cat_name][:1])
    return prompts


def apply_noise(
    cat_ids: list[int],
    cat_id_to_name: dict[int, str],
    cat_id_to_text: dict[int, list],
    noise: str,
    rng: random.Random,
) -> tuple[list[int], dict[str, list[str]]]:
    prompts: dict[str, list[str]] = {}
    out_ids = list(cat_ids)
    for cid in out_ids:
        name = cat_id_to_name.get(cid, str(cid))
        tpl = cat_id_to_text.get(cid, [name])
        prompts[str(cid)] = build_prompts_for_cat(name, tpl)

    if noise == "synonym":
        for cid in out_ids:
            name = cat_id_to_name.get(cid, "")
            if name in SYNONYM_MAP:
                prompts[str(cid)] = [rng.choice(SYNONYM_MAP[name])]

    elif noise == "missing_class" and len(out_ids) > 2:
        drop = rng.choice(out_ids)
        out_ids = [c for c in out_ids if c != drop]

    return out_ids, prompts


def subset_class_texts(
    full_texts: list,
    cat_ids: list[int],
    cat_id_to_idx: dict[int, int],
) -> list:
    """Order texts to match cat_ids via LVIS category index."""
    out = []
    for cid in cat_ids:
        idx = cat_id_to_idx.get(cid)
        if idx is not None and idx < len(full_texts):
            out.append(full_texts[idx])
        else:
            out.append([str(cid)])
    return out


def freq_sorted_cat_ids(lvis: dict) -> list[int]:
    freq_order = {"f": 0, "c": 1, "r": 2}
    cats = sorted(
        lvis["categories"],
        key=lambda c: (freq_order.get(c.get("frequency", "c"), 1), c["id"]),
    )
    return [c["id"] for c in cats]
