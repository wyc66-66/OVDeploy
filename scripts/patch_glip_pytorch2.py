#!/usr/bin/env python3
"""Patch microsoft/GLIP maskrcnn_benchmark for PyTorch 2.x (THC removal)."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA_DIR = ROOT / "third_party/GLIP/maskrcnn_benchmark/csrc/cuda"

REPLACEMENTS = [
    (r"#include <THC/THC\.h>\n", ""),
    (r"THCudaCheck\(", "AT_CUDA_CHECK("),
    (
        r"THCState \*state = at::globalContext\(\)\.lazyInitCUDA\(\);[^\n]*\n",
        "  at::globalContext().lazyInitCUDA();\n",
    ),
    (
        r"mask_dev = \(unsigned long long\*\) THCudaMalloc\(state, ([^)]+)\);",
        r"mask_dev = (unsigned long long*) c10::cuda::CUDACachingAllocator::raw_alloc(\1);",
    ),
    (r"THCudaFree\(state, mask_dev\);", "c10::cuda::CUDACachingAllocator::raw_delete(mask_dev);"),
    (r"THCCeilDiv\(([^,]+), 512L\)", r"THCCeilDiv(static_cast<int64_t>(\1), 512L)"),
]

EXTRA_INCLUDES = {
    "ROIAlign_cuda.cu": ['#include "common.hpp"\n'],
    "ROIPool_cuda.cu": ['#include "common.hpp"\n'],
    "SigmoidFocalLoss_cuda.cu": ['#include "common.hpp"\n'],
    "nms.cu": [
        "#include <c10/cuda/CUDACachingAllocator.h>\n",
        '#include "common.hpp"\n',
    ],
    "ml_nms.cu": [
        "#include <c10/cuda/CUDACachingAllocator.h>\n",
        '#include "common.hpp"\n',
    ],
}


def patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    orig = text
    for pat, repl in REPLACEMENTS:
        text = re.sub(pat, repl, text)
    extras = EXTRA_INCLUDES.get(path.name, [])
    if extras and "common.hpp" not in text:
        # insert after last ATen include block
        m = re.search(r"(#include <ATen/cuda/CUDAContext\.h>\n)", text)
        if m:
            insert_at = m.end()
            text = text[:insert_at] + "\n" + "".join(extras) + text[insert_at:]
    if text != orig:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    changed = []
    for cu in sorted(CUDA_DIR.glob("*.cu")):
        if patch_file(cu):
            changed.append(cu.name)
    print("Patched:", ", ".join(changed) if changed else "(none)")


if __name__ == "__main__":
    main()
