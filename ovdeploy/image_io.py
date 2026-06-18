"""Image loading helpers (Unicode paths on Windows)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def imread_bgr(path: Path) -> np.ndarray | None:
    """Read BGR image; works with non-ASCII paths via imdecode."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        import cv2
    except ImportError:
        return None

    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except OSError:
        pass

    return cv2.imread(str(path))
