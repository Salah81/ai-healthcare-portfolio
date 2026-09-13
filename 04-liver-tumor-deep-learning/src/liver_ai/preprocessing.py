from __future__ import annotations
import numpy as np


def window_ct(image: np.ndarray, low: float = -150.0, high: float = 250.0) -> np.ndarray:
    """Clip and scale a CT-like image to [0, 1]."""
    x = np.clip(image.astype(np.float32), low, high)
    return (x - low) / (high - low)


def zscore_nonzero(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-score using non-zero voxels/pixels, useful after spatial masking."""
    x = image.astype(np.float32).copy()
    nz = x != 0
    if not np.any(nz):
        return x
    mean = float(x[nz].mean())
    std = float(x[nz].std())
    x[nz] = (x[nz] - mean) / max(std, eps)
    return x
