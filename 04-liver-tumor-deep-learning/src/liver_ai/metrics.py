from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    p = np.asarray(pred).astype(bool)
    t = np.asarray(target).astype(bool)
    return float((2.0 * np.logical_and(p, t).sum() + eps) / (p.sum() + t.sum() + eps))


def iou_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    p = np.asarray(pred).astype(bool)
    t = np.asarray(target).astype(bool)
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    return float((inter + eps) / (union + eps))


def hausdorff95(pred: np.ndarray, target: np.ndarray) -> float:
    """Symmetric 95th percentile Hausdorff distance in pixel/voxel units."""
    p = np.asarray(pred).astype(bool)
    t = np.asarray(target).astype(bool)
    if not p.any() and not t.any():
        return 0.0
    if not p.any() or not t.any():
        return float("inf")

    p_surface = p ^ binary_erosion(p)
    t_surface = t ^ binary_erosion(t)
    d_to_t = distance_transform_edt(~t_surface)[p_surface]
    d_to_p = distance_transform_edt(~p_surface)[t_surface]
    distances = np.concatenate([d_to_t, d_to_p])
    return float(np.percentile(distances, 95))
