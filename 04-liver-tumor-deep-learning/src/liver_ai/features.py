from __future__ import annotations
import numpy as np
from scipy import ndimage


def morphology_features(mask: np.ndarray, spacing: tuple[float, ...] | None = None) -> dict[str, float]:
    """Extract transparent morphology descriptors from a binary tumour mask.

    For 2-D images the returned size is area; for 3-D it is volume.
    """
    m = np.asarray(mask).astype(bool)
    spacing = spacing or tuple([1.0] * m.ndim)
    if len(spacing) != m.ndim:
        raise ValueError("spacing dimensionality must match mask")

    voxel_measure = float(np.prod(spacing))
    size = float(m.sum() * voxel_measure)
    if not m.any():
        return {
            "size": 0.0,
            "equivalent_diameter": 0.0,
            "extent": 0.0,
            "component_count": 0.0,
        }

    coords = np.argwhere(m)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    bbox_shape = maxs - mins + 1
    bbox_measure = float(np.prod(bbox_shape * np.asarray(spacing)))
    extent = size / bbox_measure if bbox_measure > 0 else 0.0

    if m.ndim == 2:
        equivalent_diameter = float(2.0 * np.sqrt(size / np.pi))
    elif m.ndim == 3:
        equivalent_diameter = float(2.0 * ((3.0 * size) / (4.0 * np.pi)) ** (1.0 / 3.0))
    else:
        equivalent_diameter = float("nan")

    _, n_components = ndimage.label(m)
    return {
        "size": size,
        "equivalent_diameter": equivalent_diameter,
        "extent": float(extent),
        "component_count": float(n_components),
    }


def intensity_features(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    x = np.asarray(image, dtype=np.float32)
    m = np.asarray(mask).astype(bool)
    vals = x[m]
    if vals.size == 0:
        return {"mean_intensity": 0.0, "std_intensity": 0.0, "p90_intensity": 0.0}
    return {
        "mean_intensity": float(vals.mean()),
        "std_intensity": float(vals.std()),
        "p90_intensity": float(np.percentile(vals, 90)),
    }
