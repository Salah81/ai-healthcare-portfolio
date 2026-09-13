from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class NiftiVolume:
    array: np.ndarray
    spacing: tuple[float, float, float]
    affine: np.ndarray


def load_nifti(path: str | Path) -> NiftiVolume:
    """Load a 3-D NIfTI volume using the optional nibabel dependency."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "Real NIfTI loading requires nibabel. Install with: "
            "python -m pip install -e '.[medical]'"
        ) from exc

    img = nib.load(str(path))
    array = np.asarray(img.get_fdata(dtype=np.float32))
    if array.ndim != 3:
        raise ValueError(f"Expected a 3-D NIfTI volume, got shape {array.shape}")
    spacing = tuple(float(v) for v in img.header.get_zooms()[:3])
    return NiftiVolume(array=array, spacing=spacing, affine=np.asarray(img.affine))


def patient_level_split(patient_ids: list[str], seed: int = 42,
                        train_fraction: float = 0.70, val_fraction: float = 0.15):
    """Return disjoint patient-level train/validation/test identifier lists."""
    if not 0 < train_fraction < 1 or not 0 <= val_fraction < 1:
        raise ValueError("Invalid split fractions")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be < 1")
    ids = np.array(sorted(set(patient_ids)), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * train_fraction))
    n_val = int(round(n * val_fraction))
    return ids[:n_train].tolist(), ids[n_train:n_train+n_val].tolist(), ids[n_train+n_val:].tolist()
