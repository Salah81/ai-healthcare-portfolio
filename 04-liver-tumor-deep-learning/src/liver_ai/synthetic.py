from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SyntheticCase:
    image: np.ndarray
    liver_mask: np.ndarray
    tumor_mask: np.ndarray
    outcome: int
    progression_score: float


def _ellipse_mask(h: int, w: int, cy: float, cx: float, ry: float, rx: float) -> np.ndarray:
    yy, xx = np.mgrid[:h, :w]
    return (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0)


def make_case(seed: int, size: int = 96) -> SyntheticCase:
    """Create a CT-like synthetic slice with liver/tumour masks.

    This is a software-validation dataset only. It is not clinically realistic and
    must not be used to claim medical performance.
    """
    rng = np.random.default_rng(seed)
    h = w = size
    cy = size * (0.52 + rng.normal(0, 0.02))
    cx = size * (0.56 + rng.normal(0, 0.025))
    ry = size * rng.uniform(0.26, 0.31)
    rx = size * rng.uniform(0.31, 0.37)
    liver = _ellipse_mask(h, w, cy, cx, ry, rx)

    # Tumour is constrained to a central liver region for stable demo generation.
    angle = rng.uniform(0, 2 * np.pi)
    r = rng.uniform(0.05, 0.16) * size
    tcy = cy + np.sin(angle) * r
    tcx = cx + np.cos(angle) * r
    try_ = size * rng.uniform(0.035, 0.085)
    trx = size * rng.uniform(0.035, 0.09)
    tumor = _ellipse_mask(h, w, tcy, tcx, try_, trx) & liver

    image = rng.normal(-70, 18, size=(h, w)).astype(np.float32)
    image[liver] = rng.normal(65, 10, size=int(liver.sum())).astype(np.float32)
    image[tumor] = rng.normal(112, 12, size=int(tumor.sum())).astype(np.float32)

    # A smooth bias field adds scanner-like intensity variation.
    yy, xx = np.mgrid[:h, :w]
    bias = (xx / w - 0.5) * rng.uniform(-12, 12) + (yy / h - 0.5) * rng.uniform(-8, 8)
    image += bias.astype(np.float32)

    tumor_fraction = float(tumor.sum() / max(liver.sum(), 1))
    progression_score = float(0.65 * tumor_fraction + 0.35 * rng.uniform(0, 0.08))
    outcome = int(progression_score > 0.047)

    return SyntheticCase(
        image=image,
        liver_mask=liver.astype(np.uint8),
        tumor_mask=tumor.astype(np.uint8),
        outcome=outcome,
        progression_score=progression_score,
    )


def make_dataset(n: int = 80, size: int = 96, seed: int = 42) -> list[SyntheticCase]:
    return [make_case(seed + i, size=size) for i in range(n)]
