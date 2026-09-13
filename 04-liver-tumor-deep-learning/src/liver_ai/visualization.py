from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


def save_segmentation_figure(
    image: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    truth = truth.astype(bool)
    pred = pred.astype(bool)

    intersection = np.logical_and(truth, pred).sum()
    denominator = truth.sum() + pred.sum()
    dice = 2.0 * intersection / denominator if denominator > 0 else 1.0

    union = np.logical_or(truth, pred).sum()
    iou = intersection / union if union > 0 else 1.0

    overlap = np.zeros_like(truth, dtype=np.uint8)
    overlap[np.logical_and(truth, pred)] = 1
    overlap[np.logical_and(~truth, pred)] = 2
    overlap[np.logical_and(truth, ~pred)] = 3

    overlap_rgb = np.zeros((*truth.shape, 3), dtype=float)
    overlap_rgb[overlap == 1] = [0.20, 0.70, 0.30]
    overlap_rgb[overlap == 2] = [0.90, 0.30, 0.25]
    overlap_rgb[overlap == 3] = [1.00, 0.65, 0.15]

    alpha_mask = (overlap > 0).astype(float)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Synthetic CT-like image")

    axes[1].imshow(image, cmap="gray")
    axes[1].contour(truth.astype(float), levels=[0.5], linewidths=1.8)
    axes[1].set_title("Ground-truth tumour")

    axes[2].imshow(image, cmap="gray")
    axes[2].contour(pred.astype(float), levels=[0.5], linewidths=1.8)
    axes[2].set_title("U-Net prediction")

    axes[3].imshow(image, cmap="gray", alpha=0.45)
    axes[3].imshow(overlap_rgb, alpha=alpha_mask)
    axes[3].set_title(f"Overlap analysis\nDice={dice:.3f} · IoU={iou:.3f}")

    legend_elements = [
        Patch(facecolor=(0.20, 0.70, 0.30), label="True positive"),
        Patch(facecolor=(0.90, 0.30, 0.25), label="False positive"),
        Patch(facecolor=(1.00, 0.65, 0.15), label="False negative"),
    ]

    axes[3].legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=1,
        frameon=False,
        fontsize=8,
    )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        "Qualitative U-Net Segmentation — Synthetic Validation Example",
        fontsize=14,
        y=1.02,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_training_curve(
    losses: list[float],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(losses) + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, losses, marker="o", linewidth=1.8, markersize=4.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (BCE + Soft Dice)")
    ax.set_title("U-Net Training on Synthetic CT-like Data")
    ax.grid(True, alpha=0.25)

    if losses:
        ax.annotate(
            f"Initial: {losses[0]:.3f}",
            xy=(epochs[0], losses[0]),
            xytext=(12, -18),
            textcoords="offset points",
            fontsize=9,
        )
        ax.annotate(
            f"Final: {losses[-1]:.3f}",
            xy=(epochs[-1], losses[-1]),
            xytext=(-75, 12),
            textcoords="offset points",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_pipeline_figure(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    stages = [
        (0.5, 2.0, 1.7, 1.0, "CT imaging"),
        (2.6, 2.0, 1.8, 1.0, "Preprocessing\n& windowing"),
        (4.8, 2.0, 1.9, 1.0, "U-Net tumour\nsegmentation"),
        (7.1, 2.0, 1.7, 1.0, "Tumour mask"),
        (9.2, 2.0, 2.0, 1.0, "Morphology +\nintensity features"),
        (11.6, 2.0, 1.9, 1.0, "Outcome /\nprogression model"),
    ]

    for x, y, w, h, label in stages:
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.5,
            fill=False,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=10,
        )

    for i in range(len(stages) - 1):
        x, y, w, h, _ = stages[i]
        nx, ny, _, nh, _ = stages[i + 1]
        arrow = FancyArrowPatch(
            (x + w, y + h / 2),
            (nx, ny + nh / 2),
            arrowstyle="->",
            mutation_scale=16,
            linewidth=1.4,
        )
        ax.add_patch(arrow)

    ax.text(
        5.75,
        0.85,
        "Segmentation evaluation",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        5.75,
        0.45,
        "Dice · IoU · HD95",
        ha="center",
        fontsize=10,
    )
    ax.text(
        12.55,
        0.85,
        "Outcome evaluation",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        12.55,
        0.45,
        "AUROC · Accuracy",
        ha="center",
        fontsize=10,
    )
    ax.text(
        7,
        4.25,
        "LiverAI — End-to-End Research Pipeline",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        7,
        3.75,
        "Synthetic software-validation prototype · not for clinical use",
        ha="center",
        va="center",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
