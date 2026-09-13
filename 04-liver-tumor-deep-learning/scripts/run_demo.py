from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from liver_ai.synthetic import make_dataset
from liver_ai.models import UNet2D
from liver_ai.training import set_seed, train_model, predict_mask
from liver_ai.metrics import dice_score, iou_score, hausdorff95
from liver_ai.prediction import train_outcome_model
from liver_ai.visualization import (
    save_segmentation_figure,
    save_training_curve,
    save_pipeline_figure,
)



def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LiverAI reproducible synthetic smoke test.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--cases", type=int, default=64)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.cases < 20:
        raise ValueError("Use at least 20 cases so the demo has a meaningful train/test split.")

    set_seed(42)
    cases = make_dataset(n=args.cases, size=args.size, seed=100)
    split = int(0.75 * len(cases))
    train_cases, test_cases = cases[:split], cases[split:]

    model = UNet2D(base=12)
    losses = train_model(model, train_cases, epochs=args.epochs, batch_size=8, device=args.device)

    dices, ious, hd95s = [], [], []
    example = None
    for case in test_cases:
        pred, _ = predict_mask(model, case.image, device=args.device)
        dices.append(dice_score(pred, case.tumor_mask))
        ious.append(iou_score(pred, case.tumor_mask))
        hd = hausdorff95(pred, case.tumor_mask)
        if np.isfinite(hd):
            hd95s.append(hd)
        if example is None:
            example = (case.image, case.tumor_mask, pred)

    outcome = train_outcome_model(train_cases, test_cases)
    metrics = {
        "scope": "synthetic software-validation demo only; not clinical performance",
        "device": args.device,
        "n_cases": len(cases),
        "epochs": args.epochs,
        "segmentation": {
            "dice_mean": float(np.mean(dices)),
            "dice_std": float(np.std(dices)),
            "iou_mean": float(np.mean(ious)),
            "hd95_mean_pixels": float(np.mean(hd95s)) if hd95s else None,
        },
        "outcome_proxy": {
            "auc": outcome["auc"],
            "accuracy": outcome["accuracy"],
            "warning": "Outcome labels are synthetic and generated partly from tumour burden; this metric validates code flow only."
        },
        "training_loss": losses,
    }

    results = ROOT / "results"
    figures = ROOT / "figures"
    results.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    save_pipeline_figure(figures / "research_pipeline.png")
    (results / "demo_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_training_curve(losses, figures / "demo_training_curve.png")
    if example is not None:
        save_segmentation_figure(*example, figures / "demo_segmentation.png")
    torch.save(model.state_dict(), results / "demo_unet_state_dict.pt")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
