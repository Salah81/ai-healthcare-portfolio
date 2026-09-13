from __future__ import annotations
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from .preprocessing import window_ct
from .synthetic import SyntheticCase


class TumorSliceDataset(Dataset):
    def __init__(self, cases: list[SyntheticCase]):
        self.cases = cases

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int):
        case = self.cases[index]
        x = torch.from_numpy(window_ct(case.image)[None]).float()
        y = torch.from_numpy(case.tumor_mask[None]).float()
        return x, y


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        inter = (probs * target).sum(dim=dims)
        denom = probs.sum(dim=dims) + target.sum(dim=dims)
        dice_loss = 1.0 - ((2.0 * inter + 1e-6) / (denom + 1e-6)).mean()
        return bce + dice_loss


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(model: nn.Module, train_cases: list[SyntheticCase], epochs: int = 4,
                batch_size: int = 8, lr: float = 1e-3, device: str = "cpu") -> list[float]:
    model.to(device)
    loader = DataLoader(TumorSliceDataset(train_cases), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = DiceBCELoss()
    history: list[float] = []

    for _ in range(epochs):
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
    return history


def predict_mask(model: nn.Module, image: np.ndarray, threshold: float = 0.5,
                 device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    x = torch.from_numpy(window_ct(image)[None, None]).float().to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return (prob >= threshold).astype(np.uint8), prob
