import torch
from liver_ai.models import UNet2D


def test_unet_output_shape():
    model = UNet2D(base=8)
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)
