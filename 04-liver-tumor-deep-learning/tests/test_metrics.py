import numpy as np
from liver_ai.metrics import dice_score, iou_score, hausdorff95


def test_perfect_overlap():
    x = np.zeros((16, 16), dtype=np.uint8)
    x[4:10, 5:11] = 1
    assert dice_score(x, x) == 1.0
    assert iou_score(x, x) == 1.0
    assert hausdorff95(x, x) == 0.0


def test_no_overlap():
    a = np.zeros((16, 16), dtype=np.uint8)
    b = np.zeros((16, 16), dtype=np.uint8)
    a[1:4, 1:4] = 1
    b[10:13, 10:13] = 1
    assert dice_score(a, b) < 1e-5
    assert iou_score(a, b) < 1e-5
