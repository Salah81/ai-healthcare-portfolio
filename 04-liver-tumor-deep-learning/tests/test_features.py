import numpy as np
from liver_ai.features import morphology_features, intensity_features


def test_morphology_features_non_empty():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:10, 7:13] = 1
    f = morphology_features(mask)
    assert f["size"] == 30.0
    assert f["extent"] == 1.0
    assert f["component_count"] == 1.0


def test_intensity_features():
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:4, 2:4] = 1
    f = intensity_features(image, mask)
    assert f["mean_intensity"] > 0
    assert f["p90_intensity"] >= f["mean_intensity"]
