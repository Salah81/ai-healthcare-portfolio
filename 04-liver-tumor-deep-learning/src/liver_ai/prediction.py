from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .features import morphology_features, intensity_features
from .synthetic import SyntheticCase


def case_features(case: SyntheticCase, mask: np.ndarray | None = None) -> np.ndarray:
    m = case.tumor_mask if mask is None else mask
    mf = morphology_features(m)
    inf = intensity_features(case.image, m)
    liver_size = max(float(case.liver_mask.sum()), 1.0)
    return np.array([
        mf["size"] / liver_size,
        mf["equivalent_diameter"],
        mf["extent"],
        mf["component_count"],
        inf["mean_intensity"],
        inf["std_intensity"],
        inf["p90_intensity"],
    ], dtype=np.float32)


def train_outcome_model(train_cases: list[SyntheticCase], test_cases: list[SyntheticCase]) -> dict:
    x_train = np.vstack([case_features(c) for c in train_cases])
    y_train = np.array([c.outcome for c in train_cases])
    x_test = np.vstack([case_features(c) for c in test_cases])
    y_test = np.array([c.outcome for c in test_cases])

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    clf.fit(x_train, y_train)
    prob = clf.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    auc = float(roc_auc_score(y_test, prob)) if len(np.unique(y_test)) == 2 else float("nan")
    return {"model": clf, "auc": auc, "accuracy": float(accuracy_score(y_test, pred))}
