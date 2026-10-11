import numpy as np

from src.p3_supervised.knn_classifier import (
    K_CHOICES,
    WEIGHT_CHOICES,
    choose_k,
    evaluate,
    select_features,
)


def make_blobs(n_per_class=150, seed=0):
    """Three well-separated classes in 6 columns; only the first 2 columns carry signal."""
    rng = np.random.default_rng(seed)
    centres = {"Benign": 0.0, "DoS": 6.0, "PortScan": -6.0}
    X, y = [], []
    for name, centre in centres.items():
        signal = rng.normal(centre, 1.0, size=(n_per_class, 2))
        noise = rng.normal(0.0, 1.0, size=(n_per_class, 4))
        X.append(np.hstack([signal, noise]))
        y += [name] * n_per_class
    return np.vstack(X).astype("float32"), np.array(y)


def test_select_features_keeps_the_informative_columns():
    X, y = make_blobs()
    cols = [f"c{i}" for i in range(6)]
    _, chosen = select_features(X, y, cols, k=2)
    assert set(chosen) == {"c0", "c1"}


def test_select_features_never_asks_for_more_than_exist():
    X, y = make_blobs()
    _, chosen = select_features(X, y, [f"c{i}" for i in range(6)], k=20)
    assert len(chosen) == 6


def test_choose_k_returns_a_valid_choice_and_a_full_table():
    X, y = make_blobs()
    k, weights, table = choose_k(X, y)
    assert k in K_CHOICES
    assert weights in WEIGHT_CHOICES
    assert len(table) == len(K_CHOICES) * len(WEIGHT_CHOICES)
    assert table["macro_f1"].max() > 0.95  # the classes are easy to separate


def test_evaluate_reports_per_family_recall():
    y_true = np.array(["Benign"] * 8 + ["DoS"] * 2)
    y_pred = np.array(["Benign"] * 8 + ["Benign", "DoS"])
    out = evaluate(y_true, y_pred)
    assert out["accuracy"] == 0.9
    assert out["per_family"]["DoS"]["recall"] == 0.5
    assert out["per_family"]["Benign"]["recall"] == 1.0