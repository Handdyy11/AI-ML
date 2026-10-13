"""Phase 3, part 1: k-NN attack classifier, compared with two simple baselines.

Run after prepare.py, from the project root:

    python -m src.p3_supervised.knn_classifier

Reads  data/processed/train.parquet, test.parquet, scaler.joblib, feature_columns.json
Writes results/metrics_knn.json, results/figures/confusion_matrix_knn.png,
       data/processed/knn_model.joblib
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src import config

N_FEATURES = 20  # keep the 20 most useful features (k-NN gets slow and noisy with too many)
K_CHOICES = [3, 5, 7, 9, 11, 13, 15]  # odd values of k avoid ties between two classes
WEIGHT_CHOICES = ["uniform", "distance"]  # distance: closer neighbours get a bigger vote
TUNING_ROWS = 40_000  # choose k on a subset, so the search stays fast


def load_scaled():
    """Load the train/test files from Phase 2 and apply the saved scaler."""
    train = pd.read_parquet(config.DATA_PROCESSED / "train.parquet")
    test = pd.read_parquet(config.DATA_PROCESSED / "test.parquet")
    scaler = joblib.load(config.DATA_PROCESSED / "scaler.joblib")
    cols = json.loads((config.DATA_PROCESSED / "feature_columns.json").read_text())

    X_train = scaler.transform(train[cols]).astype("float32")
    X_test = scaler.transform(test[cols]).astype("float32")
    return X_train, train["family"].to_numpy(), X_test, test["family"].to_numpy(), cols


def select_features(X_train, y_train, cols, k=N_FEATURES):
    """Pick the k features that best separate the families (ANOVA F-score).

    Fitted on training data only, like the scaler, to avoid leaking test information.
    """
    k = min(k, X_train.shape[1])
    selector = SelectKBest(f_classif, k=k).fit(X_train, y_train)
    chosen = [c for c, keep in zip(cols, selector.get_support()) if keep]
    return selector, chosen


def choose_k(X, y, seed=config.SEED):
    """Try each (k, weights) pair on a validation split of the TRAINING data.

    The test set is never used here. It stays untouched until the final score.
    Returns the best k, the best weighting, and a table of every score.
    """
    if len(X) > TUNING_ROWS:
        X, _, y, _ = train_test_split(X, y, train_size=TUNING_ROWS, stratify=y, random_state=seed)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )

    rows = []
    for weights in WEIGHT_CHOICES:
        for k in K_CHOICES:
            model = KNeighborsClassifier(n_neighbors=k, weights=weights, n_jobs=-1)
            model.fit(X_fit, y_fit)
            score = f1_score(y_val, model.predict(X_val), average="macro")
            rows.append({"k": k, "weights": weights, "macro_f1": score})

    table = pd.DataFrame(rows)
    best = table.sort_values("macro_f1", ascending=False).iloc[0]
    return int(best["k"]), str(best["weights"]), table


def evaluate(y_true, y_pred):
    """Accuracy, macro F1 and per-family precision / recall / F1."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    per_family = {
        name: {m: round(vals[m], 4) for m in ("precision", "recall", "f1-score")}
        for name, vals in report.items()
        if name not in ("accuracy", "macro avg", "weighted avg")
    }
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "per_family": per_family,
    }


def plot_confusion(y_true, y_pred, path):
    """Confusion matrix: rows are the true family, columns the predicted family.

    Each cell shows the share of that true family (so each row adds up to 100%),
    which keeps rare families visible next to the huge Benign row.
    """
    labels = [f for f in config.FAMILIES if f in set(y_true) | set(y_pred)]
    counts = confusion_matrix(y_true, y_pred, labels=labels)
    shares = counts / counts.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted family")
    ax.set_ylabel("True family")
    ax.set_title("k-NN on the test set (share of each true family)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            colour = "white" if shares[i, j] > 0.5 else "black"
            ax.text(j, i, f"{shares[i, j]:.1%}\n({counts[i, j]:,})",
                    ha="center", va="center", fontsize=8, color=colour)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    X_train, y_train, X_test, y_test, cols = load_scaled()
    print(f"Train: {X_train.shape[0]:,} rows   Test: {X_test.shape[0]:,} rows")

    selector, chosen = select_features(X_train, y_train, cols)
    X_train, X_test = selector.transform(X_train), selector.transform(X_test)
    print(f"Kept {len(chosen)} of {len(cols)} features")

    print("\nChoosing k on a validation split of the training data ...")
    best_k, best_weights, table = choose_k(X_train, y_train)
    print(table.sort_values("macro_f1", ascending=False).round(4).to_string(index=False))
    print(f"\nBest: k={best_k}, weights={best_weights}")

    print("\nTraining the final k-NN on all training rows and scoring the test set "
          "(this can take a few minutes) ...")
    knn = KNeighborsClassifier(n_neighbors=best_k, weights=best_weights, n_jobs=-1)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)

    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    tree = DecisionTreeClassifier(max_depth=12, random_state=config.SEED).fit(X_train, y_train)

    tree_pred = tree.predict(X_test)
    labels = [f for f in config.FAMILIES if f in set(y_test) | set(knn_pred)]

    print("\nTraining the random forest ...")
    forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",  # gives rare attack families more weight
        n_jobs=-1,
        random_state=config.SEED,
    ).fit(X_train, y_train)
    forest_pred = forest.predict(X_test)

    results = {
        "majority_baseline": evaluate(y_test, baseline.predict(X_test)),
        "decision_tree": {
            **evaluate(y_test, tree_pred),
            "confusion": confusion_matrix(y_test, tree_pred, labels=labels).tolist(),
        },

         "random_forest": {
            **evaluate(y_test, forest_pred),
            "confusion": confusion_matrix(y_test, forest_pred, labels=labels).tolist(),
            "feature_importance": {
                f: round(float(w), 4)
                for f, w in sorted(zip(chosen, forest.feature_importances_),
                                   key=lambda x: -x[1])
            },
        },
        "knn": {
            "k": best_k, "weights": best_weights,
            **evaluate(y_test, knn_pred),
            "confusion": confusion_matrix(y_test, knn_pred, labels=labels).tolist(),
        },
        "labels": labels,
        "tuning": table.round(4).to_dict("records"),
        "features_used": chosen,
    }

    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "metrics_knn.json").write_text(json.dumps(results, indent=2))
    plot_confusion(y_test, knn_pred, config.RESULTS / "figures" / "confusion_matrix_knn.png")
    joblib.dump(
        {"model": knn, "selector": selector, "features": cols},
        config.DATA_PROCESSED / "knn_model.joblib",
    )
    joblib.dump(
        {"model": tree, "selector": selector, "features": cols},
        config.DATA_PROCESSED / "tree_model.joblib",
    )
    joblib.dump(
        {"model": forest, "selector": selector, "features": cols},
        config.DATA_PROCESSED / "forest_model.joblib",
    )
    print("\nTest-set results")
    
    models = ("knn", "decision_tree", "random_forest", "majority_baseline")
    names = {
        "knn": "k-NN",
        "decision_tree": "decision tree",
        "random_forest": "random forest",
        "majority_baseline": "always say Benign",
    }

    summary = pd.DataFrame(
        {
            "accuracy": [results[m]["accuracy"] for m in models],
            "macro_f1": [results[m]["macro_f1"] for m in models],
        },
        index=[names[m] for m in models],
    )
    print(summary.to_string())

    for m in ("knn", "decision_tree", "random_forest"):
        print(f"\n{names[m]}: recall and precision per family "
              f"(recall = share of real attacks it caught)")
        for family, vals in results[m]["per_family"].items():
            print(f"  {family:<11} recall {vals['recall']:.3f}   "
                  f"precision {vals['precision']:.3f}")

    print("\nTop 5 features by random forest importance")
    for f, w in list(results["random_forest"]["feature_importance"].items())[:5]:
        print(f"  {f:<28}{w:.4f}")

    print(f"\nSaved {config.RESULTS / 'metrics_knn.json'}")
    print(f"Saved {config.RESULTS / 'figures' / 'confusion_matrix_knn.png'}")

if __name__ == "__main__":
    main()