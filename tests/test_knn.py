import numpy as np
import mlflow
import joblib
from sklearn.neighbors import KNeighborsClassifier

from src import config
from src.mlflow_tracking import configure_tracking
from src.supervised.models import (
    K_CHOICES,
    WEIGHT_CHOICES,
    StringLabelXGBClassifier,
    _log_model_runs,
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


def test_xgboost_model_round_trips_string_class_labels(tmp_path):
    X, y = make_blobs(n_per_class=20)
    model = StringLabelXGBClassifier(
        n_estimators=10,
        max_depth=3,
        n_jobs=1,
        random_state=0,
    ).fit(X, y)

    predictions = model.predict(X)
    assert set(predictions) <= set(y)
    assert set(model.classes_) == set(y)
    assert model.feature_importances_.shape == (X.shape[1],)

    model_path = tmp_path / "xgboost.joblib"
    joblib.dump(model, model_path)
    loaded = joblib.load(model_path)
    np.testing.assert_array_equal(loaded.predict(X), predictions)
    loaded.n_jobs = 1
    assert loaded.estimator_.n_jobs == 1


def test_log_model_runs_saves_a_loadable_model_and_test_metrics(tmp_path, monkeypatch):
    X, y = make_blobs(n_per_class=20)
    cols = [f"c{i}" for i in range(X.shape[1])]
    selector, _ = select_features(X, y, cols, k=2)
    model = KNeighborsClassifier(n_neighbors=3).fit(selector.transform(X), y)
    metrics = evaluate(y, model.predict(selector.transform(X)))

    original_tracking_uri = mlflow.get_tracking_uri()
    experiment_name = "model-logging-test"
    monkeypatch.setattr(
        config,
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
    )
    monkeypatch.setattr(
        config,
        "MLFLOW_ARTIFACT_ROOT",
        (tmp_path / "artifacts").as_uri(),
    )
    monkeypatch.setattr(config, "MLFLOW_EXPERIMENT", experiment_name)
    try:
        configure_tracking()
        with mlflow.start_run(run_name="comparison") as parent_run:
            _log_model_runs(
                {"knn": model},
                {"knn": metrics},
                selector,
                X,
                cols,
            )

        client = mlflow.MlflowClient()
        experiment = mlflow.get_experiment_by_name(experiment_name)
        child_runs = client.search_runs(
            [experiment.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'",
        )
        assert len(child_runs) == 1
        child = child_runs[0]
        assert child.data.metrics["test_accuracy"] == metrics["accuracy"]
        assert child.data.metrics["test_macro_f1"] == metrics["macro_f1"]

        logged_model = mlflow.sklearn.load_model(f"runs:/{child.info.run_id}/model")
        np.testing.assert_array_equal(
            logged_model.predict(X), model.predict(selector.transform(X))
        )
    finally:
        mlflow.set_tracking_uri(original_tracking_uri)