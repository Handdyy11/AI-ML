"""Phase 2, step 2: chunked histograms, streaming stratified split, and incremental scaler.

Run after clean.py, from the project root:

    python -m src.preprocess.prepare

Streams data/processed/sample.parquet in chunks
Writes  data/processed/train.parquet, test.parquet, scaler.joblib, feature_columns.json
        results/figures/hist_*.png
"""
import json
import pickle

try:
    import joblib
except ImportError:
    joblib = None

import pandas as pd

from src import config
from src.preprocess.binning import plot_binned_histograms
from src.preprocess.sampling import (
    fit_scaler_in_chunks,
    stream_stratified_split,
)


def _save_joblib(obj, path) -> None:
    if joblib is not None:
        joblib.dump(obj, path)
    else:
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)


def _run_preparation() -> None:
    print(
        f"Streaming {config.SAMPLE_PATH.name} in chunks of {config.CHUNK_ROWS:,} rows ..."
    )

    # 1. Plot histograms in chunks directly from sample.parquet
    figures = plot_binned_histograms(config.SAMPLE_PATH)
    print(f"Saved {len(figures)} histograms to {config.RESULTS / 'figures'}")

    # 2. Stream stratified 80/20 split into train.parquet and test.parquet
    split_info = stream_stratified_split(
        sample_path=config.SAMPLE_PATH,
        train_path=config.TRAIN_PATH,
        test_path=config.TEST_PATH,
        test_size=0.2,
        batch_size=config.CHUNK_ROWS,
    )

    # 3. Fit StandardScaler incrementally on train.parquet chunks via partial_fit()
    scaler, cols = fit_scaler_in_chunks(
        train_path=config.TRAIN_PATH,
        batch_size=config.CHUNK_ROWS,
    )

    # 4. Save scaler and selected non-constant feature column names
    _save_joblib(scaler, config.DATA_PROCESSED / "scaler.joblib")
    (config.DATA_PROCESSED / "feature_columns.json").write_text(
        json.dumps(cols, indent=2)
    )

    # 5. Report class proportions across sample, train, and test
    sample_props = split_info["sample_counts"] / max(split_info["sample_rows"], 1)
    train_props = split_info["train_counts"] / max(split_info["train_rows"], 1)
    test_props = split_info["test_counts"] / max(split_info["test_rows"], 1)

    table = pd.DataFrame(
        {
            "sample": sample_props,
            "train": train_props,
            "test": test_props,
        }
    )
    print("\nShare of each family (%): the three columns should match closely")
    print((table * 100).round(2).to_string())
    print(
        f"\nTrain rows: {split_info['train_rows']:,}   "
        f"Test rows: {split_info['test_rows']:,}   "
        f"Features: {len(cols)}"
    )

    try:
        import mlflow

        mlflow.log_params(
            {
                "random_seed": config.SEED,
                "chunk_rows": config.CHUNK_ROWS,
                "test_fraction": round(
                    split_info["test_rows"] / max(split_info["sample_rows"], 1), 4
                ),
                "feature_count": len(cols),
            }
        )
        mlflow.log_metrics(
            {
                "sample_rows": split_info["sample_rows"],
                "train_rows": split_info["train_rows"],
                "test_rows": split_info["test_rows"],
            }
        )
        for split_name, series in (
            ("sample", sample_props),
            ("train", train_props),
            ("test", test_props),
        ):
            for family, share in series.items():
                mlflow.log_metric(f"{split_name}_share_{family}", float(share))

        mlflow.log_artifact(
            str(config.DATA_PROCESSED / "feature_columns.json"),
            artifact_path="preparation",
        )
        mlflow.log_artifact(
            str(config.DATA_PROCESSED / "scaler.joblib"),
            artifact_path="preparation",
        )
        for figure in figures:
            mlflow.log_artifact(str(figure), artifact_path="preparation/figures")
    except ImportError:
        pass


def main() -> None:
    try:
        import mlflow
        from src.mlflow_tracking import configure_tracking

        configure_tracking()
        with mlflow.start_run(run_name="data-preparation"):
            _run_preparation()
    except ImportError:
        _run_preparation()


if __name__ == "__main__":
    main()