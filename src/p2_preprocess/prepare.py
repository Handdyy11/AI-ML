"""Phase 2, step 2: histograms, stratified split and scaler.

Run after clean.py, from the project root:

    python -m src.p2_preprocess.prepare

Reads  data/processed/sample.parquet
Writes data/processed/train.parquet, test.parquet, scaler.joblib, feature_columns.json
       results/figures/hist_*.png
"""
import json

import joblib
import mlflow
import pandas as pd

from src import config
from src.mlflow_tracking import configure_tracking
from src.p2_preprocess.binning import plot_binned_histograms
from src.p2_preprocess.sampling import (
    class_proportions,
    feature_columns,
    fit_scaler,
    split_data,
)


def _run_preparation() -> None:
    # LOad the dataset
    df = pd.read_parquet(config.SAMPLE_PATH)
    print(f"Loaded {len(df):,} rows from {config.SAMPLE_PATH.name}")
    
    # plot histogram
    figures = plot_binned_histograms(df)
    print(f"Saved {len(figures)} histograms to {config.RESULTS / 'figures'}")

    # split
    train, test = split_data(df)
    
    # select features
    cols = feature_columns(train)
    # fit the scale
    scaler = fit_scaler(train, cols)
    
    #train
    train.to_parquet(config.DATA_PROCESSED / "train.parquet", index=False)
    test.to_parquet(config.DATA_PROCESSED / "test.parquet", index=False)

    #Saves the training and testing DataFrames as separate Parquet files.
    joblib.dump(scaler, config.DATA_PROCESSED / "scaler.joblib")
    (config.DATA_PROCESSED / "feature_columns.json").write_text(json.dumps(cols, indent=2))

    # Saves into table and checks class proportion
    table = pd.DataFrame(
        {
            "sample": class_proportions(df),
            "train": class_proportions(train),
            "test": class_proportions(test),
        }
    )
    print("\nShare of each family (%): the three columns should match closely")
    print((table * 100).round(2).to_string())
    print(f"\nTrain rows: {len(train):,}   Test rows: {len(test):,}   Features: {len(cols)}")

    mlflow.log_params(
        {
            "random_seed": config.SEED,
            "test_fraction": len(test) / len(df),
            "feature_count": len(cols),
        }
    )
    mlflow.log_metrics(
        {
            "sample_rows": len(df),
            "train_rows": len(train),
            "test_rows": len(test),
        }
    )
    for split_name, frame in (("sample", df), ("train", train), ("test", test)):
        for family, share in class_proportions(frame).items():
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


def main() -> None:
    configure_tracking()
    with mlflow.start_run(run_name="data-preparation"):
        _run_preparation()


if __name__ == "__main__":
    main()