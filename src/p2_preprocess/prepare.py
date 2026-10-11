"""Phase 2, step 2: histograms, stratified split and scaler.

Run after clean.py, from the project root:

    python -m src.p2_preprocess.prepare

Reads  data/processed/sample.parquet
Writes data/processed/train.parquet, test.parquet, scaler.joblib, feature_columns.json
       results/figures/hist_*.png
"""
import json

import joblib
import pandas as pd

from src import config
from src.p2_preprocess.binning import plot_binned_histograms
from src.p2_preprocess.sampling import (
    class_proportions,
    feature_columns,
    fit_scaler,
    split_data,
)


def main() -> None:
    df = pd.read_parquet(config.SAMPLE_PATH)
    print(f"Loaded {len(df):,} rows from {config.SAMPLE_PATH.name}")

    figures = plot_binned_histograms(df)
    print(f"Saved {len(figures)} histograms to {config.RESULTS / 'figures'}")

    train, test = split_data(df)
    cols = feature_columns(train)
    scaler = fit_scaler(train, cols)

    train.to_parquet(config.DATA_PROCESSED / "train.parquet", index=False)
    test.to_parquet(config.DATA_PROCESSED / "test.parquet", index=False)
    joblib.dump(scaler, config.DATA_PROCESSED / "scaler.joblib")
    (config.DATA_PROCESSED / "feature_columns.json").write_text(json.dumps(cols, indent=2))

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


if __name__ == "__main__":
    main()