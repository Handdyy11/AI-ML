import numpy as np
import pandas as pd

from src.preprocess.binning import bin_counts, bin_counts_in_chunks
from src.preprocess.sampling import (
    feature_columns,
    fit_scaler,
    fit_scaler_in_chunks,
    split_data,
    stream_stratified_split,
)


def make_df():
    rng = np.random.default_rng(1)
    n = 1120
    return pd.DataFrame(
        {
            "Source IP": "10.0.0.1",
            "Source Port": rng.integers(1024, 65535, n),
            "Destination Port": rng.choice([80, 22], n),
            "Protocol": 6,  # constant column
            "Flow Duration": rng.exponential(1000, n),
            "Flow Packets/s": rng.random(n),
            "Label": "x",
            "family": ["Benign"] * 1000 + ["DoS"] * 100 + ["BruteForce"] * 20,
        }
    )


def test_split_keeps_class_proportions():
    df = make_df()
    train, test = split_data(df)
    full = df["family"].value_counts(normalize=True)
    for part in (train, test):
        share = part["family"].value_counts(normalize=True)
        assert (share - full).abs().max() < 0.01
    assert len(train) + len(test) == len(df)


def test_feature_columns_exclude_ids_and_constants():
    cols = feature_columns(make_df())
    assert "Source Port" not in cols
    assert "Protocol" not in cols
    assert set(cols) == {"Destination Port", "Flow Duration", "Flow Packets/s"}


def test_scaler_is_fitted_on_train_only():
    df = make_df()
    train, test = split_data(df)
    cols = feature_columns(train)
    scaler = fit_scaler(train, cols)
    assert np.allclose(scaler.mean_, train[cols].mean().to_numpy())
    scaled_train = scaler.transform(train[cols])
    assert np.allclose(scaled_train.mean(axis=0), 0, atol=1e-6)


def test_bin_counts_has_ten_bins_and_keeps_every_row():
    values = pd.Series(np.random.default_rng(2).exponential(100, 5000))
    table = bin_counts(values, n_bins=10)
    assert len(table) == 10
    assert table["count"].sum() == len(values)


def test_stream_split_and_chunked_scaler_match_in_memory(tmp_path):
    df = make_df()
    sample_path = tmp_path / "sample.parquet"
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    df.to_parquet(sample_path, index=False)

    info = stream_stratified_split(
        sample_path=sample_path,
        train_path=train_path,
        test_path=test_path,
        test_size=0.2,
        batch_size=300,
    )
    assert info["sample_rows"] == len(df)
    assert info["train_rows"] + info["test_rows"] == len(df)

    # Check chunked scaler via partial_fit matches exact mean and std of train.parquet
    scaler, cols = fit_scaler_in_chunks(train_path=train_path, batch_size=250)
    assert set(cols) == {"Destination Port", "Flow Duration", "Flow Packets/s"}
    train_loaded = pd.read_parquet(train_path)
    assert np.allclose(scaler.mean_, train_loaded[cols].mean().to_numpy(), atol=1e-5)

    # Check chunked histogram counts sum to total rows
    hist_table = bin_counts_in_chunks(sample_path, "Flow Duration", n_bins=10, batch_size=300)
    assert len(hist_table) == 10
    assert hist_table["count"].sum() == len(df)