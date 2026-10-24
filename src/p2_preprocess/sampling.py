"""Phase 2: stratified train/test split and incremental chunked feature scaling."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except ImportError:
    train_test_split = None
    StandardScaler = None

from src import config

# Columns that are identifiers, labels or meaningless as distances, so never model inputs.
# Source Port is excluded because it is essentially random for each connection.
# Destination Port IS kept because service ports (21, 22, 80, 443) distinguish BruteForce vs DoS.
NON_FEATURE_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Label",
    "family",
    "Source Port",
]


class _FallbackStandardScaler:
    """Lightweight Welford online StandardScaler fallback if scikit-learn is not installed."""

    def __init__(self):
        self.n_samples_seen_ = 0
        self.mean_ = None
        self.var_ = None
        self.scale_ = None

    def fit(self, X):
        arr = np.asarray(X, dtype=np.float64)
        self.n_samples_seen_ = len(arr)
        self.mean_ = arr.mean(axis=0)
        self.var_ = arr.var(axis=0, ddof=0)
        self.scale_ = np.where(self.var_ > 0, np.sqrt(self.var_), 1.0)
        return self

    def partial_fit(self, X):
        arr = np.asarray(X, dtype=np.float64)
        if len(arr) == 0:
            return self
        batch_n = len(arr)
        batch_mean = arr.mean(axis=0)
        batch_var = arr.var(axis=0, ddof=0)
        if self.mean_ is None or self.n_samples_seen_ == 0:
            self.n_samples_seen_ = batch_n
            self.mean_ = batch_mean
            self.var_ = batch_var
        else:
            old_n = self.n_samples_seen_
            new_n = old_n + batch_n
            delta = batch_mean - self.mean_
            self.mean_ = self.mean_ + delta * (batch_n / new_n)
            m_a = self.var_ * old_n
            m_b = batch_var * batch_n
            m2 = m_a + m_b + (delta ** 2) * (old_n * batch_n / new_n)
            self.var_ = m2 / new_n
            self.n_samples_seen_ = new_n
        self.scale_ = np.where(self.var_ > 0, np.sqrt(self.var_), 1.0)
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=np.float64)
        return (arr - self.mean_) / self.scale_


def _make_scaler():
    return StandardScaler() if StandardScaler is not None else _FallbackStandardScaler()


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = config.SEED):
    """Split a DataFrame into train and test so every family has the same share in both parts."""
    if train_test_split is not None:
        # Handle rare families with only 1 row in a chunk gracefully
        counts = df["family"].value_counts()
        if (counts >= 2).all():
            return train_test_split(
                df, test_size=test_size, stratify=df["family"], random_state=seed
            )
    # Stratified per-family split fallback
    train_parts, test_parts = [], []
    rng = np.random.default_rng(seed)
    for _, group in df.groupby("family", sort=False):
        idx = rng.permutation(len(group))
        n_test = max(1, int(round(len(group) * test_size))) if len(group) > 1 else 0
        test_parts.append(group.iloc[idx[:n_test]])
        train_parts.append(group.iloc[idx[n_test:]])
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def feature_columns(train: pd.DataFrame) -> list:
    """Numeric model inputs: no identifiers, and no column that never changes."""
    numeric = train.select_dtypes("number").columns
    return [
        c
        for c in numeric
        if c not in NON_FEATURE_COLUMNS and train[c].nunique() > 1
    ]


def fit_scaler(train: pd.DataFrame, cols: list):
    """Learn the mean and spread from the TRAINING data only."""
    return _make_scaler().fit(train[cols])


def class_proportions(df: pd.DataFrame) -> pd.Series:
    return df["family"].value_counts(normalize=True)


def iter_parquet_chunks(
    parquet_path: Path | str,
    batch_size: int = config.CHUNK_ROWS,
    columns: list[str] | None = None,
):
    """Yield pandas DataFrames of `batch_size` rows from a Parquet file without loading it all."""
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas()


def stream_stratified_split(
    sample_path: Path | str = config.SAMPLE_PATH,
    train_path: Path | str = config.TRAIN_PATH,
    test_path: Path | str = config.TEST_PATH,
    test_size: float = 0.2,
    batch_size: int = config.CHUNK_ROWS,
) -> dict:
    """Stream sample.parquet in chunks, split each mixed chunk 80/20 stratified by family,
    and write directly to train.parquet and test.parquet without loading all rows into RAM.
    """
    sample_path = Path(sample_path)
    train_path = Path(train_path)
    test_path = Path(test_path)
    for p in (train_path, test_path):
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            p.unlink()

    sample_counts: dict[str, int] = {}
    train_counts: dict[str, int] = {}
    test_counts: dict[str, int] = {}

    train_writer: pq.ParquetWriter | None = None
    test_writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None

    try:
        for chunk_idx, chunk in enumerate(
            iter_parquet_chunks(sample_path, batch_size=batch_size)
        ):
            if chunk.empty:
                continue
            train_chunk, test_chunk = split_data(
                chunk, test_size=test_size, seed=config.SEED + chunk_idx
            )

            for fam, c in chunk["family"].value_counts().items():
                sample_counts[fam] = sample_counts.get(fam, 0) + int(c)
            for fam, c in train_chunk["family"].value_counts().items():
                train_counts[fam] = train_counts.get(fam, 0) + int(c)
            for fam, c in test_chunk["family"].value_counts().items():
                test_counts[fam] = test_counts.get(fam, 0) + int(c)

            if schema is None:
                schema = pa.Table.from_pandas(
                    train_chunk, preserve_index=False
                ).schema

            train_table = pa.Table.from_pandas(
                train_chunk, schema=schema, preserve_index=False
            )
            test_table = pa.Table.from_pandas(
                test_chunk, schema=schema, preserve_index=False
            )

            if train_writer is None:
                train_writer = pq.ParquetWriter(train_path, schema)
            if test_writer is None:
                test_writer = pq.ParquetWriter(test_path, schema)

            train_writer.write_table(train_table)
            test_writer.write_table(test_table)
    finally:
        if train_writer is not None:
            train_writer.close()
        if test_writer is not None:
            test_writer.close()

    return {
        "sample_counts": pd.Series(sample_counts),
        "train_counts": pd.Series(train_counts),
        "test_counts": pd.Series(test_counts),
        "sample_rows": int(sum(sample_counts.values())),
        "train_rows": int(sum(train_counts.values())),
        "test_rows": int(sum(test_counts.values())),
    }


def fit_scaler_in_chunks(
    train_path: Path | str = config.TRAIN_PATH,
    batch_size: int = config.CHUNK_ROWS,
):
    """Incrementally fit StandardScaler across training chunks using partial_fit().

    Pass 1 runs partial_fit() on all candidate numeric columns across all training chunks
    to identify and exclude any globally constant columns (variance == 0).
    Pass 2 runs partial_fit() on the non-constant feature columns so the returned scaler
    matches the exact saved feature_columns list.
    """
    first_chunk = next(iter_parquet_chunks(train_path, batch_size=min(batch_size, 1_000)))
    numeric_cols = [
        c
        for c in first_chunk.select_dtypes("number").columns
        if c not in NON_FEATURE_COLUMNS
    ]

    probe_scaler = _make_scaler()
    for chunk in iter_parquet_chunks(
        train_path, batch_size=batch_size, columns=numeric_cols
    ):
        if not chunk.empty:
            probe_scaler.partial_fit(chunk[numeric_cols])

    # Keep only columns that vary across the training set
    kept_cols = [
        c for c, v in zip(numeric_cols, probe_scaler.var_) if float(v) > 1e-12
    ]

    final_scaler = _make_scaler()
    for chunk in iter_parquet_chunks(
        train_path, batch_size=batch_size, columns=kept_cols
    ):
        if not chunk.empty:
            final_scaler.partial_fit(chunk[kept_cols])

    return final_scaler, kept_cols