"""Phase 2: equal-width binning (histograms) of selected numeric features, supporting
both in-memory DataFrames and out-of-core chunked Parquet files.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src import config

# Columns of the CICIDS2017 files that make good histogram examples.
# Remaining columns are skipped.
DEFAULT_COLUMNS = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    "Fwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
]


def equal_width_bins(
    values: pd.Series, n_bins: int = 10, clip_quantile: float = 0.99
) -> pd.Series:
    """Cut a numeric column into n_bins bins of equal width.

    Network features are heavily skewed: a few huge values stretch the range, so
    without clipping almost every flow lands in the first bin. We clip values above
    the 99th percentile first, then cut into equal-width bins.
    """
    clipped = values.clip(upper=values.quantile(clip_quantile))
    return pd.cut(clipped, bins=n_bins)  # integer bins = equal width


def bin_counts(
    values: pd.Series, n_bins: int = 10, clip_quantile: float = 0.99
) -> pd.DataFrame:
    """Return a table with one row per bin: the bin range and how many flows fall in it."""
    binned = equal_width_bins(values, n_bins, clip_quantile)
    counts = binned.value_counts(sort=False)
    return pd.DataFrame({"bin": counts.index.astype(str), "count": counts.to_numpy()})


def bin_counts_in_chunks(
    parquet_path: Path | str,
    col: str,
    n_bins: int = 10,
    clip_quantile: float = 0.99,
    batch_size: int = config.CHUNK_ROWS,
) -> pd.DataFrame:
    """Compute equal-width bin counts across a Parquet file one chunk at a time.

    Pass 1 estimates the global minimum and the 99th-percentile upper bound across chunks.
    Pass 2 clips each chunk to [col_min, col_p99] and accumulates histogram counts via np.histogram.
    """
    pf = pq.ParquetFile(parquet_path)
    col_min = np.inf
    quantile_samples = []
    rng = np.random.default_rng(config.SEED)

    for batch in pf.iter_batches(batch_size=batch_size, columns=[col]):
        s = batch.column(0).to_pandas().dropna()
        if s.empty:
            continue
        col_min = min(col_min, float(s.min()))
        # Take up to 5,000 values per chunk to estimate the 99th percentile accurately in low RAM
        arr = s.to_numpy(dtype="float64")
        if len(arr) > 5_000:
            arr = rng.choice(arr, size=5_000, replace=False)
        quantile_samples.append(arr)

    if not quantile_samples:
        return pd.DataFrame({"bin": [], "count": []})

    combined_sample = np.concatenate(quantile_samples)
    col_p99 = float(np.quantile(combined_sample, clip_quantile))
    if col_p99 <= col_min:
        col_p99 = col_min + 1.0

    edges = np.linspace(col_min, col_p99, n_bins + 1)
    total_counts = np.zeros(n_bins, dtype=np.int64)

    for batch in pf.iter_batches(batch_size=batch_size, columns=[col]):
        s = batch.column(0).to_pandas().dropna()
        if s.empty:
            continue
        clipped = np.clip(s.to_numpy(dtype="float64"), col_min, col_p99)
        counts, _ = np.histogram(clipped, bins=edges)
        total_counts += counts

    labels = [
        f"({edges[i]:.2f}, {edges[i + 1]:.2f}]" for i in range(n_bins)
    ]
    return pd.DataFrame({"bin": labels, "count": total_counts})


def plot_binned_histograms(
    source: pd.DataFrame | Path | str,
    columns=DEFAULT_COLUMNS,
    out_dir=None,
    n_bins: int = 10,
):
    """Save one histogram PNG per column into results/figures/.

    Accepts either an in-memory DataFrame or a path to a Parquet file (which is read
    chunk-by-chunk with minimal memory).
    """
    out_dir = out_dir or (config.RESULTS / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    is_parquet = isinstance(source, (Path, str))
    if is_parquet:
        available_cols = set(pq.ParquetFile(source).schema.names)
    else:
        available_cols = set(source.columns)

    saved = []
    for col in columns:
        if col not in available_cols:
            continue
        if is_parquet:
            table = bin_counts_in_chunks(source, col, n_bins=n_bins)
        else:
            table = bin_counts(source[col], n_bins=n_bins)

        if table.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(table)), table["count"])
        ax.set_xticks(range(len(table)))
        ax.set_xticklabels(table["bin"], rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("flows")
        ax.set_title(
            f"{col}: {n_bins} equal-width bins (values above the 99th percentile clipped)",
            fontsize=9,
        )
        fig.tight_layout()

        name = col.replace("/", "_per_").replace(" ", "_")
        path = out_dir / f"hist_{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
    return saved