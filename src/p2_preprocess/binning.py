"""Phase 2: equal-width binning (histograms) of selected numeric features."""
import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd

from src import config

# Columns of the CICIDS2017 files that make good histogram examples.
# Any column missing from your data is skipped.
DEFAULT_COLUMNS = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    "Fwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
]


def equal_width_bins(values: pd.Series, n_bins: int = 10, clip_quantile: float = 0.99) -> pd.Series:
    """Cut a numeric column into n_bins bins of equal width.

    Network features are heavily skewed: a few huge values stretch the range, so
    without clipping almost every flow lands in the first bin. We clip values above
    the 99th percentile first, then cut into equal-width bins.
    """
    clipped = values.clip(upper=values.quantile(clip_quantile))
    return pd.cut(clipped, bins=n_bins)  # integer bins = equal width


def bin_counts(values: pd.Series, n_bins: int = 10, clip_quantile: float = 0.99) -> pd.DataFrame:
    """Return a table with one row per bin: the bin range and how many flows fall in it."""
    binned = equal_width_bins(values, n_bins, clip_quantile)
    counts = binned.value_counts(sort=False)
    return pd.DataFrame({"bin": counts.index.astype(str), "count": counts.to_numpy()})


def plot_binned_histograms(df: pd.DataFrame, columns=DEFAULT_COLUMNS, out_dir=None, n_bins: int = 10):
    """Save one histogram PNG per column into results/figures/. Returns the saved paths."""
    out_dir = out_dir or (config.RESULTS / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for col in columns:
        if col not in df.columns:
            continue
        table = bin_counts(df[col], n_bins)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(table)), table["count"])
        ax.set_xticks(range(len(table)))
        ax.set_xticklabels(table["bin"], rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("flows")
        ax.set_title(f"{col}: {n_bins} equal-width bins (values above the 99th percentile clipped)", fontsize=9)
        fig.tight_layout()

        name = col.replace("/", "_per_").replace(" ", "_")
        path = out_dir / f"hist_{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
    return saved