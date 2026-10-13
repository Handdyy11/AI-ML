"""Phase 2, step 1: load the CICIDS2017 day-files, clean them, group the labels
into families, take a stratified sample and save it as data/processed/sample.parquet.

Run from the project root:

    python -m src.p2_preprocess.clean
"""
import numpy as np
import pandas as pd
import json

from src import config


def label_to_family(label: str) -> str:
    """Group the many CICIDS2017 attack labels into the project's families."""
    name = str(label).strip().lower()
    if name == "benign":
        return "Benign"
    if name in ("ftp-patator", "ssh-patator"):
        return "BruteForce"
    if name.startswith("dos") or name == "ddos" or name == "heartbleed":
        return "DoS"
    if name == "portscan":
        return "PortScan"
    return "Other"

def clean_chunk(chunk: pd.DataFrame, stats=None) -> pd.DataFrame:
    """Fix the known quirks of the CICIDS2017 CSVs for one piece of a file."""
    chunk = chunk.copy()
    chunk.columns = chunk.columns.str.strip()
    n_raw = len(chunk)

    chunk = chunk.drop(columns=["Fwd Header Length.1"], errors="ignore")
    chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
    n_after_na = len(chunk)

    chunk["Label"] = chunk["Label"].astype(str).str.strip()
    chunk["family"] = chunk["Label"].map(label_to_family)

    if "Timestamp" in chunk.columns:
        chunk["Timestamp"] = pd.to_datetime(
            chunk["Timestamp"], format="mixed", dayfirst=True, errors="coerce"
        )

    float_cols = chunk.select_dtypes("float64").columns
    chunk[float_cols] = chunk[float_cols].astype("float32")
    chunk = chunk.drop_duplicates()

    if stats is not None:
        stats["rows_raw"] += n_raw
        stats["missing_removed"] += n_raw - n_after_na
        stats["duplicates_removed"] += n_after_na - len(chunk)
        stats["rows_clean"] += len(chunk)
    return chunk


def build_sample(raw_dir=config.DATA_RAW, fraction=config.SAMPLE_FRACTION):
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    stats = {"rows_raw": 0, "missing_removed": 0, "duplicates_removed": 0,
             "rows_clean": 0, "files": {}}
    parts = []
    for path in files:
        print(f"Reading {path.name} ...")
        file_rows, n_cols = 0, 0
        for chunk in pd.read_csv(
            path, chunksize=config.CHUNK_ROWS, encoding="latin-1", low_memory=False
        ):
            file_rows += len(chunk)
            n_cols = chunk.shape[1]
            parts.append(stratified_sample(clean_chunk(chunk, stats), fraction))
        stats["files"][path.name] = {"rows": file_rows, "columns": n_cols}

    sample = pd.concat(parts, ignore_index=True).drop_duplicates()
    stats["rows_sample"] = len(sample)
    stats["sample_fraction"] = fraction
    return sample, stats

def stratified_sample(chunk: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """Keep the same fraction of every family, so class proportions are preserved."""
    return chunk.groupby("family", group_keys=False).sample(
        frac=fraction, random_state=config.SEED
    )

def main() -> None:
    sample, stats = build_sample()

    counts = sample["family"].value_counts()
    print("\nRows per family in the sample:")
    for family, n in counts.items():
        print(f"  {family:<11}{n:>9,}  ({n / len(sample):.1%})")
    print(f"  {'TOTAL':<11}{len(sample):>9,}")

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(config.SAMPLE_PATH, index=False)
    print(f"\nSaved {config.SAMPLE_PATH}")

    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "preprocessing_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Saved {config.RESULTS / 'preprocessing_stats.json'}")


if __name__ == "__main__":
    main()