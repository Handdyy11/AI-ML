"""Phase 2, step 1: load the CICIDS2017 day-files, clean them, group the labels
into families, take a stratified sample and save it as data/processed/sample.parquet.

Run from the project root:

    python -m src.p2_preprocess.clean
"""
import numpy as np
import pandas as pd

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


def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Fix the known quirks of the CICIDS2017 CSVs for one piece of a file."""
    # Column names have leading spaces, e.g. ' Destination Port'.
    chunk = chunk.copy()
    chunk.columns = chunk.columns.str.strip()

    # 'Fwd Header Length' appears twice in the original files; pandas renames
    # the second copy to 'Fwd Header Length.1'. Drop the duplicate.
    chunk = chunk.drop(columns=["Fwd Header Length.1"], errors="ignore")

    # Some rate columns contain inf and NaN. Turn inf into NaN, then drop those rows.
    chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()

    chunk["Label"] = chunk["Label"].astype(str).str.strip()
    chunk["family"] = chunk["Label"].map(label_to_family)

    # Dates like '3/7/2017 8:55' are day/month/year. Anything unparseable becomes NaT.
    if "Timestamp" in chunk.columns:
        chunk["Timestamp"] = pd.to_datetime(
            chunk["Timestamp"], format="mixed", dayfirst=True, errors="coerce"
        )

    # Halve the memory used by the float columns.
    float_cols = chunk.select_dtypes("float64").columns
    chunk[float_cols] = chunk[float_cols].astype("float32")

    return chunk.drop_duplicates()


def stratified_sample(chunk: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """Keep the same fraction of every family, so class proportions are preserved."""
    return chunk.groupby("family", group_keys=False).sample(
        frac=fraction, random_state=config.SEED
    )


def build_sample(raw_dir=config.DATA_RAW, fraction=config.SAMPLE_FRACTION) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    parts = []
    for path in files:
        print(f"Reading {path.name} ...")
        for chunk in pd.read_csv(
            path, chunksize=config.CHUNK_ROWS, encoding="latin-1", low_memory=False
        ):
            parts.append(stratified_sample(clean_chunk(chunk), fraction))

    return pd.concat(parts, ignore_index=True).drop_duplicates()


def main() -> None:
    sample = build_sample()

    counts = sample["family"].value_counts()
    print("\nRows per family in the sample:")
    for family, n in counts.items():
        print(f"  {family:<11}{n:>9,}  ({n / len(sample):.1%})")
    print(f"  {'TOTAL':<11}{len(sample):>9,}")

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(config.SAMPLE_PATH, index=False)
    print(f"\nSaved {config.SAMPLE_PATH}")


if __name__ == "__main__":
    main()