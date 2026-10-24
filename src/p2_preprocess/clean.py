"""Phase 2, step 1: stream the CICIDS2017 day-files in chunks, clean them, deduplicate
repetitive flood vectors, balance attack families across mixed shard buckets, and save
data/processed/sample.parquet without loading the full raw dataset into memory.

Run from the project root:

    python -m src.p2_preprocess.clean
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import config

# Identifier columns ignored when deduplicating identical flow feature patterns within a chunk
ID_META_COLUMNS = {
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Source Port",
}


def label_to_family(label: str) -> str:
    """Group the many CICIDS2017 attack labels into the project's families."""
    name = str(label).strip().lower()
    if name == "benign":
        return "Benign"
    if name in ("ftp-patator", "ssh-patator") or "brute force" in name:
        return "BruteForce"
    if name.startswith("dos") or name == "ddos" or name == "heartbleed":
        return "DoS"
    if name == "portscan":
        return "PortScan"
    return "Other"


def clean_chunk(
    chunk: pd.DataFrame,
    stats: dict | None = None,
    deduplicate_features: bool = False,
) -> pd.DataFrame:
    """Fix the known quirks of the CICIDS2017 CSVs for one chunk of a file."""
    chunk = chunk.copy()
    chunk.columns = chunk.columns.str.strip()
    n_raw = len(chunk)

    chunk = chunk.drop(columns=["Fwd Header Length.1"], errors="ignore")
    chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()

    # Filter known CICFlowMeter corruption where Flow Duration or rates are negative
    for non_neg_col in ("Flow Duration", "Flow Bytes/s", "Flow Packets/s"):
        if non_neg_col in chunk.columns and pd.api.types.is_numeric_dtype(chunk[non_neg_col]):
            chunk = chunk[chunk[non_neg_col] >= 0]

    n_after_na = len(chunk)

    chunk["Label"] = chunk["Label"].astype(str).str.strip()
    chunk["family"] = chunk["Label"].map(label_to_family)

    if "Timestamp" in chunk.columns:
        chunk["Timestamp"] = pd.to_datetime(
            chunk["Timestamp"], format="mixed", dayfirst=True, errors="coerce"
        )

    float_cols = chunk.select_dtypes("float64").columns
    chunk[float_cols] = chunk[float_cols].astype("float32")

    if deduplicate_features:
        # Collapse identical flood/scan feature vectors within the chunk while keeping Label/family
        dedup_cols = [c for c in chunk.columns if c not in ID_META_COLUMNS]
        chunk = chunk.drop_duplicates(subset=dedup_cols)
    else:
        chunk = chunk.drop_duplicates()

    if stats is not None:
        stats["rows_raw"] += n_raw
        stats["missing_removed"] += n_raw - n_after_na
        stats["duplicates_removed"] += n_after_na - len(chunk)
        stats["rows_clean"] += len(chunk)
    return chunk


def stratified_sample(chunk: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """Keep the same fraction of every family (used for uniform stratified sampling)."""
    if chunk.empty:
        return chunk
    return chunk.groupby("family", group_keys=False).sample(
        frac=fraction, random_state=config.SEED
    )


def smart_chunk_sample(
    chunk: pd.DataFrame,
    retention: dict[str, float] | None = None,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Class-aware chunk sampling: keep 100% of rare attacks (BruteForce, Other) and
    downsample only massive repetitive classes (Benign, DoS, PortScan).
    """
    if chunk.empty:
        return chunk
    retention = retention or config.FAMILY_CHUNK_RETENTION
    parts = []
    for family, group in chunk.groupby("family", sort=False):
        frac = float(retention.get(family, 1.0))
        if frac >= 1.0:
            parts.append(group)
        elif frac > 0.0 and len(group) > 0:
            # Ensure at least 1 row is kept when a rare class appears in a chunk
            n_keep = max(1, int(round(len(group) * frac)))
            parts.append(group.sample(n=min(len(group), n_keep), random_state=seed))
    if not parts:
        return chunk.iloc[0:0]
    return pd.concat(parts, ignore_index=True)


def _distribute_chunk_to_shards(
    chunk: pd.DataFrame,
    shard_writers: list[pq.ParquetWriter | None],
    shard_paths: list[Path],
    schema: pa.Schema,
    rng: np.random.Generator,
) -> None:
    """Spread a chunk's rows evenly across all N shards, stratified by attack family.

    Because each CSV file in CICIDS2017 only contains 1 attack type (e.g. Tuesday only has
    BruteForce, Wednesday only has DoS, Friday only has PortScan), distributing every chunk
    across all N shards guarantees that every shard ends up with all attack families from all files.
    """
    n_shards = len(shard_paths)
    shard_Buckets = [[] for _ in range(n_shards)]

    for _, group in chunk.groupby("family", sort=False):
        shuffled = group.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))
        assignments = np.arange(len(shuffled)) % n_shards
        rng.shuffle(assignments)
        for s_idx in range(n_shards):
            sub = shuffled.iloc[assignments == s_idx]
            if not sub.empty:
                shard_Buckets[s_idx].append(sub)

    for s_idx, sub_list in enumerate(shard_Buckets):
        if not sub_list:
            continue
        shard_df = pd.concat(sub_list, ignore_index=True)
        table = pa.Table.from_pandas(shard_df, schema=schema, preserve_index=False)
        if shard_writers[s_idx] is None:
            shard_writers[s_idx] = pq.ParquetWriter(shard_paths[s_idx], schema)
        shard_writers[s_idx].write_table(table)


def build_sample(
    raw_dir: Path = config.DATA_RAW,
    fraction: float | None = None,
    chunk_rows: int = config.CHUNK_ROWS,
    num_shards: int = config.NUM_SHARDS,
    max_per_family: dict[str, int] | None = None,
):
    """Stream all raw CSV files in chunks, mix across shards, enforce per-family caps,
    and write data/processed/sample.parquet out-of-core.
    """
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    max_per_family = max_per_family or config.MAX_ROWS_PER_FAMILY
    stats = {
        "rows_raw": 0,
        "missing_removed": 0,
        "duplicates_removed": 0,
        "rows_clean": 0,
        "rows_sample": 0,
        "sample_fraction": 0.0,
        "chunk_rows": chunk_rows,
        "num_shards": num_shards,
        "family_counts": {},
        "files": {},
    }

    shards_dir = config.DATA_PROCESSED / "shards"
    if shards_dir.exists():
        shutil.rmtree(shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    shard_paths = [shards_dir / f"shard_{i:02d}.parquet" for i in range(num_shards)]
    shard_writers: list[pq.ParquetWriter | None] = [None] * num_shards
    master_schema: pa.Schema | None = None
    rng = np.random.default_rng(config.SEED)

    try:
        # Pass 1: Stream each CSV in chunks, clean, sample, and distribute across N shards
        for path in files:
            print(f"Streaming {path.name} in chunks of {chunk_rows:,} rows ...")
            file_rows, n_cols = 0, 0
            for raw_chunk in pd.read_csv(
                path, chunksize=chunk_rows, encoding="latin-1", low_memory=False
            ):
                file_rows += len(raw_chunk)
                n_cols = raw_chunk.shape[1]

                cleaned = clean_chunk(
                    raw_chunk, stats, deduplicate_features=(fraction is None)
                )
                if cleaned.empty:
                    continue

                if fraction is not None:
                    sampled = stratified_sample(cleaned, fraction)
                else:
                    sampled = smart_chunk_sample(cleaned, seed=config.SEED)

                if sampled.empty:
                    continue

                if master_schema is None:
                    master_schema = pa.Table.from_pandas(
                        sampled, preserve_index=False
                    ).schema

                _distribute_chunk_to_shards(
                    sampled, shard_writers, shard_paths, master_schema, rng
                )

            stats["files"][path.name] = {"rows": file_rows, "columns": n_cols}
    finally:
        for w in shard_writers:
            if w is not None:
                w.close()

    # Pass 2: Read one mixed shard at a time, enforce per-shard family caps, shuffle,
    # and stream directly into sample.parquet
    if config.SAMPLE_PATH.exists():
        config.SAMPLE_PATH.unlink()

    family_totals: dict[str, int] = {}
    per_shard_cap = {
        fam: max(1, int(np.ceil(cap / num_shards)))
        for fam, cap in max_per_family.items()
    }

    sample_writer: pq.ParquetWriter | None = None
    final_shards = []
    try:
        for s_idx, s_path in enumerate(shard_paths):
            if not s_path.exists():
                continue
            shard_df = pd.read_parquet(s_path)
            if fraction is None:
                # Apply per-shard family cap so no flood class overwhelms BruteForce
                capped_parts = []
                for fam, grp in shard_df.groupby("family", sort=False):
                    cap = per_shard_cap.get(fam, len(grp))
                    if len(grp) > cap:
                        grp = grp.sample(n=cap, random_state=config.SEED + s_idx)
                    capped_parts.append(grp)
                shard_df = (
                    pd.concat(capped_parts, ignore_index=True)
                    if capped_parts
                    else shard_df
                )

            # Shuffle the mixed shard so rows from Monday, Tuesday, Wednesday, Friday are interleaved
            shard_df = shard_df.sample(
                frac=1.0, random_state=config.SEED + s_idx
            ).reset_index(drop=True)

            for fam, cnt in shard_df["family"].value_counts().items():
                family_totals[fam] = family_totals.get(fam, 0) + int(cnt)

            table = pa.Table.from_pandas(
                shard_df, schema=master_schema, preserve_index=False
            )
            if sample_writer is None:
                sample_writer = pq.ParquetWriter(config.SAMPLE_PATH, master_schema)
            sample_writer.write_table(table)
            final_shards.append(shard_df)
    finally:
        if sample_writer is not None:
            sample_writer.close()
        shutil.rmtree(shards_dir, ignore_errors=True)

    sample = (
        pd.concat(final_shards, ignore_index=True)
        if final_shards
        else pd.DataFrame()
    )
    stats["rows_sample"] = int(sum(family_totals.values()))
    stats["sample_fraction"] = (
        round(stats["rows_sample"] / max(stats["rows_clean"], 1), 4)
        if fraction is None
        else fraction
    )
    stats["family_counts"] = family_totals
    return sample, stats


def _run_cleaning() -> None:
    sample, stats = build_sample()

    counts = pd.Series(stats["family_counts"]).sort_values(ascending=False)
    print("\nRows per family in the mixed, chunk-processed sample:")
    for family, n in counts.items():
        print(f"  {family:<11}{n:>9,}  ({n / max(len(sample), 1):.1%})")
    print(f"  {'TOTAL':<11}{len(sample):>9,}")
    print(f"\nSaved chunked & interleaved sample to {config.SAMPLE_PATH}")

    config.RESULTS.mkdir(parents=True, exist_ok=True)
    stats_path = config.RESULTS / "preprocessing_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Saved {stats_path}")

    try:
        import mlflow

        mlflow.log_params(
            {
                "sampling_fraction": stats["sample_fraction"],
                "random_seed": config.SEED,
                "chunk_rows": config.CHUNK_ROWS,
                "num_shards": config.NUM_SHARDS,
                "input_file_count": len(stats["files"]),
                "input_files": ", ".join(stats["files"]),
            }
        )
        for metric in (
            "rows_raw",
            "missing_removed",
            "duplicates_removed",
            "rows_clean",
            "rows_sample",
        ):
            mlflow.log_metric(metric, stats[metric])
        for family, row_count in counts.items():
            mlflow.log_metric(f"sample_rows_{family}", int(row_count))
        mlflow.log_artifact(str(stats_path), artifact_path="preprocessing")
    except ImportError:
        pass


def main() -> None:
    try:
        import mlflow
        from src.mlflow_tracking import configure_tracking

        configure_tracking()
        with mlflow.start_run(run_name="data-cleaning"):
            _run_cleaning()
    except ImportError:
        _run_cleaning()


if __name__ == "__main__":
    main()