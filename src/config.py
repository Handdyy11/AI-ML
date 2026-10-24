"""Shared paths and settings. Every phase imports from here so results stay reproducible."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SHARDS_DIR = DATA_PROCESSED / "shards"
DOCS = PROJECT_ROOT / "docs"
RESULTS = PROJECT_ROOT / "results"
MLFLOW_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlruns' / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACT_ROOT = (PROJECT_ROOT / "mlruns" / "artifacts").as_uri()
MLFLOW_EXPERIMENT = "SENTRIX"

SAMPLE_PATH = DATA_PROCESSED / "sample.parquet"
TRAIN_PATH = DATA_PROCESSED / "train.parquet"
TEST_PATH = DATA_PROCESSED / "test.parquet"

SEED = 42  # use this everywhere a random seed is needed

# Attack families used throughout the project (see clean.label_to_family)
FAMILIES = ["Benign", "DoS", "PortScan", "BruteForce", "Other"]

# Fallback flat sampling fraction kept for backward compatibility
SAMPLE_FRACTION = 0.10

# Read and process files in 50,000-row chunks so memory usage stays low (<150 MB)
CHUNK_ROWS = 50_000

# Number of mixed shard buckets used when streaming across multiple single-attack CSV files.
# Every chunk spreads its rows evenly across all shards so each shard holds all attack families.
NUM_SHARDS = 10

# Per-family retention inside each chunk:
# Keep 100% of rare attacks (BruteForce, Other), moderate shares of flood attacks (DoS, PortScan)
# after feature-level deduplication, and downsample repetitive Benign background traffic.
FAMILY_CHUNK_RETENTION = {
    "Benign": 0.08,
    "DoS": 0.35,
    "PortScan": 0.40,
    "BruteForce": 1.00,
    "Other": 1.00,
}

# Maximum total rows kept per family across the entire dataset so no single flood attack
# (like DoS or PortScan) overwhelms rare attacks (BruteForce) or slows down k-NN.
MAX_ROWS_PER_FAMILY = {
    "Benign": 85_000,
    "DoS": 45_000,
    "PortScan": 40_000,
    "BruteForce": 25_000,
    "Other": 25_000,
}