"""Shared paths and settings. Every phase imports from here so results stay reproducible."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DOCS = PROJECT_ROOT / "docs"
RESULTS = PROJECT_ROOT / "results"
MLFLOW_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlruns' / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACT_ROOT = (PROJECT_ROOT / "mlruns" / "artifacts").as_uri()
MLFLOW_EXPERIMENT = "SENTRIX"

SAMPLE_PATH = DATA_PROCESSED / "sample.parquet"

SEED = 42  # use this everywhere a random seed is needed

# Attack families used throughout the project (see clean.label_to_family)
FAMILIES = ["Benign", "DoS", "PortScan", "BruteForce", "Other"]

# Keep 10% of every family from the raw files. This is proportional stratified
# sampling, so rare attacks stay rare, as in real traffic. Raise it for a bigger
# sample, lower it if your laptop runs out of memory.
SAMPLE_FRACTION = 0.10

# The raw files are read in pieces of this many rows to keep memory low.
CHUNK_ROWS = 200_000