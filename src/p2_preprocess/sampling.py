"""Phase 2: stratified train/test split and feature scaling."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config

# Columns that are identifiers, labels or meaningless as distances, so never model inputs.
# Source Port is excluded because it is essentially random for each connection.
# The IP and timestamp columns stay in the saved files: Phase 4 needs them.
NON_FEATURE_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Label",
    "family",
    "Source Port",
]


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """Split into train and test so every family has the same share in both parts."""
    return train_test_split(
        df, test_size=test_size, stratify=df["family"], random_state=config.SEED
    )


def feature_columns(train: pd.DataFrame) -> list:
    """Numeric model inputs: no identifiers, and no column that never changes."""
    numeric = train.select_dtypes("number").columns
    return [c for c in numeric if c not in NON_FEATURE_COLUMNS and train[c].nunique() > 1]


def fit_scaler(train: pd.DataFrame, cols: list) -> StandardScaler:
    """Learn the mean and spread from the TRAINING data only.

    Fitting on all the data would let test-set information leak into training and make
    every later score look better than it really is.
    """
    return StandardScaler().fit(train[cols])


def class_proportions(df: pd.DataFrame) -> pd.Series:
    return df["family"].value_counts(normalize=True)