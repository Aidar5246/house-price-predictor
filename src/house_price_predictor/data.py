from typing import cast
import pandas as pd
from sklearn.datasets import fetch_california_housing

TARGET_COLUMN = "MedHouseVal"


def load_data() -> pd.DataFrame:
    """Load the California Housing dataset."""
    dataset = fetch_california_housing(as_frame=True)
    return cast(pd.DataFrame, dataset.frame)


def validate_data(df: pd.DataFrame) -> None:
    """Validate the basic structure of the dataset."""
    if df.empty:
        raise ValueError("Dataset must not be empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing.")

    if df.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names.")

    if df[TARGET_COLUMN].isna().all():
        raise ValueError("Target column contains only missing values.")
