import pandas as pd
from sklearn.model_selection import train_test_split

from house_price_predictor.data import TARGET_COLUMN, validate_data


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Validate the data and split it into training and test sets."""
    validate_data(df)

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    clean_df = df.dropna(subset=[TARGET_COLUMN]).copy()

    if clean_df.empty:
        raise ValueError("No rows with valid target values remain.")

    features = clean_df.drop(columns=[TARGET_COLUMN])
    target = clean_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test
