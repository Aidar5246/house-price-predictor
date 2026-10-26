import numpy as np
import pandas as pd
import pytest

from house_price_predictor.data import TARGET_COLUMN, validate_data
from house_price_predictor.model import build_model, evaluate_model
from house_price_predictor.preprocessing import split_data


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create a small synthetic housing dataset."""
    return pd.DataFrame(
        {
            "MedInc": np.linspace(1.0, 5.0, 50),
            "HouseAge": np.linspace(5.0, 40.0, 50),
            "AveRooms": np.linspace(3.0, 8.0, 50),
            TARGET_COLUMN: np.linspace(0.5, 3.0, 50),
        }
    )


def test_validate_data_accepts_valid_data(sample_data: pd.DataFrame) -> None:
    validate_data(sample_data)


def test_validate_data_rejects_empty_data() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        validate_data(pd.DataFrame())


def test_split_data_returns_train_and_test_sets(
    sample_data: pd.DataFrame,
) -> None:
    X_train, X_test, y_train, y_test = split_data(sample_data)

    assert len(X_train) == 40
    assert len(X_test) == 10
    assert TARGET_COLUMN not in X_train.columns
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_split_data_rejects_invalid_test_size(
    sample_data: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="test_size"):
        split_data(sample_data, test_size=1.0)


def test_model_training_and_evaluation(sample_data: pd.DataFrame) -> None:
    X_train, X_test, y_train, y_test = split_data(sample_data)

    model = build_model()
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    assert set(metrics) == {"mae", "rmse", "r2"}
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0


def test_pipeline_handles_missing_features(sample_data: pd.DataFrame) -> None:
    X_train, X_test, y_train, y_test = split_data(sample_data)

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train.iloc[0, 0] = np.nan
    X_test.iloc[0, 0] = np.nan

    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert np.isfinite(predictions).all()
