import logging
from pathlib import Path

import joblib

from house_price_predictor.data import load_data
from house_price_predictor.eda import create_visualizations, inspect_data
from house_price_predictor.model import build_model, evaluate_model
from house_price_predictor.preprocessing import split_data

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = REPORTS_DIR / "house_price_model.joblib"


def train() -> dict[str, float]:
    """Run data analysis, train the model, and save artifacts."""
    logger.info("Loading dataset")
    data = load_data()

    logger.info("Inspecting dataset")
    inspect_data(data)
    create_visualizations(data, REPORTS_DIR)

    logger.info("Splitting data")
    X_train, X_test, y_train, y_test = split_data(data)

    logger.info("Training model")
    model = build_model()
    model.fit(X_train, y_train)

    logger.info("Evaluating model")
    metrics = evaluate_model(model, X_test, y_test)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    logger.info("Model saved to %s", MODEL_PATH)

    for name, value in metrics.items():
        logger.info("%s: %.4f", name.upper(), value)

    return metrics
