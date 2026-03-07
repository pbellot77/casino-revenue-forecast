# src/train.py

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

FEATURES = [
    "is_weekend",
    "promo_spend",
    "table_occupancy",
    "slot_occupancy",
    "local_event",
]
TARGET = "revenue"
DATA_PATH = "data/casino_revenue.csv"
MODEL_PATH = "models/model_v1.joblib"
METADATA_PATH = "models/metadata.json"


def train() -> None:
    logger.info("Loading dataset from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info("Train size: %d  |  Test size: %d", len(X_train), len(X_test))

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
    )

    logger.info("Training GradientBoostingRegressor...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    logger.info("Test MAE: $%.2f", mae)

    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info("Model artifact saved -> %s", MODEL_PATH)

    metadata = {
        "model_version": "v1",
        "model_type": "GradientBoostingRegressor",
        "features": FEATURES,
        "target": TARGET,
        "test_mae": round(mae, 2),
        "trained_at": datetime.utcnow().isoformat(),
        "data_path": DATA_PATH,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved -> %s", METADATA_PATH)


if __name__ == "__main__":
    train()
