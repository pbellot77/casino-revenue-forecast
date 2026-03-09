# src/retrain.py

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
METADATA_PATH = "models/metadata.json"


def get_next_version() -> str:
    if not Path(METADATA_PATH).exists():
        return "v2"
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    current = metadata.get("model_version", "v1")
    num = int(current.replace("v", "")) + 1
    return f"v{num}"


def retrain() -> None:
    version = get_next_version()
    model_path = f"models/model_{version}.joblib"

    logger.info("Starting retraining -> %s", model_path)

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=4,
        random_state=42,
    )

    logger.info("Training new model...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    logger.info("Retrained model MAE: $%.2f", mae)

    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Model artifact saved -> %s", model_path)

    metadata = {
        "model_version": version,
        "model_type": "GradientBoostingRegressor",
        "features": FEATURES,
        "target": TARGET,
        "test_mae": round(mae, 2),
        "trained_at": datetime.utcnow().isoformat(),
        "data_path": DATA_PATH,
        "retrain_reason": "drift_detected",
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata updated -> %s", METADATA_PATH)
    logger.info("Retraining complete. New version: %s", version)


if __name__ == "__main__":
    retrain()
