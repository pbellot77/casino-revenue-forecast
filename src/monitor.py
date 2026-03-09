# src/monitor.py

import logging
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_PATH = "models/model_v1.joblib"
DATA_PATH = "data/casino_revenue.csv"
MAE_THRESHOLD = 5000.0
CONSECUTIVE_DAYS_LIMIT = 3

FEATURES = [
    "is_weekend",
    "promo_spend",
    "table_occupancy",
    "slot_occupancy",
    "local_event",
]
TARGET = "revenue"


def load_model():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def compute_daily_mae(df: pd.DataFrame, model) -> pd.Series:
    logger.info("Computing daily MAE over %d records", len(df))
    df = df.copy()
    df["predicted"] = model.predict(df[FEATURES])
    df["abs_error"] = (df[TARGET] - df["predicted"]).abs()
    daily_mae = df.groupby("date")["abs_error"].mean()
    return daily_mae


def detect_drift(daily_mae: pd.Series) -> bool:
    logger.info("Running drift detection (threshold=$%.0f, window=%d days)",
                MAE_THRESHOLD, CONSECUTIVE_DAYS_LIMIT)

    breaches = daily_mae > MAE_THRESHOLD
    consecutive = 0
    max_consecutive = 0

    for breach in breaches:
        if breach:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0

    logger.info("Max consecutive threshold breaches: %d", max_consecutive)

    if max_consecutive >= CONSECUTIVE_DAYS_LIMIT:
        logger.warning(
            "DRIFT DETECTED: MAE exceeded $%.0f for %d consecutive days",
            MAE_THRESHOLD, max_consecutive
        )
        return True

    logger.info("No drift detected. Model performance is within acceptable bounds.")
    return False


def trigger_retraining() -> None:
    logger.warning("Triggering automated retraining...")
    result = subprocess.run(
        [sys.executable, "src/retrain.py"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        logger.info("Retraining completed successfully.")
        logger.info(result.stdout)
    else:
        logger.error("Retraining failed: %s", result.stderr)


def run_monitoring() -> None:
    logger.info("=== Starting monitoring run ===")

    model = load_model()
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    daily_mae = compute_daily_mae(df, model)

    logger.info("Sample of daily MAE (last 7 days):")
    for date, mae in daily_mae.tail(7).items():
        flag = " *** ABOVE THRESHOLD ***" if mae > MAE_THRESHOLD else ""
        logger.info("  %s  MAE: $%.2f%s", date.date(), mae, flag)

    drift_detected = detect_drift(daily_mae)

    if drift_detected:
        trigger_retraining()
    else:
        logger.info("=== Monitoring run complete. No action needed. ===")


if __name__ == "__main__":
    run_monitoring()
