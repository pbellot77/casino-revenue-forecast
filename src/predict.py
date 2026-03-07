# src/predict.py

import logging
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_PATH = "models/model_v1.joblib"

FEATURES = [
    "is_weekend",
    "promo_spend",
    "table_occupancy",
    "slot_occupancy",
    "local_event",
]


def load_model():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
    return model


def predict_revenue(
    is_weekend: int,
    promo_spend: float,
    table_occupancy: float,
    slot_occupancy: float,
    local_event: int,
) -> float:
    model = load_model()
    input_df = pd.DataFrame([{
        "is_weekend":       is_weekend,
        "promo_spend":      promo_spend,
        "table_occupancy":  table_occupancy,
        "slot_occupancy":   slot_occupancy,
        "local_event":      local_event,
    }])
    prediction = model.predict(input_df)[0]
    return round(float(prediction), 2)
