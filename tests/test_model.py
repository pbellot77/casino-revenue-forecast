# tests/test_model.py

import pytest
import joblib
import pandas as pd
from pathlib import Path


def test_model_artifact_exists():
    assert Path("models/model_v1.joblib").exists(), "Model artifact not found"


def test_metadata_exists():
    assert Path("models/metadata.json").exists(), "Metadata file not found"


def test_model_loads():
    model = joblib.load("models/model_v1.joblib")
    assert model is not None


def test_model_predicts():
    model = joblib.load("models/model_v1.joblib")
    sample = pd.DataFrame([{
        "is_weekend": 1,
        "promo_spend": 12000,
        "table_occupancy": 0.82,
        "slot_occupancy": 0.74,
        "local_event": 1,
    }])
    prediction = model.predict(sample)
    assert len(prediction) == 1
    assert prediction[0] > 0, "Prediction should be positive"


def test_prediction_in_realistic_range():
    model = joblib.load("models/model_v1.joblib")
    sample = pd.DataFrame([{
        "is_weekend": 0,
        "promo_spend": 10000,
        "table_occupancy": 0.65,
        "slot_occupancy": 0.60,
        "local_event": 0,
    }])
    prediction = model.predict(sample)[0]
    assert 50_000 < prediction < 300_000, f"Prediction ${prediction:,.0f} outside realistic range"
