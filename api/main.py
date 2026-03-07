# api/main.py

import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict_revenue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Casino Revenue Forecast API",
    description="Predicts daily casino revenue based on operational features.",
    version="1.0.0",
)

METADATA_PATH = "models/metadata.json"


class PredictRequest(BaseModel):
    is_weekend: int = Field(..., ge=0, le=1, description="1 if weekend, 0 if weekday")
    promo_spend: float = Field(..., gt=0, description="Daily promotional spend in USD")
    table_occupancy: float = Field(..., ge=0, le=1, description="Fraction of tables occupied")
    slot_occupancy: float = Field(..., ge=0, le=1, description="Fraction of slots occupied")
    local_event: int = Field(..., ge=0, le=1, description="1 if local event, 0 if not")


class PredictResponse(BaseModel):
    predicted_revenue: float
    currency: str = "USD"


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_type: str
    test_mae: float


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    logger.info("Prediction request received: %s", request.model_dump())
    try:
        revenue = predict_revenue(
            is_weekend=request.is_weekend,
            promo_spend=request.promo_spend,
            table_occupancy=request.table_occupancy,
            slot_occupancy=request.slot_occupancy,
            local_event=request.local_event,
        )
        logger.info("Predicted revenue: $%.2f", revenue)
        return PredictResponse(predicted_revenue=revenue)
    except Exception as e:
        logger.error("Prediction failed: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
def health():
    if not Path(METADATA_PATH).exists():
        raise HTTPException(status_code=503, detail="Model metadata not found")
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return HealthResponse(
        status="ok",
        model_version=metadata["model_version"],
        model_type=metadata["model_type"],
        test_mae=metadata["test_mae"],
    )
