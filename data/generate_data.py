# data/generate_data.py

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


def generate_casino_revenue(
    n_days: int = 365,
    seed: int = 42,
    output_path: str = "data/casino_revenue.csv"
) -> pd.DataFrame:
    logger.info(f"Generating {n_days} days of synthetic casino data (seed={seed})")

    np.random.seed(seed)
    random.seed(seed)

    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")

    is_weekend = (dates.dayofweek >= 5).astype(int)
    promo_spend = np.random.uniform(5_000, 20_000, n_days).round(2)
    table_occupancy = np.clip(
        np.random.normal(loc=0.65, scale=0.12, size=n_days), 0.1, 1.0
    ).round(4)
    slot_occupancy = np.clip(
        np.random.normal(loc=0.60, scale=0.15, size=n_days), 0.1, 1.0
    ).round(4)
    local_event = np.random.choice([0, 1], size=n_days, p=[0.80, 0.20])

    base_revenue = 50_000
    revenue = (
        base_revenue
        + is_weekend         * 15_000
        + promo_spend        * 1.8
        + table_occupancy    * 30_000
        + slot_occupancy     * 20_000
        + local_event        * 12_000
        + np.random.normal(loc=0, scale=3_000, size=n_days)
    ).round(2)

    df = pd.DataFrame({
        "date":             dates,
        "is_weekend":       is_weekend,
        "promo_spend":      promo_spend,
        "table_occupancy":  table_occupancy,
        "slot_occupancy":   slot_occupancy,
        "local_event":      local_event,
        "revenue":          revenue,
    })

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    logger.info(f"Dataset saved -> {output}  |  shape: {df.shape}")
    logger.info(f"Revenue range: ${df['revenue'].min():,.0f} - ${df['revenue'].max():,.0f}")

    return df


if __name__ == "__main__":
    generate_casino_revenue()
