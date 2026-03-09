# tests/test_data.py

import pandas as pd
from pathlib import Path


def test_csv_exists():
    assert Path("data/casino_revenue.csv").exists(), "Dataset CSV not found"


def test_csv_schema():
    df = pd.read_csv("data/casino_revenue.csv")
    expected_columns = [
        "date", "is_weekend", "promo_spend",
        "table_occupancy", "slot_occupancy", "local_event", "revenue"
    ]
    assert list(df.columns) == expected_columns


def test_csv_row_count():
    df = pd.read_csv("data/casino_revenue.csv")
    assert len(df) == 365, f"Expected 365 rows, got {len(df)}"


def test_revenue_positive():
    df = pd.read_csv("data/casino_revenue.csv")
    assert (df["revenue"] > 0).all(), "All revenue values should be positive"


def test_occupancy_bounds():
    df = pd.read_csv("data/casino_revenue.csv")
    assert df["table_occupancy"].between(0, 1).all()
    assert df["slot_occupancy"].between(0, 1).all()
