"""Central configuration: paths, dataset URLs, and forecasting constants."""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

for _p in (DATA_RAW, DATA_PROCESSED):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data sources — Vadimkin/ukrainian-air-raid-sirens-dataset (raw CSV on GitHub)
# ---------------------------------------------------------------------------
_RAW_BASE = (
    "https://raw.githubusercontent.com/Vadimkin/"
    "ukrainian-air-raid-sirens-dataset/main/datasets"
)

DATASET_URLS = {
    # Volunteer data is OBLAST-level for the whole period (most consistent for
    # a long, uniform time series) and starts earliest (25 Feb 2022).
    "volunteer_en": f"{_RAW_BASE}/volunteer_data_en.csv",
    "volunteer_uk": f"{_RAW_BASE}/volunteer_data_uk.csv",
    # Official data is authoritative but moved to raion-level in Dec 2025.
    "official_en": f"{_RAW_BASE}/official_data_en.csv",
    "official_uk": f"{_RAW_BASE}/official_data_uk.csv",
}

DEFAULT_DATASET = "volunteer_en"

# Local cache filenames
def raw_csv_path(dataset_key: str) -> Path:
    return DATA_RAW / f"{dataset_key}.csv"

SERIES_PARQUET = DATA_PROCESSED / "series.parquet"
EVENTS_PARQUET = DATA_PROCESSED / "events.parquet"
MODEL_METRICS_JSON = DATA_PROCESSED / "model_metrics.json"

# ---------------------------------------------------------------------------
# Time-series / forecasting constants
# ---------------------------------------------------------------------------
# Source timestamps are UTC; we display in Kyiv time.
SOURCE_TZ = "UTC"
DISPLAY_TZ = "Europe/Kyiv"

# Forecast target: "will an alert be ACTIVE at some point in the next H hours?"
FORECAST_HORIZON_HOURS = 6

# Lag features (hours of recent activity to look back on)
LAG_HOURS = [1, 2, 3, 6, 12, 24, 48]
ROLLING_WINDOWS = [3, 6, 12, 24, 72]
