"""Project-wide constants.

Started as a four-station Kolkata MVP; now also supports any place in the world
through src/world_aqi.py (Open-Meteo data, Indian CPCB AQI scale).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT / "data" / "raw"
PROCESSED_PATH = ROOT / "data" / "processed" / "model_dataset.csv"
AUDIT_PATH = ROOT / "reports" / "data_quality_report.json"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
DEFAULT_PLACES_FILE = ROOT / "places.txt"
FORECAST_DIR = ROOT / "forecasts"


SUPPORTED_STATIONS = (
    "Ballygunge",
    "Bidhannagar",
    "Fort William",
    "Rabindra Bharati University",
)

KOLKATA_LATITUDE = 22.556
KOLKATA_LONGITUDE = 88.338
TARGET_COLUMN = "aqi_next_day"


LAGS = (1, 2, 3)
ROLLING_WINDOWS = (3, 7)

HISTORY_PADDING_DAYS = max(max(LAGS), max(ROLLING_WINDOWS)) + 1


AQI_BANDS = (
    (50, "Good"),
    (100, "Satisfactory"),
    (200, "Moderate"),
    (300, "Poor"),
    (400, "Very Poor"),
    (500, "Severe"),
)


SOURCE_STATION = "monitoring_station"
SOURCE_WORLD = "open_meteo_model"



NON_FEATURE_COLUMNS = {
    "date", "station", TARGET_COLUMN, "daily_aqi_definition",
    "source", "aqi_category", "latitude", "longitude",
}


FORECAST_HORIZON_DAYS = 365
MIN_HISTORY_DAYS = 365          
DEFAULT_HISTORY_START = "2022-08-01"
