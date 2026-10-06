"""Import, audit, and feature construction for station-level AQI forecasting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import KOLKATA_LATITUDE, KOLKATA_LONGITUDE, LAGS, ROLLING_WINDOWS

WEATHER_COLUMNS = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "pressure_msl_mean",
]


COLUMN_ALIASES = {
    "date": "date", "timestamp": "date", "datetime": "date", "date_time": "date", "date_ist": "date",
    "station": "station", "station_name": "station", "monitoring_station": "station",
    "aqi": "aqi", "air_quality_index": "aqi",
    "pm2_5": "pm25", "pm2.5": "pm25", "pm25": "pm25", "pm10": "pm10",
    "no2": "no2", "so2": "so2", "co": "co", "o3": "o3", "nh3": "nh3",
    "temperature_c": "temperature_c", "temperature": "temperature_c",
    "humidity_pct": "humidity_pct", "humidity": "humidity_pct",
    "wind_speed_ms": "wind_speed_ms", "wind_speed": "wind_speed_ms",
    "wind_direction_deg": "wind_direction_deg", "wind_direction": "wind_direction_deg",
    "rain_mm": "rain_mm", "rainfall": "rain_mm",
}


def _normalise_column_name(name: object) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def read_raw_files(input_dir: Path) -> pd.DataFrame:
    """Read all CSV/XLSX source exports without changing the originals."""
    paths = sorted(path for path in input_dir.glob("*") if path.suffix.lower() in {".csv", ".xlsx"})
    if not paths:
        raise FileNotFoundError(
            f"No CSV or XLSX files found in {input_dir}. Place original portal exports there."
        )
    frames: list[pd.DataFrame] = []
    for path in paths:
        if path.suffix.lower() == ".csv":
            source_frames = [("csv", pd.read_csv(path))]
        else:
            book = pd.ExcelFile(path)
            source_frames = [(sheet, pd.read_excel(path, sheet_name=sheet)) for sheet in book.sheet_names]
        for sheet, frame in source_frames:
            frame["source_file"] = path.name
            frame["source_sheet"] = sheet
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def standardise_air_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Create a canonical table while retaining provenance columns."""
    data = raw.copy()
    data.columns = [_normalise_column_name(column) for column in data.columns]
    data = data.rename(columns={column: COLUMN_ALIASES.get(column, column) for column in data.columns})
    if "date" not in data or "aqi" not in data:
        raise ValueError("Source data must include a date/timestamp column and an AQI column.")
    if "station" not in data:
        raise ValueError("Source data must include a station or station_name column.")
    data["date"] = pd.to_datetime(data["date"], errors="coerce", dayfirst=True)
    data["station"] = data["station"].astype(str).str.strip()
    numeric_columns = [
        "aqi", "pm25", "pm10", "no2", "so2", "co", "o3", "nh3",
        "temperature_c", "humidity_pct", "wind_speed_ms", "wind_direction_deg", "rain_mm",
    ]
    for column in numeric_columns:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.loc[data["date"].notna() & data["station"].notna()].copy()
    data["date"] = data["date"].dt.tz_localize(None)
    return data


def create_data_quality_report(data: pd.DataFrame) -> dict[str, object]:
    """Return source-quality diagnostics without silently repairing data."""
    report: dict[str, object] = {"rows_read": int(len(data)), "stations": {}}
    for station, group in data.groupby("station", dropna=False):
        group = group.sort_values("date")
        report["stations"][str(station)] = {
            "rows": int(len(group)),
            "start": group["date"].min().isoformat() if not group.empty else None,
            "end": group["date"].max().isoformat() if not group.empty else None,
            "duplicate_timestamps": int(group.duplicated(["station", "date"]).sum()),
            "missing_by_column": {key: int(value) for key, value in group.isna().sum().items()},
            "aqi_outside_0_500": int((~group["aqi"].between(0, 500)).sum()),
        }
    return report


def aggregate_to_daily(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sub-daily measurements to daily means and document the target definition."""
    data = data.copy()
    data["date"] = data["date"].dt.normalize()
    numeric = data.select_dtypes(include="number").columns.tolist()
    daily = data.groupby(["station", "date"], as_index=False)[numeric].mean()
    # This is a daily mean of reported AQI, not a replacement official AQI calculation.
    daily["daily_aqi_definition"] = "mean_of_source_observations"
    return daily.sort_values(["station", "date"]).reset_index(drop=True)


def fetch_daily_weather(
    start: pd.Timestamp,
    end: pd.Timestamp,
    latitude: float = KOLKATA_LATITUDE,
    longitude: float = KOLKATA_LONGITUDE,
    timezone: str = "Asia/Kolkata",
) -> pd.DataFrame:
    """Fetch reproducible daily weather from Open-Meteo Archive.

    Defaults reproduce the original Kolkata behaviour; pass latitude/longitude (and
    timezone="auto") for any other place in the world.
    """
    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "daily": ",".join(WEATHER_COLUMNS),
            "timezone": timezone,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if "daily" not in payload:
        raise RuntimeError(f"Open-Meteo returned no daily weather: {json.dumps(payload)[:300]}")
    weather = pd.DataFrame(payload["daily"]).rename(columns={"time": "date"})
    weather["date"] = pd.to_datetime(weather["date"])
    return weather


def build_features(daily: pd.DataFrame, weather: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Build station-wise features using only information available by the end of day t.

    `weather` may carry a `station` column (one weather series per place, needed for the world
    dataset); without it the same weather is joined to every station by date (original behaviour).
    require_target=False keeps rows whose next-day AQI is unknown, so the latest day can be forecast.
    """
    merge_keys = ["station", "date"] if "station" in weather.columns else ["date"]
    data = daily.merge(weather, on=merge_keys, how="left").sort_values(["station", "date"]).reset_index(drop=True)
    weather_mapping = {
        "temperature_c": "temperature_2m_mean",
        "humidity_pct": "relative_humidity_2m_mean",
        "wind_speed_ms": "wind_speed_10m_max",
    }
    for existing, archive in weather_mapping.items():
        if existing in data:
            data[archive] = data[existing].combine_first(data[archive])
    for _, group in data.groupby("station"):
        for lag in LAGS:
            data.loc[group.index, f"aqi_lag_{lag}"] = group["aqi"].shift(lag)
        for window in ROLLING_WINDOWS:
            data.loc[group.index, f"aqi_rolling_mean_{window}"] = group["aqi"].shift(1).rolling(window).mean()
        for pollutant in ["pm25", "pm10", "no2", "so2", "co", "o3", "nh3"]:
            if pollutant in group:
                data.loc[group.index, f"{pollutant}_lag_1"] = group[pollutant].shift(1)
        data.loc[group.index, "aqi_next_day"] = group["aqi"].shift(-1)
    if "wind_direction_deg" in data:
        angle = np.deg2rad(data["wind_direction_deg"])
        data["wind_direction_sin"] = np.sin(angle)
        data["wind_direction_cos"] = np.cos(angle)
    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    data["day_of_year_sin"] = np.sin(2 * np.pi * data["date"].dt.dayofyear / 365.25)
    data["day_of_year_cos"] = np.cos(2 * np.pi * data["date"].dt.dayofyear / 365.25)
    required = ["aqi_lag_3", "aqi_rolling_mean_7"] + (["aqi_next_day"] if require_target else [])
    return data.dropna(subset=required).copy()
