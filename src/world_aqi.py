"""AQI and air-quality category for any place in the world.

Data source: Open-Meteo (free, no API key): air-quality model + weather archive.
AQI is computed on the Indian CPCB 0-500 scale (max sub-index of the pollutants).

Command line:
    python -m src.world_aqi "Delhi" 2025-01-01 2025-03-31
    python -m src.world_aqi "London, UK" 2025-06-01 2025-06-30 --out london.csv

From other code:
    from src.world_aqi import build_place_features
    features, failures = build_place_features(["Delhi", "Tokyo, Japan"], "2025-01-01", "2025-03-31")
"""

from __future__ import annotations

import argparse
import re
import warnings

import numpy as np
import pandas as pd
import requests

from src.config import AQI_BANDS, HISTORY_PADDING_DAYS, SOURCE_WORLD
from src.data_pipeline import WEATHER_COLUMNS, build_features, fetch_daily_weather

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARIABLES = "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone,ammonia"
AQI_DEFINITION = "cpcb_max_subindex_of_daily_mean"


BREAKPOINTS = {
    "pm25_ug_m3": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200), (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500)],
    "pm10_ug_m3": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200), (251, 350, 201, 300), (351, 430, 301, 400), (431, 600, 401, 500)],
    "no2_ug_m3": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200), (181, 280, 201, 300), (281, 400, 301, 400), (401, 600, 401, 500)],
    "so2_ug_m3": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200), (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2400, 401, 500)],
    "o3_ug_m3": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200), (169, 208, 201, 300), (209, 748, 301, 400), (749, 1000, 401, 500)],
    "co_mg_m3": [(0, 1, 0, 50), (1.1, 2, 51, 100), (2.1, 10, 101, 200), (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 50, 401, 500)],
    "nh3_ug_m3": [(0, 200, 0, 50), (201, 400, 51, 100), (401, 800, 101, 200), (801, 1200, 201, 300), (1201, 1800, 301, 400), (1801, 2400, 401, 500)],
}


def categorize(aqi: float) -> str | float:
    """Map an AQI value to its level: Good ... Severe. Missing AQI stays missing."""
    if aqi is None or pd.isna(aqi):
        return np.nan
    for upper, label in AQI_BANDS:
        if aqi <= upper:
            return label
    return AQI_BANDS[-1][1]


def sub_index(column: str, concentration: float) -> float:
    if pd.isna(concentration):
        return np.nan
    for c_low, c_high, i_low, i_high in BREAKPOINTS[column]:
        if concentration <= c_high:
            if c_high == c_low:
                return float(i_low)
            return i_low + (i_high - i_low) * (max(concentration, c_low) - c_low) / (c_high - c_low)
    return 500.0


def compute_aqi(daily: pd.DataFrame) -> pd.Series:
    """CPCB AQI = highest pollutant sub-index; needs at least PM2.5 or PM10."""
    subs = pd.DataFrame({col: daily[col].map(lambda v, c=col: sub_index(c, v)) for col in BREAKPOINTS})
    aqi = subs.max(axis=1)
    aqi[daily["pm25_ug_m3"].isna() & daily["pm10_ug_m3"].isna()] = np.nan
    return aqi.round()


def geocode(place: str) -> dict:
    """Find a place. 'London, UK' / 'Springfield, Illinois' -> later parts narrow the match."""
    parts = [part.strip() for part in place.split(",") if part.strip()]
    if not parts:
        raise ValueError("Empty place name.")
    response = requests.get(
        GEOCODING_URL, params={"name": parts[0], "count": 10, "language": "en"}, timeout=30
    )
    response.raise_for_status()
    hits = response.json().get("results") or []
    if not hits:
        raise ValueError(f"Place not found: {place!r}")
    hints = [part.lower() for part in parts[1:]]

    def matches(hit: dict) -> bool:
        fields = {str(hit.get(key, "")).lower() for key in ("country", "country_code", "admin1", "admin2")}
        return any(hint in fields for hint in hints)

    chosen = next((hit for hit in hits if matches(hit)), hits[0]) if hints else hits[0]
    label = ", ".join(dict.fromkeys(p for p in (chosen["name"], chosen.get("admin1"), chosen.get("country")) if p))
    return {"label": label, "latitude": chosen["latitude"], "longitude": chosen["longitude"]}


AIR_CHUNK_DAYS = 180  # long histories are requested in pieces to keep each API call small


def _fetch_air_hourly(latitude: float, longitude: float, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    response = requests.get(
        AIR_QUALITY_URL,
        params={
            "latitude": latitude, "longitude": longitude,
            "start_date": start.date().isoformat(), "end_date": end.date().isoformat(),
            "hourly": HOURLY_VARIABLES, "timezone": "auto",
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if "hourly" not in payload:
        raise RuntimeError(f"Open-Meteo returned no air-quality data: {str(payload)[:300]}")
    return pd.DataFrame(payload["hourly"])


def fetch_air_daily(latitude: float, longitude: float, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Hourly model pollutants -> daily means in the project's column names, plus CPCB AQI."""
    chunks, chunk_start = [], start
    while chunk_start <= end:
        chunk_end = min(chunk_start + pd.Timedelta(days=AIR_CHUNK_DAYS - 1), end)
        chunks.append(_fetch_air_hourly(latitude, longitude, chunk_start, chunk_end))
        chunk_start = chunk_end + pd.Timedelta(days=1)
    hourly = pd.concat(chunks, ignore_index=True)
    hourly["date"] = pd.to_datetime(hourly["time"]).dt.normalize()
    daily = hourly.drop(columns="time").groupby("date", as_index=False).mean()
    daily = daily.rename(columns={
        "pm2_5": "pm25_ug_m3", "pm10": "pm10_ug_m3", "nitrogen_dioxide": "no2_ug_m3",
        "sulphur_dioxide": "so2_ug_m3", "ozone": "o3_ug_m3", "ammonia": "nh3_ug_m3",
    })
    daily["co_mg_m3"] = daily.pop("carbon_monoxide") / 1000.0  # ug/m3 -> mg/m3
    for column in BREAKPOINTS:  # ammonia is often unavailable outside Europe
        if column not in daily:
            daily[column] = np.nan
    daily["aqi"] = compute_aqi(daily)
    return daily


def fetch_place_history(place: str, start, end) -> pd.DataFrame:
    """Daily AQI history (station, date, aqi) for any place: the input of the 1-year outlook."""
    info = geocode(place)
    air = fetch_air_daily(info["latitude"], info["longitude"], pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize())
    air["station"] = info["label"]
    return air[["station", "date", "aqi"]]


def _fetch_recent_weather(latitude: float, longitude: float, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """The archive lags a few days behind; the forecast API covers the most recent days."""
    today = pd.Timestamp.today().normalize()
    response = requests.get(
        FORECAST_URL,
        params={
            "latitude": latitude, "longitude": longitude, "daily": ",".join(WEATHER_COLUMNS),
            "past_days": max((today - start).days, 0), "forecast_days": 1, "timezone": "auto",
        },
        timeout=30,
    )
    response.raise_for_status()
    weather = pd.DataFrame(response.json()["daily"]).rename(columns={"time": "date"})
    weather["date"] = pd.to_datetime(weather["date"])
    return weather[(weather["date"] >= start) & (weather["date"] <= end)]


def fetch_weather(latitude: float, longitude: float, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Archive for older days, forecast API for the last few days. Gaps stay NaN (model imputes them)."""
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=6)
    frames = []
    if start <= cutoff:
        try:
            frames.append(fetch_daily_weather(start, min(end, cutoff), latitude, longitude, "auto"))
        except Exception as exc:  # noqa: BLE001 - weather is optional, air quality is not
            warnings.warn(f"Weather archive unavailable: {exc}")
    if end > cutoff:
        try:
            frames.append(_fetch_recent_weather(latitude, longitude, max(start, cutoff + pd.Timedelta(days=1)), end))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Recent weather unavailable: {exc}")
    if not frames:
        return pd.DataFrame(columns=["date", *WEATHER_COLUMNS])
    return pd.concat(frames, ignore_index=True).drop_duplicates("date")


def build_place_features(
    places: list[str], start, end, require_target: bool = True
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Daily AQI + pollutants + weather + lag/rolling features for each place.

    Returns (features, failures). Same columns and feature definitions as the Kolkata dataset.
    require_target=False keeps the final day (whose next-day AQI is unknown) so it can be forecast.
    """
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    fetch_start = start - pd.Timedelta(days=HISTORY_PADDING_DAYS)
    daily_frames, weather_frames, failures = [], [], {}
    for place in places:
        try:
            info = geocode(place)
            air = fetch_air_daily(info["latitude"], info["longitude"], fetch_start, end)
            weather = fetch_weather(info["latitude"], info["longitude"], fetch_start, end)
        except Exception as exc:  # noqa: BLE001 - report per place, keep going
            failures[place] = str(exc)
            continue
        air["station"] = weather["station"] = info["label"]
        air["daily_aqi_definition"] = AQI_DEFINITION
        air["latitude"], air["longitude"] = info["latitude"], info["longitude"]
        air["source"] = SOURCE_WORLD
        daily_frames.append(air)
        weather_frames.append(weather)
    if not daily_frames:
        raise RuntimeError(f"No place could be fetched: {failures}")
    features = build_features(
        pd.concat(daily_frames, ignore_index=True),
        pd.concat(weather_frames, ignore_index=True),
        require_target=require_target,
    )
    features = features[features["date"] >= start].reset_index(drop=True)
    features.insert(features.columns.get_loc("aqi") + 1, "aqi_category", features["aqi"].map(categorize))
    return features, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily AQI and air-quality category for any place in the world.")
    parser.add_argument("place", help='e.g. "Delhi" or "London, UK"')
    parser.add_argument("start", help="YYYY-MM-DD")
    parser.add_argument("end", help="YYYY-MM-DD")
    parser.add_argument("--out", help="output CSV path (default: <place>_aqi.csv)")
    args = parser.parse_args()
    features, _ = build_place_features([args.place], args.start, args.end, require_target=False)
    out = args.out or re.sub(r"[^A-Za-z0-9]+", "_", args.place).strip("_") + "_aqi.csv"
    features.to_csv(out, index=False)
    print(f"Saved {len(features)} rows to {out}")
    print(features[["station", "date", "aqi", "aqi_category"]].tail())


if __name__ == "__main__":
    main()
