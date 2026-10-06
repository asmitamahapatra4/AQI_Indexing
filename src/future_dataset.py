"""Future rows in the SAME columns as data/processed/model_dataset.csv.

For every station, the next `--horizon-days` days are filled with expected (seasonal) values:
  * aqi                         the 1-year outlook model of src/forecast_year.py (backtested)
  * pollutants and weather      the expected value for that time of year (seasonal curve per column)
  * aqi_lag_*, rolling means,
    aqi_next_day, calendar      derived from the forecast AQI exactly as src/data_pipeline.py does
  * daily_aqi_definition        set to "seasonal_forecast" so forecast rows can never be mistaken for data

These rows are expectations, not measurements: smooth, with no random day-to-day swings. Never use them
as training data; the models would just learn from other models' output.

Command line (run from the project root):
    python -m src.future_dataset                                   # next 365 days, every station
    python -m src.future_dataset --station Ballygunge --horizon-days 180
    python -m src.future_dataset --start 2026-11-01 --with-category --with-range --combine
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FORECAST_DIR, FORECAST_HORIZON_DAYS, LAGS, PROCESSED_PATH, ROLLING_WINDOWS
from src.forecast_year import SeasonalModel, forecast_series
from src.world_aqi import categorize

FORECAST_LABEL = "seasonal_forecast"
MIN_COLUMN_OBSERVATIONS = 60

EXPECTED_COLUMNS = {
    "pm25_ug_m3": (0, None, 2), "pm10_ug_m3": (0, None, 2), "no2_ug_m3": (0, None, 2),
    "so2_ug_m3": (0, None, 2), "co_mg_m3": (0, None, 2), "o3_ug_m3": (0, None, 2),
    "nh3_ug_m3": (0, None, 2),
    "temperature_2m_mean": (None, None, 1), "relative_humidity_2m_mean": (0, 100, 0),
    "wind_speed_10m_max": (0, None, 1), "pressure_msl_mean": (None, None, 1),
}
CARRY_FORWARD = ("latitude", "longitude", "source")  # constant per station in the full dataset


def build_future_rows(station_history: pd.DataFrame, start, horizon_days: int, label: str) -> pd.DataFrame:
    """Future rows for one station. Raises ValueError if the history is too short to learn from."""
    hist = station_history.sort_values("date").reset_index(drop=True)
    last_day = hist["date"].max()
    first_forecast_day = last_day + pd.Timedelta(days=1)
    start = max(pd.Timestamp(start).normalize(), first_forecast_day)
    end = start + pd.Timedelta(days=horizon_days - 1)

    
    n_days = (end + pd.Timedelta(days=1) - first_forecast_day).days + 1
    outlook = forecast_series(hist, first_forecast_day, n_days, label=label)["daily"]
    frame = pd.DataFrame({"date": outlook["date"], "aqi": outlook["aqi_forecast"]})
    frame["aqi_lower"], frame["aqi_upper"] = outlook["aqi_lower"], outlook["aqi_upper"]

    future_index = pd.DatetimeIndex(frame["date"])
    for column, (lower, upper, decimals) in EXPECTED_COLUMNS.items():
        if column not in hist:
            continue
        observed = hist[["date", column]].dropna()
        if len(observed) < MIN_COLUMN_OBSERVATIONS:
            frame[column] = np.nan
            continue
        model = SeasonalModel("harmonic_k3", clip=(lower, upper)).fit(
            pd.DatetimeIndex(observed["date"]), observed[column].to_numpy()
        )
        frame[column] = np.round(model.predict(future_index), decimals)

    
    tail = hist.loc[hist["aqi"].notna(), ["date", "aqi"]].tail(max(max(LAGS), max(ROLLING_WINDOWS)) + 1)
    path = pd.concat([tail, frame[["date", "aqi"]]], ignore_index=True)
    for lag in LAGS:
        path[f"aqi_lag_{lag}"] = path["aqi"].shift(lag)
    for window in ROLLING_WINDOWS:
        path[f"aqi_rolling_mean_{window}"] = path["aqi"].shift(1).rolling(window).mean()
    path["aqi_next_day"] = path["aqi"].shift(-1)
    path = path.iloc[len(tail):].reset_index(drop=True)
    for column in path.columns.difference(["date", "aqi"]):
        frame[column] = path[column]

    frame["station"] = label
    frame["daily_aqi_definition"] = FORECAST_LABEL
    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["month"] = frame["date"].dt.month
    frame["day_of_year_sin"] = np.sin(2 * np.pi * frame["date"].dt.dayofyear / 365.25)
    frame["day_of_year_cos"] = np.cos(2 * np.pi * frame["date"].dt.dayofyear / 365.25)
    frame["aqi_category"] = frame["aqi"].map(categorize)
    for column in CARRY_FORWARD:
        if column in hist:
            frame[column] = hist[column].dropna().iloc[-1] if hist[column].notna().any() else np.nan
    if "relative_humidity_2m_mean" in frame and frame["relative_humidity_2m_mean"].notna().all():
        frame["relative_humidity_2m_mean"] = frame["relative_humidity_2m_mean"].astype(int)
    return frame[(frame["date"] >= start) & (frame["date"] <= end)].reset_index(drop=True)


def build_future_dataset(
    data: pd.DataFrame, start, horizon_days: int = FORECAST_HORIZON_DAYS, station_contains: str | None = None
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Future rows for every station, with the columns of `data` in their original order."""
    frames, skipped = [], {}
    for station, group in data.groupby("station"):
        if station_contains and station_contains.lower() not in str(station).lower():
            continue
        try:
            frames.append(build_future_rows(group, start, horizon_days, str(station)))
        except ValueError as exc:
            skipped[str(station)] = str(exc)
    if not frames:
        return pd.DataFrame(columns=data.columns), skipped
    future = pd.concat(frames, ignore_index=True)
    return future, skipped


def to_dataset_columns(future: pd.DataFrame, original_columns: list[str], with_category: bool, with_range: bool) -> pd.DataFrame:
    """Order the columns like the original CSV; optionally add category / range columns after `aqi`."""
    columns = list(original_columns)
    extras = (["aqi_category"] if with_category and "aqi_category" not in columns else []) + \
             (["aqi_lower", "aqi_upper"] if with_range else [])
    columns[columns.index("aqi") + 1:columns.index("aqi") + 1] = extras
    out = future.reindex(columns=columns).copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Future rows with the same columns as model_dataset.csv.")
    parser.add_argument("--input", type=Path, default=PROCESSED_PATH, help="the dataset CSV to continue")
    parser.add_argument("--out", type=Path, default=FORECAST_DIR / "future_dataset.csv")
    parser.add_argument("--station", help="only stations whose name contains this text")
    parser.add_argument("--start", default=(date.today() + timedelta(days=1)).isoformat(), help="first future day (default: tomorrow)")
    parser.add_argument("--horizon-days", type=int, default=FORECAST_HORIZON_DAYS)
    parser.add_argument("--with-category", action="store_true", help="add an aqi_category column (Good ... Severe)")
    parser.add_argument("--with-range", action="store_true", help="add aqi_lower / aqi_upper (80%% range) columns")
    parser.add_argument("--combine", action="store_true", help="also save history + future in one file")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} not found. Run `python -m src.prepare_data` first or pass --input.")
    data = pd.read_csv(args.input, parse_dates=["date"])
    future, skipped = build_future_dataset(data, args.start, args.horizon_days, args.station)
    for station, reason in skipped.items():
        print(f"SKIPPED {station}: {reason}")
    if future.empty:
        raise SystemExit("Nothing to forecast.")

    out = to_dataset_columns(future, list(data.columns), args.with_category, args.with_range)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Saved {len(out)} future rows ({out['station'].nunique()} stations, {out['date'].min()} to {out['date'].max()}) to {args.out}")
    if args.combine:
        history = to_dataset_columns(data, list(data.columns), args.with_category, args.with_range)
        if args.with_category and "aqi_category" in history:
            history["aqi_category"] = data["aqi"].map(categorize)
        combined = pd.concat([history, out], ignore_index=True).sort_values(["station", "date"])
        combined_path = args.out.with_name("model_dataset_with_future.csv")
        combined.to_csv(combined_path, index=False)
        print(f"Saved history + future ({len(combined)} rows) to {combined_path}")


if __name__ == "__main__":
    main()
