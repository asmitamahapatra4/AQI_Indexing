"""One-year-ahead AQI outlook (seasonal model).

Why this is separate from the next-day model: the next-day model needs yesterday's AQI, pollutants and
weather. None of those exist for dates months ahead, so a year-ahead forecast can only describe the
EXPECTED SEASONAL LEVEL (e.g. "December is usually Poor"), not the AQI of a specific future day.

For every station/place the module:
  1. learns the yearly pattern from the AQI history (calendar features only),
  2. backtests several simple seasonal models on the most recent part of the history and keeps the
     one with the lowest error (baselines such as "always the average" compete too),
  3. forecasts the next 365 days with an uncertainty range,
  4. summarises each month: mean AQI, level, and the share of days expected in every level.

Command line (run from the project root):
    python -m src.forecast_year                          # every station in data/processed/model_dataset.csv
    python -m src.forecast_year --station Ballygunge     # stations whose name contains this text
    python -m src.forecast_year --place "Tokyo, Japan"   # any place in the world (live data)
    python -m src.forecast_year --start 2026-11-01 --horizon-days 365
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.config import (
    AQI_BANDS,
    DEFAULT_HISTORY_START,
    FORECAST_DIR,
    FORECAST_HORIZON_DAYS,
    MIN_HISTORY_DAYS,
    PROCESSED_PATH,
)
from src.world_aqi import categorize

CANDIDATES = ("constant_mean", "monthly_climatology", "harmonic_k2", "harmonic_k3", "harmonic_k3_trend")
FALLBACK_MODEL = "harmonic_k3"      # used when there is too little history to hold anything out
MIN_TRAIN_ROWS, MIN_TEST_ROWS = 120, 30
LEVEL_EDGES = [upper for upper, _ in AQI_BANDS[:-1]]            # 50, 100, 200, 300, 400
LEVEL_KEYS = [label.lower().replace(" ", "_") for _, label in AQI_BANDS]


def calendar_features(dates: pd.DatetimeIndex, origin: pd.Timestamp, harmonics: int, trend: bool) -> pd.DataFrame:
    angle = 2 * np.pi * dates.dayofyear.to_numpy() / 365.25
    columns: dict[str, np.ndarray] = {}
    for k in range(1, harmonics + 1):
        columns[f"sin{k}"] = np.sin(k * angle)
        columns[f"cos{k}"] = np.cos(k * angle)
    if trend:
        columns["trend_years"] = (dates - origin).days.to_numpy() / 365.25
    return pd.DataFrame(columns, index=dates)


class SeasonalModel:
    """AQI as a function of the calendar only (day of year, optionally a linear trend)."""

    def __init__(self, kind: str, clip: tuple[float | None, float | None] | None = (0, 500)):
        self.kind = kind
        self.clip = clip  # (lower, upper); None on a side = unbounded. Default suits AQI.

    def fit(self, dates: pd.DatetimeIndex, y: np.ndarray) -> "SeasonalModel":
        self.origin = dates.min()
        self.mean_ = float(np.mean(y))
        if self.kind == "monthly_climatology":
            by_month = pd.Series(y, index=dates.month).groupby(level=0).mean()
            self.table_ = by_month.reindex(range(1, 13)).fillna(self.mean_)
        elif self.kind != "constant_mean":
            self.harmonics = int(re.search(r"_k(\d+)", self.kind).group(1))
            self.trend = self.kind.endswith("_trend")
            features = calendar_features(dates, self.origin, self.harmonics, self.trend)
            self.regressor_ = Ridge(alpha=1.0).fit(features, y)
        return self

    def predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        if self.kind == "constant_mean":
            values = np.full(len(dates), self.mean_)
        elif self.kind == "monthly_climatology":
            values = self.table_.loc[dates.month].to_numpy()
        else:
            values = self.regressor_.predict(calendar_features(dates, self.origin, self.harmonics, self.trend))
        if self.clip and any(bound is not None for bound in self.clip):
            values = np.clip(values, *self.clip)
        return values


def _clean_history(history: pd.DataFrame) -> pd.DataFrame:
    df = history[["date", "aqi"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.dropna().groupby("date", as_index=False)["aqi"].mean().sort_values("date")
    return df.reset_index(drop=True)


def forecast_series(history: pd.DataFrame, start, horizon_days: int = FORECAST_HORIZON_DAYS, label: str = "") -> dict:
    """Outlook for one station/place. `history` needs `date` and `aqi` columns (gaps are fine)."""
    df = _clean_history(history)
    if df.empty:
        raise ValueError("No AQI values to learn from.")
    span_days = int((df["date"].max() - df["date"].min()).days) + 1
    if span_days < MIN_HISTORY_DAYS:
        raise ValueError(f"Need at least {MIN_HISTORY_DAYS} days of history to learn a seasonal pattern; found {span_days}.")

 
    holdout_days = min(FORECAST_HORIZON_DAYS, span_days // 3)
    cutoff = df["date"].max() - pd.Timedelta(days=holdout_days)
    train, test = df[df["date"] <= cutoff], df[df["date"] > cutoff]
    backtest: dict[str, dict[str, float]] = {}
    holdout_errors: dict[str, np.ndarray] = {}
    if len(train) >= MIN_TRAIN_ROWS and len(test) >= MIN_TEST_ROWS:
        for kind in CANDIDATES:
            model = SeasonalModel(kind).fit(pd.DatetimeIndex(train["date"]), train["aqi"].to_numpy())
            error = test["aqi"].to_numpy() - model.predict(pd.DatetimeIndex(test["date"]))
            holdout_errors[kind] = error
            backtest[kind] = {
                "mae": round(float(np.mean(np.abs(error))), 2),
                "rmse": round(float(np.sqrt(np.mean(error ** 2))), 2),
            }
        chosen = min(backtest, key=lambda kind: backtest[kind]["mae"])
        residuals = holdout_errors[chosen]
    else:
        chosen, holdout_days, residuals = FALLBACK_MODEL, 0, None


    all_dates, all_aqi = pd.DatetimeIndex(df["date"]), df["aqi"].to_numpy()
    model = SeasonalModel(chosen).fit(all_dates, all_aqi)
    if residuals is None:
        residuals = all_aqi - model.predict(all_dates)  # in-sample: optimistic
    if len(residuals) > 2000:
        residuals = np.random.default_rng(0).choice(residuals, 2000, replace=False)

    dates = pd.date_range(pd.Timestamp(start).normalize(), periods=horizon_days)
    point = model.predict(dates)
    q_low, q_high = np.quantile(residuals, [0.10, 0.90])
    daily = pd.DataFrame({
        "station": label,
        "date": dates,
        "aqi_forecast": np.round(point),
        "aqi_lower": np.clip(np.round(point + q_low), 0, 500),
        "aqi_upper": np.clip(np.round(point + q_high), 0, 500),
    })
    daily["aqi_category"] = daily["aqi_forecast"].map(categorize)

    
    simulated = np.clip(point[:, None] + residuals[None, :], 0, 500)
    level = np.digitize(simulated, LEVEL_EDGES, right=True)
    shares = np.stack([(level == i).mean(axis=1) for i in range(len(AQI_BANDS))], axis=1) * 100

    work = daily[["date", "aqi_lower", "aqi_upper"]].copy()
    work["month"] = work["date"].dt.to_period("M").astype(str)
    work["aqi_mean"] = point
    for i, key in enumerate(LEVEL_KEYS):
        work[f"pct_{key}"] = shares[:, i]
    monthly = work.drop(columns="date").groupby("month", as_index=False).agg(
        days=("aqi_mean", "size"), mean_aqi=("aqi_mean", "mean"),
        typical_day_low=("aqi_lower", "mean"), typical_day_high=("aqi_upper", "mean"),
        **{f"pct_{key}": (f"pct_{key}", "mean") for key in LEVEL_KEYS},
    )
    monthly["aqi_category"] = monthly["mean_aqi"].map(categorize)
    monthly["pct_poor_or_worse"] = monthly[[f"pct_{k}" for k in LEVEL_KEYS[3:]]].sum(axis=1)
    history_by_month = df.groupby(df["date"].dt.month)["aqi"].mean()
    monthly["history_mean_same_month"] = monthly["month"].str[5:7].astype(int).map(history_by_month)
    monthly.insert(0, "station", label)
    rounding = {c: 1 for c in monthly.columns if c.startswith(("mean_", "typical_", "pct_", "history_"))}
    monthly = monthly.round(rounding)

    notes = []
    if span_days < 2 * 365:
        notes.append("Less than two years of history: the yearly pattern is based on a single repeat and is unverified.")
    if holdout_days == 0:
        notes.append("Too little data to backtest; error ranges are in-sample and optimistic.")
    if chosen == "constant_mean":
        notes.append("No seasonal model beat the plain average in the backtest: treat month-to-month differences with caution.")
    if chosen.endswith("_trend"):
        notes.append("The selected model extrapolates a linear trend; trends from short histories can mislead.")
    metadata = {
        "station": label,
        "history_start": str(df["date"].min().date()),
        "history_end": str(df["date"].max().date()),
        "history_span_days": span_days,
        "observed_days": int(len(df)),
        "backtest_holdout_days": int(holdout_days),
        "chosen_model": chosen,
        "backtest_note": "Model chosen on this same holdout, so its error is slightly optimistic.",
        "backtest": backtest,
        "interval": "80% range (10th-90th percentile of backtest errors)",
        "residual_p10": round(float(q_low), 1),
        "residual_p90": round(float(q_high), 1),
        "notes": notes,
    }
    return {"station": label, "daily": daily, "monthly": monthly, "metadata": metadata}


def forecast_dataset(
    data: pd.DataFrame, start, horizon_days: int = FORECAST_HORIZON_DAYS, station_contains: str | None = None
) -> tuple[list[dict], dict[str, str]]:
    """Run the outlook for every station in a `station, date, aqi` table. Returns (results, skipped)."""
    results, skipped = [], {}
    for station, group in data.groupby("station"):
        if station_contains and station_contains.lower() not in str(station).lower():
            continue
        try:
            results.append(forecast_series(group, start, horizon_days, label=str(station)))
        except ValueError as exc:
            skipped[str(station)] = str(exc)
    return results, skipped


def save_results(results: list[dict], out_dir: Path = FORECAST_DIR, prefix: str = "year_forecast") -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{prefix}_daily.csv", out_dir / f"{prefix}_monthly.csv", out_dir / f"{prefix}_metadata.json"]
    pd.concat([r["daily"] for r in results], ignore_index=True).to_csv(paths[0], index=False)
    pd.concat([r["monthly"] for r in results], ignore_index=True).to_csv(paths[1], index=False)
    paths[2].write_text(json.dumps([r["metadata"] for r in results], indent=2), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="One-year AQI outlook for stations in the dataset or any place.")
    parser.add_argument("--place", help='any place in the world, e.g. "Tokyo, Japan" (fetches history live)')
    parser.add_argument("--station", help="only stations whose name contains this text (dataset mode)")
    parser.add_argument("--start", default=(date.today() + timedelta(days=1)).isoformat(),
                        help="first forecast day (default: tomorrow)")
    parser.add_argument("--horizon-days", type=int, default=FORECAST_HORIZON_DAYS)
    parser.add_argument("--history-start", default=DEFAULT_HISTORY_START, help="place mode: first history day")
    args = parser.parse_args()

    if args.place:
        from src.world_aqi import fetch_place_history

        history = fetch_place_history(args.place, args.history_start, date.today() - timedelta(days=1))
        results, skipped = forecast_dataset(history, args.start, args.horizon_days)
        prefix = re.sub(r"[^A-Za-z0-9]+", "_", args.place).strip("_").lower()
    else:
        if not PROCESSED_PATH.exists():
            raise FileNotFoundError("Run `python -m src.prepare_data` first (or use --place).")
        data = pd.read_csv(PROCESSED_PATH, usecols=["station", "date", "aqi"], parse_dates=["date"])
        results, skipped = forecast_dataset(data, args.start, args.horizon_days, args.station)
        prefix = "year_forecast"
    for station, reason in skipped.items():
        print(f"SKIPPED {station}: {reason}")
    if not results:
        raise SystemExit("Nothing to forecast.")
    for r in results:
        m, meta = r["monthly"], r["metadata"]
        worst, best = m.loc[m["mean_aqi"].idxmax()], m.loc[m["mean_aqi"].idxmin()]
        best_mae = meta["backtest"].get(meta["chosen_model"], {}).get("mae", "n/a")
        base_mae = meta["backtest"].get("constant_mean", {}).get("mae", "n/a")
        print(f"\n{r['station']}: model={meta['chosen_model']} (backtest MAE {best_mae} vs plain average {base_mae})")
        print(f"  most polluted month: {worst['month']} (~{worst['mean_aqi']:.0f}, {worst['aqi_category']})")
        print(f"  cleanest month:      {best['month']} (~{best['mean_aqi']:.0f}, {best['aqi_category']})")
        for note in meta["notes"]:
            print(f"  note: {note}")
    for path in save_results(results, prefix=prefix):
        print("Saved", path)


if __name__ == "__main__":
    main()
