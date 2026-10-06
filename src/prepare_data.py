"""Audit raw source files and build the model-ready dataset.

Original behaviour (Kolkata station files in data/raw):
    python -m src.prepare_data

Add places from anywhere in the world (Open-Meteo data, see src/world_aqi.py):
    python -m src.prepare_data --places "Delhi" "London, UK" "Tokyo, Japan"
    python -m src.prepare_data --places-file places.txt --start 2024-01-01 --end 2026-08-31

World places only (no Kolkata raw files needed):
    python -m src.prepare_data --places-file places.txt --world-only
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.config import (
    AUDIT_PATH,
    DEFAULT_HISTORY_START,
    KOLKATA_LATITUDE,
    KOLKATA_LONGITUDE,
    PROCESSED_PATH,
    RAW_DATA_DIR,
    SOURCE_STATION,
)
from src.data_pipeline import (
    aggregate_to_daily,
    build_features,
    create_data_quality_report,
    fetch_daily_weather,
    read_raw_files,
    standardise_air_data,
)
from src.world_aqi import build_place_features, categorize


def read_places(places: list[str], places_file: Path | None) -> list[str]:
    """Places from --places plus one-per-line from --places-file ('#' starts a comment)."""
    result = list(places)
    if places_file:
        for line in Path(places_file).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                result.append(line)
    return list(dict.fromkeys(result))  # drop duplicates, keep order


def build_station_dataset() -> tuple[pd.DataFrame, dict]:
    """The original Kolkata pipeline, unchanged apart from the extra provenance columns."""
    raw = read_raw_files(RAW_DATA_DIR)
    air = standardise_air_data(raw)
    audit = create_data_quality_report(air)
    daily = aggregate_to_daily(air)
    weather = fetch_daily_weather(daily["date"].min(), daily["date"].max())
    data = build_features(daily, weather)
    data["source"] = SOURCE_STATION
    data["latitude"] = KOLKATA_LATITUDE
    data["longitude"] = KOLKATA_LONGITUDE
    return data, audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the model-ready AQI dataset.")
    parser.add_argument("--places", nargs="*", default=[], help='world places, e.g. "Delhi" "London, UK"')
    parser.add_argument("--places-file", type=Path, help="text file with one place per line")
    parser.add_argument("--start", default=DEFAULT_HISTORY_START, help="world data start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=(date.today() - timedelta(days=1)).isoformat(),
                        help="world data end date (default: yesterday)")
    parser.add_argument("--world-only", action="store_true", help="skip the Kolkata raw station files")
    args = parser.parse_args()

    places = read_places(args.places, args.places_file)
    if args.world_only and not places:
        parser.error("--world-only needs --places or --places-file")

    frames: list[pd.DataFrame] = []
    audit: dict[str, object] = {}
    if not args.world_only:
        stations, audit = build_station_dataset()
        frames.append(stations)
    if places:
        world, failures = build_place_features(places, args.start, args.end)
        frames.append(world)
        audit["world_places"] = {
            "requested": places,
            "fetched": sorted(world["station"].unique().tolist()),
            "failed": failures,
            "start": args.start,
            "end": args.end,
        }
        for place, reason in failures.items():
            print(f"WARNING: could not fetch {place!r}: {reason}")

    data = pd.concat(frames, ignore_index=True, sort=False)
    data["aqi_category"] = data["aqi"].map(categorize)
    data = data.sort_values(["station", "date"]).reset_index(drop=True)

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_PATH, index=False)

    missing_weather = data.filter(regex="temperature|humidity|wind|pressure").isna().mean().mean() * 100
    print(f"Saved data-quality report to {AUDIT_PATH}")
    print(f"Saved {len(data)} model rows ({data['station'].nunique()} stations/places) to {PROCESSED_PATH}")
    print(f"Average weather-feature missingness: {missing_weather:.2f}%")


if __name__ == "__main__":
    main()
