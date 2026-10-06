"""Interactive demo: AQI lookup for any place in the world + next-day forecast backtest."""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import AQI_BANDS, DEFAULT_HISTORY_START, METADATA_PATH, MODEL_PATH, PROCESSED_PATH  # noqa: E402
from src.forecast_year import forecast_series  
from src.world_aqi import build_place_features, categorize, fetch_place_history  

st.set_page_config(page_title="World AQI", page_icon="🌫️")
st.title("World AQI lookup and next-day forecast")
st.caption("Research demonstration only. It is not a health-alert or regulatory system.")

MODEL_READY = MODEL_PATH.exists() and METADATA_PATH.exists() and PROCESSED_PATH.exists()


@st.cache_resource
def load_model_files():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return metadata, joblib.load(MODEL_PATH), pd.read_csv(PROCESSED_PATH, parse_dates=["date"])


@st.cache_data(ttl=3600, show_spinner="Fetching air-quality data...")
def fetch_world(place: str, start: date, end: date):
    return build_place_features([place], start, end, require_target=False)


@st.cache_data
def load_history() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_PATH, usecols=["station", "date", "aqi"], parse_dates=["date"])


@st.cache_data(ttl=3600, show_spinner="Building the 1-year outlook...")
def outlook_for_dataset_station(station: str, start: date):
    history = load_history()
    return forecast_series(history[history["station"].eq(station)], start, label=station)


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching years of history (this can take a minute)...")
def outlook_for_place(place: str, start: date):
    history = fetch_place_history(place, DEFAULT_HISTORY_START, date.today() - timedelta(days=1))
    return forecast_series(history, start, label=str(history["station"].iloc[0]))


def show_outlook(result: dict) -> None:
    daily, monthly, meta = result["daily"], result["monthly"], result["metadata"]
    worst = monthly.loc[monthly["mean_aqi"].idxmax()]
    best = monthly.loc[monthly["mean_aqi"].idxmin()]
    st.markdown(f"**{result['station']}**: {daily['date'].min():%Y-%m-%d} to {daily['date'].max():%Y-%m-%d}")
    c1, c2 = st.columns(2)
    c1.metric("Most polluted month", worst["month"])
    c1.caption(f"Expected AQI about {worst['mean_aqi']:.0f} ({worst['aqi_category']})")
    c2.metric("Cleanest month", best["month"])
    c2.caption(f"Expected AQI about {best['mean_aqi']:.0f} ({best['aqi_category']})")
    st.caption("Line = expected AQI; outer lines = range holding about 80% of days (from the backtest).")
    st.line_chart(daily.set_index("date")[["aqi_lower", "aqi_forecast", "aqi_upper"]])
    st.dataframe(
        monthly[["month", "days", "mean_aqi", "aqi_category", "typical_day_low", "typical_day_high",
                 "pct_poor_or_worse", "history_mean_same_month"]]
    )
    with st.expander("How reliable is this?"):
        st.write(
            f"History used: {meta['history_start']} to {meta['history_end']} ({meta['observed_days']} days). "
            f"Backtest on the last {meta['backtest_holdout_days']} days; selected model: **{meta['chosen_model']}**."
        )
        if meta["backtest"]:
            st.dataframe(pd.DataFrame(meta["backtest"]).T.rename_axis("model"))
            st.caption("Error in AQI points on days the model had not seen. Lower is better; "
                       "'constant_mean' is the plain average, the bar a seasonal model must beat.")
        for note in meta["notes"]:
            st.warning(note)
    d1, d2 = st.columns(2)
    d1.download_button("Download daily CSV", daily.to_csv(index=False).encode("utf-8"),
                       file_name="aqi_outlook_daily.csv", mime="text/csv")
    d2.download_button("Download monthly CSV", monthly.to_csv(index=False).encode("utf-8"),
                       file_name="aqi_outlook_monthly.csv", mime="text/csv")


def predict(model, feature_names: list[str], rows: pd.DataFrame) -> float:
    """Missing numeric inputs are left as NaN; the model's own imputer fills them."""
    return float(np.clip(model.predict(rows.reindex(columns=feature_names))[0], 0, 500))


with st.expander("AQI levels"):
    low = 0
    for upper, label in AQI_BANDS:
        st.write(f"**{label}**: {low}-{upper}")
        low = upper + 1

tab_world, tab_year, tab_backtest = st.tabs(["World lookup (live)", "1-year outlook", "Backtest (dataset)"])

with tab_world:
    st.subheader("AQI and air quality of any place")
    place = st.text_input("Place (add a country for precision, e.g. 'London, UK')", value="Delhi")
    today = date.today()
    picked = st.date_input(
        "Date range", value=(today - timedelta(days=30), today), max_value=today, key="world_dates"
    )
    st.caption(
        "Values come from the Open-Meteo air-quality model (not a ground station), converted to the "
        "0-500 scale above. Very old dates may not be available."
    )
    if st.button("Get AQI", type="primary") and place.strip():
        if not (isinstance(picked, tuple) and len(picked) == 2):
            st.warning("Pick both a start and an end date.")
        else:
            try:
                world, _ = fetch_world(place.strip(), picked[0], picked[1])
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not get data for {place!r}: {exc}")
            else:
                if world.empty:
                    st.warning("No usable days returned. Try a longer or more recent range.")
                else:
                    latest = world.iloc[-1]
                    st.markdown(f"**{latest['station']}**")
                    c1, c2 = st.columns(2)
                    c1.metric(f"AQI on {latest['date']:%Y-%m-%d}", f"{latest['aqi']:.0f}")
                    c2.metric("Air quality", str(latest["aqi_category"]))
                    if MODEL_READY:
                        metadata, model, _ = load_model_files()
                        forecast = predict(model, metadata["features"], world.iloc[[-1]])
                        st.metric("Model's predicted next-day AQI", f"{forecast:.0f}")
                        st.caption(
                            f"Predicted category: {categorize(forecast)}. The model was trained on: "
                            f"{', '.join(metadata.get('stations_in_training', ['Kolkata stations']))[:200]}. "
                            "Accuracy for other places is untested until you retrain on them."
                        )
                    st.line_chart(world.set_index("date")["aqi"])
                    st.dataframe(
                        world[["date", "aqi", "aqi_category", "pm25_ug_m3", "pm10_ug_m3", "no2_ug_m3",
                               "so2_ug_m3", "co_mg_m3", "o3_ug_m3", "nh3_ug_m3"]],
                    )
                    st.download_button(
                        "Download CSV", world.to_csv(index=False).encode("utf-8"),
                        file_name="world_aqi.csv", mime="text/csv",
                    )


with tab_year:
    st.subheader("Expected AQI for the next 12 months")
    st.caption(
        "A seasonal outlook learned from past years, not a day-by-day forecast: future weather and "
        "pollution are unknown, so it shows the expected level for the time of year with a range. "
        "Events, policy changes and long-term trends are not captured."
    )
    outlook_start = date.today() + timedelta(days=1)
    source = st.radio("Source", ["Station/place in the dataset", "Any place (live)"], horizontal=True, key="year_source")
    if source == "Station/place in the dataset":
        if not PROCESSED_PATH.exists():
            st.info("Dataset not found. Run `python -m src.prepare_data` first.")
        else:
            options = sorted(load_history()["station"].dropna().unique())
            choice = st.selectbox("Station / place", options, key="year_station")
            try:
                show_outlook(outlook_for_dataset_station(choice, outlook_start))
            except ValueError as exc:
                st.warning(str(exc))
    else:
        year_place = st.text_input("Place (add a country for precision)", value="Delhi, India", key="year_place")
        st.caption(f"Fetches daily history from {DEFAULT_HISTORY_START} to yesterday. Needs internet and can take a minute.")
        if st.button("Build 1-year outlook", type="primary") and year_place.strip():
            try:
                show_outlook(outlook_for_place(year_place.strip(), outlook_start))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not build the outlook for {year_place!r}: {exc}")


with tab_backtest:
    if not MODEL_READY:
        st.info("Model files are not available yet. Prepare data and train the model first; see README.md.")
    else:
        metadata, model, data = load_model_files()
        numeric_features = [f for f in metadata["features"] if f != "station"]

        stations = sorted(data["station"].dropna().unique())
        station = st.selectbox("Monitoring station / place", stations)
        station_data = data.loc[data["station"].eq(station)].sort_values("date").reset_index(drop=True)

        st.subheader("Backtest: see what the model would have predicted")
        st.caption(
            f"Data covers {data['date'].min():%Y-%m-%d} to {data['date'].max():%Y-%m-%d}. "
            "Pick any date; the forecast is for the day after it."
        )
        dates = station_data["date"].dt.date.tolist()
        selected_date = st.select_slider("Choose a date", options=dates, value=dates[-1])
        row = station_data.loc[station_data["date"].dt.date.eq(selected_date)].iloc[0]

        inputs = {"station": station}
        for feature in numeric_features:
            inputs[feature] = float(row[feature]) if pd.notna(row[feature]) else float(data[feature].median())

        prediction = float(np.clip(model.predict(pd.DataFrame([inputs]))[0], 0, 500))

        col1, col2 = st.columns(2)
        with col1:
            shown_aqi = f"{row['aqi']:.0f}" if pd.notna(row["aqi"]) else "N/A"
            st.metric(f"AQI on {selected_date}", shown_aqi)
            if pd.notna(row["aqi"]):
                st.caption(f"Category: {categorize(row['aqi'])}")
        with col2:
            st.metric("Model's predicted next-day AQI", f"{prediction:.0f}")
            st.caption(f"Predicted category: {categorize(prediction)}")

        if pd.notna(row.get("aqi_next_day")):
            actual_next = row["aqi_next_day"]
            error = abs(prediction - actual_next)
            st.write(f"**Actual recorded AQI the next day: {actual_next:.0f}**  \nModel error: {error:.1f} AQI points")
        else:
            st.write("Actual next-day AQI is not available for this date (end of the station's record).")

        with st.expander("Adjust inputs manually (what-if analysis)"):
            st.caption("Try changing pollutant or weather values to see how the forecast responds.")
            manual_inputs = {"station": station}
            for feature in numeric_features:
                manual_inputs[feature] = st.number_input(
                    feature.replace("_", " ").title(), value=inputs[feature], key=feature
                )
            if st.button("Recalculate forecast with manual inputs", type="primary"):
                manual_pred = float(np.clip(model.predict(pd.DataFrame([manual_inputs]))[0], 0, 500))
                st.metric("Adjusted predicted AQI", f"{manual_pred:.0f}")
                st.success(f"Predicted category: {categorize(manual_pred)}")

st.caption(
    "This is a decision-support forecast, not a substitute for physical monitoring."
)
