"""Train and evaluate leakage-safe next-day AQI regression models."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import METADATA_PATH, MODEL_PATH, NON_FEATURE_COLUMNS, PROCESSED_PATH, TARGET_COLUMN


def regression_metrics(actual: pd.Series, predicted) -> dict[str, float]:
    predicted = pd.Series(predicted, index=actual.index)
    valid = actual.notna() & predicted.notna()
    actual, predicted = actual[valid], predicted[valid]
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 3),
        "rmse": round(float(mean_squared_error(actual, predicted) ** 0.5), 3),
        "r2": round(float(r2_score(actual, predicted)), 3),
        "n_rows": int(valid.sum()),
    }


def make_pipeline(model: object, numeric_features: list[str]) -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric_features),
            ("station", OneHotEncoder(handle_unknown="ignore"), ["station"]),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def main() -> None:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError("Run `python -m src.prepare_data` before training.")
    data = pd.read_csv(PROCESSED_PATH, parse_dates=["date"]).sort_values("date")
    # Identifiers, the target, text labels and coordinates are never model inputs.
    candidate_features = [column for column in data.columns if column not in NON_FEATURE_COLUMNS]
    numeric_features = [
        column for column in candidate_features if pd.api.types.is_numeric_dtype(data[column])
    ]
    features = ["station", *numeric_features]
    if len(data) < 120:
        raise ValueError("Need at least 120 prepared rows to reserve an honest test set.")

    # The final 20% is never used for model selection or hyperparameter choices.
    split_index = int(len(data) * 0.80)
    train, test = data.iloc[:split_index], data.iloc[split_index:]
    X_train, y_train = train[features], train[TARGET_COLUMN]
    X_test, y_test = test[features], test[TARGET_COLUMN]

    candidates = {
        "linear_regression": make_pipeline(LinearRegression(), numeric_features),
        "random_forest": make_pipeline(
            RandomForestRegressor(
                n_estimators=400, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            numeric_features,
        ),
    }
    results: dict[str, dict[str, float]] = {}
    baseline = test["aqi"]
    results["persistence_baseline"] = regression_metrics(y_test, baseline)

    tscv = TimeSeriesSplit(n_splits=4)
    fitted_models: dict[str, Pipeline] = {}
    for name, pipeline in candidates.items():
        cv_scores = -cross_val_score(pipeline, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error")
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
        results[name] = regression_metrics(y_test, pipeline.predict(X_test))
        results[name]["cv_mae_mean"] = round(float(cv_scores.mean()), 3)
        results[name]["cv_mae_std"] = round(float(cv_scores.std()), 3)

    baseline_cv_scores = []
    for _, val_idx in tscv.split(X_train):
        val = train.iloc[val_idx]
        actual, pred = val[TARGET_COLUMN], val["aqi"]
        valid = actual.notna() & pred.notna()
        baseline_cv_scores.append(mean_absolute_error(actual[valid], pred[valid]))
    results["persistence_baseline"]["cv_mae_mean"] = round(float(np.mean(baseline_cv_scores)), 3)
    results["persistence_baseline"]["cv_mae_std"] = round(float(np.std(baseline_cv_scores)), 3)

    best_name = min(fitted_models, key=lambda name: results[name]["mae"])
    best_model = fitted_models[best_name]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    metadata = {
        "target": TARGET_COLUMN,
        "forecast_horizon": "next day (t+1)",
        "features": features,
        "best_model": best_name,
        "train_rows": len(train),
        "test_rows": len(test),
        "test_start": str(test["date"].min().date()),
        "test_end": str(test["date"].max().date()),
        "stations_in_training": sorted(data["station"].unique().tolist()),
        "results": results,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
