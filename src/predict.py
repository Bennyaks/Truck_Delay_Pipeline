"""
predict.py
----------
Loads a trained model and generates delay predictions for NEW trips
(trips that haven't happened yet, so we don't know their actual
outcome). Reuses the EXACT same feature-building functions from
merge_data.py and feature_engineering.py that training used, so
predictions go through an identical pipeline -- any drift between
training features and prediction features would silently break
accuracy, so nothing here is reimplemented separately.

IMPORTANT CAVEAT: traffic and route-weather features are built by
averaging historical readings that fall inside each trip's own time
window. A genuinely FUTURE trip has no such readings yet, so the
pipeline automatically falls back to that route's overall historical
average (the same fallback already built into aggregate_traffic() /
aggregate_route_weather() in merge_data.py) -- a reasonable proxy for
"typical conditions," but not a live forecast. City weather uses the
nearest available reading via merge_asof, so a far-future trip will
just reuse the most recent known reading for that city.
"""

import os
import joblib
import pandas as pd

from config import BASE_DIR
from load_data import load_all
from clean_data import clean_all
from merge_data import (
    add_route_info,
    aggregate_traffic,
    aggregate_route_weather,
    add_nearest_city_weather,
    add_driver_and_truck_info,
)
from feature_engineering import engineer_features, prepare_model_matrix

MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_trained_artifacts(model_name: str = "random_forest"):
    """model_name: 'random_forest' (recommended, better recall/AUC) or 'logistic_regression'."""
    model = joblib.load(os.path.join(MODELS_DIR, f"{model_name}.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    return model, scaler


def load_reference_tables() -> dict:
    """Loads + cleans every table EXCEPT truck_schedule -- these are the
    static/historical tables new trips get joined against."""
    tables = load_all()
    return clean_all(tables)


def build_features_for_new_trips(new_trips: pd.DataFrame, tables: dict) -> pd.DataFrame:
    """
    new_trips must have columns: truck_id, route_id, departure_date,
    estimated_arrival -- same shape as a truck_schedule row, just
    without a 'delay' column since that's what we're predicting.
    """
    new_trips = new_trips.copy()
    new_trips["departure_date"] = pd.to_datetime(new_trips["departure_date"])
    new_trips["estimated_arrival"] = pd.to_datetime(new_trips["estimated_arrival"])

    df = add_route_info(new_trips, tables["routes"])
    df = aggregate_traffic(df, tables["traffic"])
    df = aggregate_route_weather(df, tables["routes_weather"])
    df = add_nearest_city_weather(df, tables["city_weather"])
    df = add_driver_and_truck_info(df, tables["drivers"], tables["trucks"])

    return engineer_features(df)


def align_to_training_columns(model_input: pd.DataFrame, expected_columns) -> pd.DataFrame:
    """
    One-hot encoding a small new-trips batch can produce a different
    column set than training had (e.g. if every truck in this batch
    happens to be diesel, 'fuel_type_gas' simply won't exist here).
    Reindexing against the columns the scaler was FIT on guarantees an
    exact match: missing columns are added as 0, unexpected ones dropped.
    """
    return model_input.reindex(columns=expected_columns, fill_value=0)


def predict_delay(new_trips: pd.DataFrame, model_name: str = "random_forest") -> pd.DataFrame:
    """Main entry point: takes upcoming trips, returns predicted label + probability."""
    model, scaler = load_trained_artifacts(model_name)
    tables = load_reference_tables()

    features_df = build_features_for_new_trips(new_trips, tables)
    model_input = prepare_model_matrix(features_df)

    # scaler.feature_names_in_ is set automatically by sklearn because
    # it was fit on a DataFrame during training -- using it here is what
    # guarantees prediction-time columns exactly match training-time columns.
    model_input = align_to_training_columns(model_input, scaler.feature_names_in_)

    scaled_input = pd.DataFrame(
        scaler.transform(model_input), columns=model_input.columns, index=model_input.index
    )

    predictions = model.predict(scaled_input)
    probabilities = model.predict_proba(scaled_input)[:, 1]

    results = new_trips[["truck_id", "route_id", "departure_date", "estimated_arrival"]].copy()
    results["predicted_delay"] = predictions
    results["delay_probability"] = probabilities.round(3)
    results["predicted_delay_label"] = results["predicted_delay"].map({0: "On Time", 1: "Delayed"})
    return results


if __name__ == "__main__":
    # Demo: predict on the 5 most recent real trips from truck_schedule,
    # then compare against their ACTUAL outcome as a sanity check that
    # the pipeline produces sensible results end-to-end.
    schedule = load_all()["truck_schedule"]
    recent = schedule.sort_values("departure_date").tail(5)

    sample_input = recent[["truck_id", "route_id", "departure_date", "estimated_arrival"]]
    predictions = predict_delay(sample_input)

    actual = recent[["truck_id", "route_id", "departure_date", "delay"]].copy()
    actual["actual_label"] = actual["delay"].map({0: "On Time", 1: "Delayed"})

    check = predictions.merge(actual, on=["truck_id", "route_id", "departure_date"])
    print(check[["truck_id", "route_id", "predicted_delay_label", "actual_label", "delay_probability"]])