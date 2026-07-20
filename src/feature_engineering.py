"""
feature_engineering.py
-----------------------
Turns the merged trip-level dataset into model-ready features.

Two things found while profiling the merged data shaped the design here:
1. 'chanceofrain', 'chanceoffog', 'chanceofsnow', 'chanceofthunder' are
   ALWAYS 0 across the entire raw dataset (checked on city_weather AND
   routes_weather, 480K+ rows combined) -> zero signal, so they're
   dropped instead of turned into features.
2. EVERY trip departs at exactly 07:00 -> a day/night flag would be
   constant and useless, so that idea was dropped too. Day-of-week
   DOES vary, so that's kept as a feature instead.
"""

import pandas as pd
from merge_data import build_merged_dataset

# Columns confirmed to carry no signal (always 0) -- dropped up front.
ZERO_SIGNAL_COLUMNS = [
    "chanceofrain", "chanceoffog", "chanceofsnow", "chanceofthunder",
    "origin_chanceofrain", "origin_chanceoffog", "origin_chanceofsnow", "origin_chanceofthunder",
    "destination_chanceofrain", "destination_chanceoffog", "destination_chanceofsnow", "destination_chanceofthunder",
]

# Identifier / timestamp columns that are useful for joining but not for
# feeding directly into a model (kept in the "full" dataframe, dropped
# only in the final model matrix).
ID_COLUMNS = [
    "truck_id", "route_id", "driver_id", "name",
    "origin_id", "destination_id", "departure_date", "estimated_arrival",
]


def add_bad_weather_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary 'bad weather' indicators for route, origin, and destination.
    Thresholds picked from the actual distribution (see profiling):
      - precip > 0            -> roughly top 10-25% of trips (rain/snow present)
      - visibility < 3         -> half of the max observed visibility (6)
      - wind_speed > 15        -> ~95th percentile, i.e. genuinely windy
    Any one of these being true marks that leg of the trip as bad weather.
    """
    df = df.copy()

    def _flag(prefix: str) -> pd.Series:
        precip_col = f"{prefix}precip" if prefix else "precip"
        vis_col = f"{prefix}visibility" if prefix else "visibility"
        wind_col = f"{prefix}wind_speed" if prefix else "wind_speed"
        return (
            (df[precip_col] > 0) | (df[vis_col] < 3) | (df[wind_col] > 15)
        ).astype(int)

    df["route_bad_weather"] = _flag("")
    df["origin_bad_weather"] = _flag("origin_")
    df["destination_bad_weather"] = _flag("destination_")
    df["any_bad_weather"] = (
        (df["route_bad_weather"] == 1)
        | (df["origin_bad_weather"] == 1)
        | (df["destination_bad_weather"] == 1)
    ).astype(int)
    return df


def add_traffic_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    'high_traffic' = trip's average vehicle count is above the overall
    75th percentile (computed live from the data, not a hardcoded guess,
    so it stays correct if the dataset changes).
    """
    df = df.copy()
    threshold = df["no_of_vehicles"].quantile(0.75)
    df["high_traffic"] = (df["no_of_vehicles"] > threshold).astype(int)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Day-of-week is the only calendar signal worth extracting here --
    departure_date's HOUR is constant (always 07:00) across all 12,308
    trips, confirmed during profiling, so no time-of-day feature is added.
    """
    df = df.copy()
    df["departure_day_of_week"] = df["departure_date"].dt.dayofweek  # 0=Mon ... 6=Sun
    df["is_weekend_departure"] = df["departure_day_of_week"].isin([5, 6]).astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Runs all feature engineering steps and drops zero-signal columns."""
    df = df.drop(columns=ZERO_SIGNAL_COLUMNS)
    df = add_bad_weather_flags(df)
    df = add_traffic_flags(df)
    df = add_calendar_features(df)
    return df


def prepare_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces the final matrix ready for sklearn: drops IDs/timestamps
    and one-hot encodes the remaining categorical columns. Kept as a
    separate step from engineer_features() so the "full" dataframe
    (with IDs) is still available for inspection/debugging.
    """
    model_df = df.drop(columns=ID_COLUMNS)
    categorical_cols = ["gender", "driving_style", "fuel_type"]
    model_df = pd.get_dummies(model_df, columns=categorical_cols, drop_first=True)
    return model_df


def build_final_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline entry point: load -> clean -> merge -> engineer.
    Returns (full_df, model_df):
      full_df  -> engineered features + IDs, good for EDA/debugging
      model_df -> encoded, ID-free matrix, ready for train/test split
    """
    merged = build_merged_dataset()
    full_df = engineer_features(merged)
    model_df = prepare_model_matrix(full_df)
    return full_df, model_df


if __name__ == "__main__":
    full_df, model_df = build_final_dataset()
    print("full_df shape :", full_df.shape)
    print("model_df shape:", model_df.shape)
    print("\nmodel_df dtypes:\n", model_df.dtypes.value_counts())
    print("\nany nulls in model_df:", model_df.isnull().sum().sum())
    print("\nbad weather / high traffic rates:")
    print(full_df[["any_bad_weather", "high_traffic", "is_weekend_departure"]].mean())