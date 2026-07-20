"""
clean_data.py
-------------
Handles the null-value cleanup found when profiling the raw tables.
(Full-row duplicate check was also run on every table -> zero duplicates
found anywhere, so there's no dedup logic needed here.)

Null summary from profiling (see EDA step):
    drivers:  gender (23), driving_style (52)
    trucks:   fuel_type (40), load_capacity_pounds (57)
    traffic:  no_of_vehicles (1152)
Every other table (routes, truck_schedule, city_weather, routes_weather)
was clean, so no functions are needed for those.
"""

import pandas as pd
from load_data import load_all


def clean_drivers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Categorical nulls -> fill with the mode (most frequent value).
    # 'gender' is ~94% male, so mode-fill is a safe, low-risk choice.
    df["gender"] = df["gender"].fillna(df["gender"].mode()[0])

    # 'driving_style' is close to a 50/50 split (648 proactive vs 600
    # conservative), so mode-fill here is a weaker assumption than for
    # gender -- flagging this so we remember it if the model behaves
    # oddly on this feature later.
    df["driving_style"] = df["driving_style"].fillna(df["driving_style"].mode()[0])
    return df


def clean_trucks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fuel_type"] = df["fuel_type"].fillna(df["fuel_type"].mode()[0])

    # load_capacity_pounds is right-skewed (mean 8610 vs median 6000),
    # so the median is a more representative fill value than the mean
    # -- using the mean would overstate capacity for smaller trucks.
    df["load_capacity_pounds"] = df["load_capacity_pounds"].fillna(
        df["load_capacity_pounds"].median()
    )
    return df


def clean_traffic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Traffic volume varies a lot route to route (a highway vs a rural
    # road), so filling with one global median would be misleading.
    # Instead we fill within each route's own median first...
    df["no_of_vehicles"] = df.groupby("route_id")["no_of_vehicles"].transform(
        lambda s: s.fillna(s.median())
    )
    # ...and fall back to the global median for the rare case where an
    # entire route_id group has no non-null values to compute from.
    df["no_of_vehicles"] = df["no_of_vehicles"].fillna(df["no_of_vehicles"].median())
    return df


def clean_all(tables: dict) -> dict:
    """Applies the relevant cleaning function to each table that needs it."""
    tables["drivers"] = clean_drivers(tables["drivers"])
    tables["trucks"] = clean_trucks(tables["trucks"])
    tables["traffic"] = clean_traffic(tables["traffic"])
    return tables


if __name__ == "__main__":
    # Quick sanity check: load raw tables, clean them, confirm zero
    # nulls remain in the columns we targeted.
    tables = load_all()
    tables = clean_all(tables)

    for name in ["drivers", "trucks", "traffic"]:
        remaining_nulls = tables[name].isnull().sum().sum()
        print(f"{name:10s} remaining nulls after cleaning: {remaining_nulls}")