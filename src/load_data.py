"""
load_data.py
------------
Reads every raw CSV into a pandas DataFrame and applies the minimal
fixes needed so the data is USABLE (correct dtypes, parsed dates).
Deeper cleaning (nulls, dedup, outliers) is deliberately left for
clean_data.py so this file stays a simple, predictable "load" step.
"""

import pandas as pd
from config import PATHS
from utils import normalize_hour

def load_drivers() -> pd.DataFrame:
    df = pd.read_csv(PATHS["drivers"])
    return df


def load_trucks() -> pd.DataFrame:
    df = pd.read_csv(PATHS["trucks"])
    return df


def load_routes() -> pd.DataFrame:
    df = pd.read_csv(PATHS["routes"])
    return df


def load_traffic() -> pd.DataFrame:
    df = pd.read_csv(PATHS["traffic"])
    # Parse the plain date column so later date-based joins/filters work
    df["date"] = pd.to_datetime(
                    df["date"],
                    errors="coerce"
                )
    df["hour"] = normalize_hour(df["hour"])
    return df


def load_truck_schedule() -> pd.DataFrame:
    df = pd.read_csv(PATHS["truck_schedule"])
    # These two columns are full datetimes (departure/arrival), not just dates
    df["departure_date"] = pd.to_datetime(df["departure_date"])
    df["estimated_arrival"] = pd.to_datetime(df["estimated_arrival"])
    return df


def load_city_weather() -> pd.DataFrame:
    df = pd.read_csv(PATHS["city_weather"])
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = normalize_hour(df["hour"])
    return df


def load_routes_weather() -> pd.DataFrame:
    df = pd.read_csv(PATHS["routes_weather"])
    # Here the column is literally called "Date" (capital D) and already
    # contains a full timestamp with the hour baked in, so no hour-fix needed.
    df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_all() -> dict[str, pd.DataFrame]:
    """Convenience loader: returns every table as a dict of DataFrames."""
    tables = {
        "drivers": load_drivers(),
        "trucks": load_trucks(),
        "routes": load_routes(),
        "traffic": load_traffic(),
        "truck_schedule": load_truck_schedule(),
        "city_weather": load_city_weather(),
        "routes_weather": load_routes_weather(),
    }
    return tables


if __name__ == "__main__":
    # Quick sanity check when running this file directly:
    # confirms every CSV loads and prints row/col counts + dtypes.
    tables = load_all()
    for name, df in tables.items():
        print(f"{name:15s} shape={df.shape}")
    print("\nAll tables loaded successfully.")
    

def table_info(name, df):
    memory = df.memory_usage(deep=True).sum() / 1024**2

    print(
        f"{name:18}"
        f"{df.shape[0]:>10,} rows"
        f"{df.shape[1]:>4} cols"
        f"{memory:>8.2f} MB"
    )