"""
merge_data.py
-------------
Builds one row-per-trip dataset by joining every table onto
truck_schedule. The tricky part is that traffic (2.6M rows) and
routes_weather (425K rows) are TIME-SERIES per route, while
truck_schedule is one row per actual trip -- so a plain merge on
route_id alone would multiply rows massively AND mix weather/traffic
from unrelated times into the wrong trip.

Strategy: for each trip, only look at traffic/weather observations
that fall INSIDE that trip's own departure->arrival window, then
aggregate (mean/max) down to a single value per trip.
"""

import pandas as pd
from load_data import load_all
from clean_data import clean_all

# Columns that uniquely identify a scheduled trip.
TRIP_KEYS = [
    "truck_id",
    "route_id",
    "departure_date",
]

# Same as TRIP_KEYS but includes the trip end time.
TRIP_WINDOW = TRIP_KEYS + ["estimated_arrival"]


def add_route_info(schedule: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    """Attach origin/destination city IDs, distance, and average_hours to each trip."""
    return schedule.merge(routes, on="route_id", how="left")


def aggregate_traffic(schedule: pd.DataFrame, traffic: pd.DataFrame) -> pd.DataFrame:
    """
    For each trip, average 'no_of_vehicles' and flag whether ANY
    accident occurred on that route during the trip's time window.

    Traffic only has date + hour (no minutes), so we build a proper
    timestamp column first to compare against departure/arrival.
    """
    traffic = traffic.copy()
    traffic["timestamp"] = traffic["date"] + pd.to_timedelta(traffic["hour"], unit="h")

    # Join every trip to every traffic row on the SAME route first.
    # This is safe size-wise (~13M rows) because we immediately filter
    # down to only the rows inside each trip's own time window.
    merged = schedule[TRIP_WINDOW].merge(
        traffic[["route_id", "timestamp", "no_of_vehicles", "accident"]],
        on="route_id",
        how="left",
    )

    in_window = merged[
        (merged["timestamp"] >= merged["departure_date"])
        & (merged["timestamp"] <= merged["estimated_arrival"])
    ]

    agg = (
        in_window.groupby(TRIP_KEYS)
        .agg(no_of_vehicles=("no_of_vehicles", "mean"), accident=("accident", "max"))
        .reset_index()
    )

    result = schedule.merge(
    agg,
    on=TRIP_KEYS,
    how="left",
)
    # Trips with no traffic reading in their exact window (rare, short
    # trips falling between hourly readings) fall back to the route's
    # overall average rather than being left null.
    route_fallback = traffic.groupby("route_id")["no_of_vehicles"].mean()
    result["no_of_vehicles"] = result["no_of_vehicles"].fillna(
        result["route_id"].map(route_fallback)
    )
    result["accident"] = result["accident"].fillna(0)
    return result


def aggregate_route_weather(schedule: pd.DataFrame, routes_weather: pd.DataFrame) -> pd.DataFrame:
    """
    For each trip, average every weather metric (temp, wind_speed,
    precip, humidity, visibility, pressure, chance-of-X columns) across
    all routes_weather readings that fall inside the trip's window.
    Same route-join-then-filter-then-aggregate pattern as traffic.
    """
    weather_cols = [
        "temp", "wind_speed", "precip", "humidity", "visibility",
        "pressure", "chanceofrain", "chanceoffog", "chanceofsnow", "chanceofthunder",
    ]

    merged = schedule[TRIP_WINDOW].merge(
        routes_weather[["route_id", "date"] + weather_cols],
        on="route_id",
        how="left",
    )

    in_window = merged[
        (merged["date"] >= merged["departure_date"])
        & (merged["date"] <= merged["estimated_arrival"])
    ]

    agg = (
        in_window.groupby(TRIP_KEYS)[weather_cols]
        .mean()
        .reset_index()
    )

    result = schedule.merge(
        agg, on=TRIP_KEYS, how="left"
    )
    # Fallback for trips with no reading inside their window: route's overall average.
    route_fallback = routes_weather.groupby("route_id")[weather_cols].mean()
    for col in weather_cols:
        result[col] = result[col].fillna(result["route_id"].map(route_fallback[col]))
    return result


def add_nearest_city_weather(
    schedule: pd.DataFrame, city_weather: pd.DataFrame
) -> pd.DataFrame:
    """
    Attach ORIGIN weather at the moment of departure, and DESTINATION
    weather at the moment of estimated arrival -- using the nearest
    available hourly reading for that city (merge_asof = 'nearest match'
    join, built for exactly this kind of time-series lookup).
    """
    cw = city_weather.copy()
    cw["timestamp"] = cw["date"] + pd.to_timedelta(cw["hour"], unit="h")
    cw = cw.sort_values("timestamp")

    # merge_asof requires the two datetime "by" columns to share the exact
    # same resolution (ns vs us) or it raises a MergeError -- the schedule
    # table's timestamps carry sub-second precision (us) while city_weather's
    # don't, so we align both to nanosecond resolution here.
    cw["timestamp"] = cw["timestamp"].astype("datetime64[ns]")
    schedule = schedule.copy()
    schedule["departure_date"] = schedule["departure_date"].astype("datetime64[ns]")
    schedule["estimated_arrival"] = schedule["estimated_arrival"].astype("datetime64[ns]")

    weather_cols = [
        "temp", "wind_speed", "precip", "humidity", "visibility",
        "pressure", "chanceofrain", "chanceoffog", "chanceofsnow", "chanceofthunder",
    ]

    # --- Origin weather at departure time ---
    origin = schedule[TRIP_KEYS+ ["origin_id"]].sort_values(
        "departure_date"
    )
    origin_weather = pd.merge_asof(
        origin,
        cw.rename(columns={"city_id": "origin_id"}),
        left_on="departure_date",
        right_on="timestamp",
        by="origin_id",
        direction="nearest",
    )
    origin_weather = origin_weather[TRIP_KEYS + weather_cols]
    origin_weather = origin_weather.rename(columns={c: f"origin_{c}" for c in weather_cols})

    # --- Destination weather at estimated arrival time ---
    dest = schedule[TRIP_WINDOW+ ["destination_id"]].sort_values(
        "estimated_arrival"
    )
    dest_weather = pd.merge_asof(
        dest,
        cw.rename(columns={"city_id": "destination_id"}),
        left_on="estimated_arrival",
        right_on="timestamp",
        by="destination_id",
        direction="nearest",
    )
    dest_weather = dest_weather[TRIP_KEYS + weather_cols]
    dest_weather = dest_weather.rename(columns={c: f"destination_{c}" for c in weather_cols})

    result = schedule.merge(origin_weather, on=TRIP_KEYS, how="left")
    result = result.merge(dest_weather, on=TRIP_KEYS, how="left")
    return result


def add_driver_and_truck_info(
    schedule: pd.DataFrame, drivers: pd.DataFrame, trucks: pd.DataFrame
) -> pd.DataFrame:
    """
    drivers.vehicle_no and trucks.truck_id are the SAME id space as
    schedule.truck_id (confirmed by inspecting the raw data), so both
    join directly on truck_id.
    """
    result = schedule.merge(
        drivers.rename(columns={"vehicle_no": "truck_id"}), on="truck_id", how="left"
    )
    result = result.merge(trucks, on="truck_id", how="left")
    return result


def build_merged_dataset() -> pd.DataFrame:
    """Runs the full load -> clean -> merge pipeline and returns one trip-per-row dataframe."""
    tables = load_all()
    tables = clean_all(tables)

    df = add_route_info(tables["truck_schedule"], tables["routes"])
    df = aggregate_traffic(df, tables["traffic"])
    df = aggregate_route_weather(df, tables["routes_weather"])
    df = add_nearest_city_weather(df, tables["city_weather"])
    df = add_driver_and_truck_info(df, tables["drivers"], tables["trucks"])
    return df


if __name__ == "__main__":
    df = build_merged_dataset()
    print("Final merged shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nNulls remaining per column:")
    nulls = df.isnull().sum()
    print(nulls[nulls > 0] if nulls.sum() else "none")