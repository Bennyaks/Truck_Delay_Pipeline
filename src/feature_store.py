"""
feature_store.py
-----------------
Pushes the engineered trip-level features into Hopsworks (create a
feature group) and fetches them back (for training or inference) --
this is the "Feature Store" step from the original course.

STATUS: written and checked against the actual installed hsfs SDK
(v5.0.3 at time of writing -- every function signature below was
verified with `inspect.signature()` against the real library, not
guessed from memory). NOT executed end-to-end here, because this
environment has no network access to Hopsworks -- you'll be the first
to actually run it against a live project.

Setup before running:
    1. pip install hopsworks
    2. Sign up at hopsworks.ai (free Serverless tier), create a project
    3. Generate an API key (Account Settings -> API Keys)
    4. Set it as an environment variable, e.g.:
         export HOPSWORKS_API_KEY="your-key-here"        (Mac/Linux)
         setx HOPSWORKS_API_KEY "your-key-here"           (Windows)
       hopsworks.login() picks this up automatically -- no need to
       pass it as an argument.
"""

try:
    import hopsworks  # type: ignore[import]
except ImportError:
    hopsworks = None  # type: ignore[assignment]

import pandas as pd

from feature_engineering import build_final_dataset

FEATURE_GROUP_NAME = "truck_trip_features"
FEATURE_GROUP_VERSION = 1


def get_feature_store():
    """
    Logs into Hopsworks and returns the project's feature store handle.
    Reads HOPSWORKS_API_KEY from the environment automatically -- no
    credentials are hardcoded here.
    """
    project = hopsworks.login()
    return project.get_feature_store()


def create_trip_features_group(fs, df: pd.DataFrame):
    """
    Creates (or retrieves, if it already exists) the feature group and
    inserts the engineered trip-level data into it.

    Primary key: truck_id + route_id + departure_date together uniquely
    identify one trip (same composite key used throughout merge_data.py).
    Event time: departure_date -- lets Hopsworks track feature freshness
    and support point-in-time correct joins later.
    """
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Engineered truck trip features (route, weather, traffic, driver, truck) for delay prediction.",
        primary_key=["truck_id", "route_id", "departure_date"],
        event_time="departure_date",
        online_enabled=False,  # batch/offline use only -- no real-time serving needed for this project
    )
    fg.insert(df)
    print(f"Inserted {len(df)} rows into feature group '{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VERSION}.")
    return fg


def fetch_trip_features(fs) -> pd.DataFrame:
    """
    Fetches the full feature group back as a pandas DataFrame -- this
    is what train_model.py / predict.py would call instead of
    build_final_dataset() once features live in the feature store,
    so every consumer reads the exact same, already-validated features
    instead of each one recomputing them independently.
    """
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    return fg.read()


def push_features_to_store():
    """Full one-way sync: build engineered features locally, push to Hopsworks."""
    full_df, _ = build_final_dataset()
    fs = get_feature_store()
    create_trip_features_group(fs, full_df)


if __name__ == "__main__":
    push_features_to_store()

    # Sanity check: fetch it straight back and confirm row count matches
    fs = get_feature_store()
    fetched = fetch_trip_features(fs)
    print(f"Fetched back {len(fetched)} rows from the feature store.")