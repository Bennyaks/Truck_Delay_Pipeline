"""
config.py
---------
Single source of truth for file paths and shared constants.
Keeping these in one place means if the folder structure changes,
we only edit this file instead of hunting through every script.
"""

import os

# Project root = one level up from this file (src/ -> project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Raw CSV file paths (matches the "Training_data" folder from the course)
PATHS = {
    "drivers": os.path.join(RAW_DIR, "drivers_table.csv"),
    "trucks": os.path.join(RAW_DIR, "trucks_table.csv"),
    "routes": os.path.join(RAW_DIR, "routes_table.csv"),
    "traffic": os.path.join(RAW_DIR, "traffic_table.csv"),
    "truck_schedule": os.path.join(RAW_DIR, "truck_schedule_table.csv"),
    "city_weather": os.path.join(RAW_DIR, "city_weather.csv"),
    "routes_weather": os.path.join(RAW_DIR, "routes_weather.csv"),
}

# Output path for the final merged/feature-engineered dataset
FINAL_FEATURES_PATH = os.path.join(PROCESSED_DIR, "final_features.csv")