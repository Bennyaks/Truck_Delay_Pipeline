# ============================================================
# NOTE
# ------------------------------------------------------------
# This file is intentionally fully commented out.
#
# It serves as a reference implementation showing how this
# project can be extended from local CSV files to AWS RDS and
# SageMaker, as required in the original project specification.
#
# Uncomment and configure it only after creating your own AWS
# RDS instances and credentials.
# ============================================================



"""
aws_rds.py
----------
OPTIONAL. Everything in this file is commented out on purpose -- it's
a drop-in replacement for load_data.py's CSV-based loaders that pulls
straight from AWS RDS instead. It's written and its logic has been
validated (tested against a stand-in database using the exact table
names from your truck-eta-mysql.sql / truck-eta-postgres.sql dumps),
but it's inert until YOU have real RDS instances to point it at.

If you don't have AWS RDS set up: ignore this file completely, the
rest of the pipeline (load_data.py) already works fully from the local
CSVs and doesn't need this.

If you DO have RDS set up:
  1. pip install sqlalchemy pymysql psycopg2-binary
  2. Restore truck-eta-mysql.sql into a MySQL RDS instance
     (holds: city_weather, drivers_details, traffic_details,
      truck_details, truck_schedule_data)
  3. Restore truck-eta-postgres.sql into a PostgreSQL RDS instance
     (holds: routes_details, routes_weather)
  4. Set the environment variables below (recommended over hardcoding
     credentials directly -- keeps secrets out of version control)
  5. Uncomment everything below
  6. In merge_data.py / feature_engineering.py, swap:
         from load_data import load_all
     for:
         from aws_rds import load_all_from_rds as load_all
     No other code changes needed -- load_all_from_rds() returns the
     same dict-of-DataFrames shape, with the same friendly table names
     and the same dtype fixes (parsed dates, normalized hour column)
     that load_data.py applies to the CSVs.
"""

# import os
# import pandas as pd
# from sqlalchemy import create_engine
# from utils import normalize_hour
#
# # --- MySQL RDS connection ---
# # (holds: city_weather, drivers_details, traffic_details,
# #  truck_details, truck_schedule_data)
# MYSQL_HOST = os.environ.get("TRUCK_ETA_MYSQL_HOST", "your-rds-endpoint.rds.amazonaws.com")
# MYSQL_PORT = os.environ.get("TRUCK_ETA_MYSQL_PORT", "3306")
# MYSQL_USER = os.environ.get("TRUCK_ETA_MYSQL_USER", "admin")
# MYSQL_PASSWORD = os.environ.get("TRUCK_ETA_MYSQL_PASSWORD", "")
# MYSQL_DB = os.environ.get("TRUCK_ETA_MYSQL_DB", "truck_eta")
#
# mysql_engine = create_engine(
#     f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
# )
#
# # --- PostgreSQL RDS connection ---
# # (holds: routes_details, routes_weather)
# POSTGRES_HOST = os.environ.get("TRUCK_ETA_POSTGRES_HOST", "your-rds-endpoint.rds.amazonaws.com")
# POSTGRES_PORT = os.environ.get("TRUCK_ETA_POSTGRES_PORT", "5432")
# POSTGRES_USER = os.environ.get("TRUCK_ETA_POSTGRES_USER", "postgres")
# POSTGRES_PASSWORD = os.environ.get("TRUCK_ETA_POSTGRES_PASSWORD", "")
# POSTGRES_DB = os.environ.get("TRUCK_ETA_POSTGRES_DB", "truck_eta")
#
# postgres_engine = create_engine(
#     f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
# )
#
#
# def load_table_from_mysql(table_name: str) -> pd.DataFrame:
#     """table_name: one of city_weather, drivers_details, traffic_details,
#     truck_details, truck_schedule_data."""
#     return pd.read_sql_table(table_name, mysql_engine)
#
#
# def load_table_from_postgres(table_name: str) -> pd.DataFrame:
#     """table_name: routes_details or routes_weather."""
#     return pd.read_sql_table(table_name, postgres_engine)
#
#
# def load_all_from_rds() -> dict:
#     """
#     Drop-in replacement for load_data.load_all(). Pulls from RDS,
#     renames RDS's dump-schema names back to this project's friendly
#     names, and applies the same date-parsing / hour-normalization
#     fixes load_data.py applies to the CSVs -- so everything downstream
#     (merge_data.py, feature_engineering.py, train_model.py, predict.py)
#     works completely unchanged regardless of whether data came from
#     CSVs or RDS.
#     """
#     tables = {
#         "drivers": load_table_from_mysql("drivers_details"),
#         "trucks": load_table_from_mysql("truck_details"),
#         "traffic": load_table_from_mysql("traffic_details"),
#         "truck_schedule": load_table_from_mysql("truck_schedule_data"),
#         "city_weather": load_table_from_mysql("city_weather"),
#         "routes": load_table_from_postgres("routes_details"),
#         "routes_weather": load_table_from_postgres("routes_weather"),
#     }
#
#     tables["traffic"]["date"] = pd.to_datetime(tables["traffic"]["date"], errors="coerce")
#     tables["traffic"]["hour"] = normalize_hour(tables["traffic"]["hour"])
#     tables["truck_schedule"]["departure_date"] = pd.to_datetime(tables["truck_schedule"]["departure_date"])
#     tables["truck_schedule"]["estimated_arrival"] = pd.to_datetime(tables["truck_schedule"]["estimated_arrival"])
#     tables["city_weather"]["date"] = pd.to_datetime(tables["city_weather"]["date"])
#     tables["city_weather"]["hour"] = normalize_hour(tables["city_weather"]["hour"])
#     tables["routes_weather"]["date"] = pd.to_datetime(tables["routes_weather"]["date"])
#     return tables
#
#
# if __name__ == "__main__":
#     tables = load_all_from_rds()
#     for name, df in tables.items():
#         print(f"{name:15s} shape={df.shape}")
#     print("\nAll tables loaded successfully from RDS.")


# ============================================================
# AWS SAGEMAKER NOTE
# ------------------------------------------------------------
# No code changes are needed to run this project inside a SageMaker
# notebook instance -- it's the same Python/pandas code either way.
# Steps to move this project there once you have an instance running:
#
#   1. Open Jupyter/JupyterLab from the SageMaker console.
#   2. Upload (or git clone) this entire truck_delay_pipeline/ folder.
#   3. Open a terminal in JupyterLab:
#        cd truck_delay_pipeline
#        pip install -r requirements.txt
#   4. If pulling from RDS (not CSVs), make sure the SageMaker
#      notebook's security group has network access to your RDS
#      instances' security group (same VPC, or an inbound rule
#      allowing the notebook's IP/security group on port 3306/5432).
#   5. Run notebooks/01_eda.ipynb or src/train_model.py exactly as
#      you do locally.
#   6. Stop the notebook instance when done to avoid ongoing charges.
# ============================================================