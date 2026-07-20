import pandas as pd

def normalize_hour(hour_series: pd.Series) -> pd.Series:
    """
    city_weather.csv and traffic_table.csv store 'hour' as 0, 100, 200 ...
    2300 (military-clock style), NOT as 0-23. If we don't fix this, any
    merge/groupby on 'hour' against routes_weather (which has a real
    0-23 hour embedded in a datetime) will silently produce wrong joins.
    Formula: divide by 100 -> gives the true 0-23 hour.
    """
    return (hour_series // 100).astype(int)