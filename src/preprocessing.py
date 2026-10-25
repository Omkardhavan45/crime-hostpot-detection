"""
Data Preprocessing and Feature Engineering Module
Handles data cleaning, coordinate bounds validation, datetime extraction, and geospatial transformations.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

# Earth's mean radius in kilometers
EARTH_RADIUS_KM = 6371.0088

# Chicago bounding box coordinates
CHICAGO_BBOX = {
    "lat_min": 41.60,
    "lat_max": 42.10,
    "lon_min": -87.95,
    "lon_max": -87.50
}

def clean_and_filter_data(
    df: pd.DataFrame,
    bbox: dict = CHICAGO_BBOX,
    save_cleaned_path: Optional[str] = "data/cleaned_crimes_2016.csv"
) -> pd.DataFrame:
    """
    Cleans raw crime data:
    - Drops missing Latitude, Longitude, or Date
    - Removes (0,0) and coordinates outside Chicago bounding box
    - Parses dates and extracts temporal features
    - Optionally saves cleaned dataset to disk

    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe for the year
    bbox : dict
        Latitude and longitude bounds
    save_cleaned_path : str, optional
        Path to save cleaned CSV

    Returns:
    --------
    pd.DataFrame
        Cleaned and enriched dataframe
    """
    initial_rows = len(df)
    print(f"[Preprocessing] Starting cleaning with {initial_rows:,} records.")

    # 1. Drop missing lat/lon/date
    df_clean = df.dropna(subset=["Latitude", "Longitude", "Date"]).copy()
    dropped_nulls = initial_rows - len(df_clean)
    print(f"[Preprocessing] Dropped {dropped_nulls:,} rows with missing coordinates/dates.")

    # 2. Exclude (0,0) coordinates
    df_clean = df_clean[(df_clean["Latitude"] != 0) & (df_clean["Longitude"] != 0)]

    # 3. Chicago Bounding Box Filter
    in_bounds = (
        (df_clean["Latitude"] >= bbox["lat_min"]) &
        (df_clean["Latitude"] <= bbox["lat_max"]) &
        (df_clean["Longitude"] >= bbox["lon_min"]) &
        (df_clean["Longitude"] <= bbox["lon_max"])
    )
    df_clean = df_clean[in_bounds].copy()
    print(f"[Preprocessing] Rows within Chicago bounding box: {len(df_clean):,}")

    # 4. Temporal feature extraction
    df_clean = derive_temporal_features(df_clean)

    # 5. Clean text categories
    if "Primary Type" in df_clean.columns:
        df_clean["Primary Type"] = df_clean["Primary Type"].astype(str).str.strip().str.upper()
    if "Location Description" in df_clean.columns:
        df_clean["Location Description"] = df_clean["Location Description"].fillna("OTHER").astype(str).str.strip().str.upper()
    if "Arrest" in df_clean.columns:
        # Standardize Arrest to boolean
        if df_clean["Arrest"].dtype != bool:
            df_clean["Arrest"] = df_clean["Arrest"].astype(str).str.lower().isin(["true", "1", "yes", "t"])

    if save_cleaned_path:
        out_dir = os.path.dirname(save_cleaned_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df_clean.to_csv(save_cleaned_path, index=False)
        print(f"[Preprocessing] Saved cleaned dataset to: {save_cleaned_path}")

    return df_clean

def derive_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses 'Date' column and extracts hour, day_of_week, day_name, month, and time_bucket.
    """
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    if df["Date"].isna().all():  # Fallback if format differs
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    df["hour"] = df["Date"].dt.hour
    df["month"] = df["Date"].dt.month
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["day_name"] = df["Date"].dt.day_name()

    # Time buckets for patrol scheduling:
    # Night: 00:00 - 05:59, Morning: 06:00 - 11:59, Afternoon: 12:00 - 17:59, Evening: 18:00 - 23:59
    bins = [-1, 5, 11, 17, 24]
    labels = ["Night (00-06)", "Morning (06-12)", "Afternoon (12-18)", "Evening (18-24)"]
    df["time_bucket"] = pd.cut(df["hour"], bins=bins, labels=labels)

    return df

def km_to_radians(km: float) -> float:
    """Converts distance in kilometers to radians on Earth's surface."""
    return km / EARTH_RADIUS_KM

def radians_to_km(rad: float) -> float:
    """Converts radians on Earth's surface to kilometers."""
    return rad * EARTH_RADIUS_KM

def to_radians(df: pd.DataFrame, lat_col: str = "Latitude", lon_col: str = "Longitude") -> np.ndarray:
    """
    Converts Latitude and Longitude columns to an (N, 2) array of radians.
    Order is [latitude, longitude] in radians as expected by scikit-learn's haversine metric.
    """
    return np.radians(df[[lat_col, lon_col]].to_numpy())
