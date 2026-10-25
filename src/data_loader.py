"""
Data Loader Module for Crime Hotspot Detection
Reads raw crime records in chunks and filters for the selected year.
"""

import os
import pandas as pd
from typing import Optional, List

REQUIRED_COLUMNS = [
    "Date",
    "Primary Type",
    "Location Description",
    "Arrest",
    "Year",
    "Latitude",
    "Longitude"
]

def load_raw_crime_data(
    filepath: str = "crimes.csv",
    year: int = 2016,
    chunksize: int = 500000,
    usecols: Optional[List[str]] = None,
    cached_path: Optional[str] = "data/cleaned_crimes_2016.csv"
) -> pd.DataFrame:
    """
    Loads crime records for a specified year from the CSV file.
    If cached_path exists, loads directly from cache.
    Otherwise, reads filepath in chunks to conserve memory.

    Parameters:
    -----------
    filepath : str
        Path to raw crimes.csv
    year : int
        Target year to extract (default: 2016)
    chunksize : int
        Number of rows per chunk
    usecols : list of str, optional
        Columns to load
    cached_path : str, optional
        Path to save/load cached single-year data

    Returns:
    --------
    pd.DataFrame
        Filtered dataframe for the specified year.
    """
    if cached_path and os.path.exists(cached_path):
        print(f"[DataLoader] Found cached dataset at: {cached_path}")
        df = pd.read_csv(cached_path)
        print(f"[DataLoader] Successfully loaded {len(df):,} rows from cache.")
        return df

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Source file not found at: {filepath}")

    if usecols is None:
        usecols = REQUIRED_COLUMNS

    print(f"[DataLoader] Reading '{filepath}' in chunks of {chunksize:,} for year {year}...")
    chunks = pd.read_csv(
        filepath,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
        on_bad_lines="skip"
    )

    year_dfs = []
    total_scanned = 0

    for idx, chunk in enumerate(chunks):
        total_scanned += len(chunk)
        if "Year" in chunk.columns:
            subset = chunk[chunk["Year"] == year]
        else:
            subset = chunk
        if len(subset) > 0:
            year_dfs.append(subset)
        if (idx + 1) % 4 == 0:
            print(f"[DataLoader] Scanned {total_scanned:,} rows...")

    if not year_dfs:
        raise ValueError(f"No records found for year {year} in {filepath}")

    df_year = pd.concat(year_dfs, ignore_index=True)
    print(f"[DataLoader] Extraction complete: {len(df_year):,} records found for year {year}.")

    return df_year
