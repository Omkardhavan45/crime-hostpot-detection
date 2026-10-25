"""
Cluster Analysis and Hotspot Profiling Module
Aggregates crime cluster characteristics, calculates spatial centroids and radii,
and identifies operational risk metrics (dominant crime, peak hours, arrest rates).
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from .preprocessing import EARTH_RADIUS_KM

def haversine_vectorized(lat1_rad, lon1_rad, lat2_rad, lon2_rad) -> np.ndarray:
    """Computes haversine distance in kilometers between point(s) 1 and point(s) 2."""
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return EARTH_RADIUS_KM * c

def analyze_clusters(
    df: pd.DataFrame,
    cluster_col: str = "cluster"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Computes comprehensive statistics for each detected hotspot cluster:
    - incident_count: Total crimes
    - centroid_lat, centroid_lon: Geographic center
    - radius_km: 90th percentile spread from centroid
    - max_radius_km: Maximum spread from centroid
    - dominant_crime: Most frequent crime category
    - dominant_crime_pct: Share of the dominant crime
    - peak_hour: Most frequent incident hour
    - peak_time_bucket: Most frequent patrol window
    - arrest_rate: Proportion of incidents resulting in arrest
    - dominant_location: Primary physical premises
    - risk_rank: Sorted ranking by incident density/count

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with assigned cluster labels and spatial/temporal columns.
    cluster_col : str
        Name of cluster column (-1 indicates noise).

    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, Any]]:
        - clusters_df: Ranked table of hotspots
        - summary_stats: Global clustering performance metrics
    """
    if cluster_col not in df.columns:
        raise ValueError(f"Cluster column '{cluster_col}' not found in dataframe.")

    total_records = len(df)
    noise_mask = df[cluster_col] == -1
    noise_count = noise_mask.sum()
    noise_ratio = noise_count / total_records

    clusters = df[~noise_mask]
    unique_clusters = clusters[cluster_col].unique()
    num_clusters = len(unique_clusters)

    print(f"[ClusterAnalysis] Analyzing {num_clusters} valid clusters ({len(clusters):,} clustered incidents)...")

    cluster_records = []

    for c_id in unique_clusters:
        c_data = clusters[clusters[cluster_col] == c_id]
        count = len(c_data)

        # Centroid
        centroid_lat = float(c_data["Latitude"].mean())
        centroid_lon = float(c_data["Longitude"].mean())

        # Radius (90th percentile distance from centroid)
        lat_rad = np.radians(c_data["Latitude"].to_numpy())
        lon_rad = np.radians(c_data["Longitude"].to_numpy())
        c_lat_rad = np.radians(centroid_lat)
        c_lon_rad = np.radians(centroid_lon)

        distances = haversine_vectorized(c_lat_rad, c_lon_rad, lat_rad, lon_rad)
        radius_90 = float(np.percentile(distances, 90))
        radius_max = float(np.max(distances))

        # Dominant Crime Type
        top_crimes = c_data["Primary Type"].value_counts()
        dominant_crime = top_crimes.index[0] if len(top_crimes) > 0 else "UNKNOWN"
        dominant_crime_pct = float(top_crimes.iloc[0] / count) * 100 if len(top_crimes) > 0 else 0.0

        # Dominant Location Description
        top_locs = c_data["Location Description"].value_counts()
        dominant_location = top_locs.index[0] if len(top_locs) > 0 else "UNKNOWN"

        # Peak Hour
        top_hours = c_data["hour"].value_counts()
        peak_hour = int(top_hours.index[0]) if len(top_hours) > 0 else -1

        # Peak Time Bucket
        if "time_bucket" in c_data.columns:
            top_buckets = c_data["time_bucket"].value_counts()
            peak_time_bucket = str(top_buckets.index[0]) if len(top_buckets) > 0 else "UNKNOWN"
        else:
            peak_time_bucket = "N/A"

        # Arrest Rate
        arrest_rate = float(c_data["Arrest"].mean() * 100) if "Arrest" in c_data.columns else 0.0

        cluster_records.append({
            "cluster_id": int(c_id),
            "incident_count": count,
            "centroid_lat": round(centroid_lat, 5),
            "centroid_lon": round(centroid_lon, 5),
            "radius_km": round(radius_90, 3),
            "max_radius_km": round(radius_max, 3),
            "dominant_crime": dominant_crime,
            "dominant_crime_pct": round(dominant_crime_pct, 1),
            "dominant_location": dominant_location,
            "peak_hour": peak_hour,
            "peak_time_bucket": peak_time_bucket,
            "arrest_rate": round(arrest_rate, 1)
        })

    clusters_df = pd.DataFrame(cluster_records)

    # Sort by incident count descending to assign risk ranks
    if not clusters_df.empty:
        clusters_df = clusters_df.sort_values(by="incident_count", ascending=False).reset_index(drop=True)
        clusters_df["risk_rank"] = clusters_df.index + 1
        # Reorder columns with risk_rank first
        cols = ["risk_rank", "cluster_id", "incident_count", "dominant_crime", "dominant_crime_pct",
                "peak_hour", "peak_time_bucket", "arrest_rate", "radius_km", "max_radius_km",
                "dominant_location", "centroid_lat", "centroid_lon"]
        clusters_df = clusters_df[cols]

    summary_stats = {
        "total_incidents": total_records,
        "clustered_incidents": int(len(clusters)),
        "noise_incidents": int(noise_count),
        "noise_ratio_pct": round(noise_ratio * 100, 2),
        "total_hotspots": num_clusters,
        "max_hotspot_size": int(clusters_df["incident_count"].max()) if not clusters_df.empty else 0,
        "median_hotspot_size": float(clusters_df["incident_count"].median()) if not clusters_df.empty else 0,
        "mean_radius_km": round(float(clusters_df["radius_km"].mean()), 3) if not clusters_df.empty else 0.0
    }

    print(f"[ClusterAnalysis] Top hotspot has {summary_stats['max_hotspot_size']:,} crimes.")
    print(f"[ClusterAnalysis] Median hotspot size: {summary_stats['median_hotspot_size']} crimes.")

    return clusters_df, summary_stats
