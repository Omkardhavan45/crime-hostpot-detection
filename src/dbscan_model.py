"""
DBSCAN Clustering and Parameter Selection Module
Implements k-distance elbow computation, Haversine-based DBSCAN clustering, and hyperparameter grid evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from typing import Tuple, List, Optional
from .preprocessing import km_to_radians, radians_to_km

def compute_k_distance(
    coords_rad: np.ndarray,
    k: int = 25,
    sample_size: Optional[int] = 25000,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes k-nearest neighbor distances for elbow curve analysis.
    Uses BallTree with haversine metric.

    Parameters:
    -----------
    coords_rad : np.ndarray
        Array of (lat, lon) coordinates in radians.
    k : int
        Number of nearest neighbors (usually corresponds to min_samples).
    sample_size : int, optional
        Sample size for fast computation. If None or len(coords_rad) <= sample_size, uses all.
    random_state : int
        Seed for reproducibility.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]:
        (sorted_distances_km, query_indices)
    """
    n_points = len(coords_rad)
    if sample_size and n_points > sample_size:
        rng = np.random.default_rng(random_state)
        sample_indices = rng.choice(n_points, size=sample_size, replace=False)
        query_coords = coords_rad[sample_indices]
    else:
        query_coords = coords_rad
        sample_indices = np.arange(n_points)

    print(f"[DBSCAN] Fitting BallTree NearestNeighbors (k={k}) on {len(coords_rad):,} points...")
    nn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", metric="haversine", n_jobs=-1)
    nn.fit(coords_rad)

    print(f"[DBSCAN] Querying {len(query_coords):,} sample points...")
    distances, _ = nn.kneighbors(query_coords)

    # Distances to the k-th neighbor (last column), converted from radians to km
    k_distances_rad = distances[:, -1]
    k_distances_km = np.sort(radians_to_km(k_distances_rad))

    return k_distances_km, sample_indices

def run_dbscan(
    coords_rad: np.ndarray,
    eps_km: float = 0.20,
    min_samples: int = 25,
    n_jobs: int = -1
) -> np.ndarray:
    """
    Runs DBSCAN clustering with the haversine metric.

    Parameters:
    -----------
    coords_rad : np.ndarray
        (N, 2) array of coordinates in radians [lat, lon].
    eps_km : float
        Epsilon neighborhood radius in kilometers.
    min_samples : int
        Minimum points required to form a dense core point.
    n_jobs : int
        Number of parallel jobs.

    Returns:
    --------
    np.ndarray:
        Cluster labels (-1 for noise).
    """
    eps_rad = km_to_radians(eps_km)
    print(f"[DBSCAN] Initializing DBSCAN with eps={eps_km:.3f} km ({eps_rad:.6f} rad), min_samples={min_samples}...")

    db = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
        n_jobs=n_jobs
    )
    labels = db.fit_predict(coords_rad)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = (labels == -1).sum()
    noise_ratio = noise_count / len(labels)

    print(f"[DBSCAN] Clustering finished: {n_clusters} clusters found | Noise points: {noise_count:,} ({noise_ratio:.2%})")
    return labels

def grid_search_dbscan(
    coords_rad: np.ndarray,
    eps_km_list: List[float] = [0.10, 0.15, 0.20, 0.25, 0.30],
    min_samples_list: List[int] = [15, 25, 35, 50],
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Performs grid search across parameter candidates to evaluate cluster formation.

    Returns:
    --------
    pd.DataFrame:
        Table of parameter performance metrics.
    """
    results = []
    total_points = len(coords_rad)

    print(f"[DBSCAN GridSearch] Evaluating {len(eps_km_list) * len(min_samples_list)} parameter combinations...")
    for eps_km in eps_km_list:
        for min_samples in min_samples_list:
            eps_rad = km_to_radians(eps_km)
            db = DBSCAN(
                eps=eps_rad,
                min_samples=min_samples,
                metric="haversine",
                algorithm="ball_tree",
                n_jobs=n_jobs
            )
            labels = db.fit_predict(coords_rad)

            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
            noise_count = (labels == -1).sum()
            noise_ratio = noise_count / total_points

            if n_clusters > 0:
                cluster_sizes = [np.sum(labels == c) for c in unique_labels if c != -1]
                max_size = max(cluster_sizes)
                avg_size = np.mean(cluster_sizes)
            else:
                max_size = 0
                avg_size = 0

            results.append({
                "eps_km": eps_km,
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "noise_count": noise_count,
                "noise_ratio": round(noise_ratio * 100, 2),
                "max_cluster_size": max_size,
                "avg_cluster_size": round(avg_size, 1)
            })

    results_df = pd.DataFrame(results)
    return results_df
