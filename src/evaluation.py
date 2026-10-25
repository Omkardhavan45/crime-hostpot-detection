"""
Evaluation and Validation Module for Spatial Clustering
Computes intrinsic clustering quality metrics (Silhouette, Davies-Bouldin),
density assessments, noise ratios, and operational validation statistics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Dict, Any, Optional

def evaluate_clustering(
    coords_rad: np.ndarray,
    labels: np.ndarray,
    sample_size: int = 10000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Evaluates DBSCAN clustering output using haversine silhouette score,
    noise ratio, and cluster concentration metrics.

    Parameters:
    -----------
    coords_rad : np.ndarray
        (N, 2) array of coordinates in radians.
    labels : np.ndarray
        Cluster labels from DBSCAN (-1 represents noise).
    sample_size : int
        Sample size for silhouette calculation (to optimize CPU time).
    random_state : int
        Seed for sampling.

    Returns:
    --------
    Dict[str, Any]:
        Evaluation metrics dictionary.
    """
    total_points = len(labels)
    noise_mask = (labels == -1)
    noise_count = int(np.sum(noise_mask))
    noise_ratio = float(noise_count / total_points)

    clustered_indices = np.where(~noise_mask)[0]
    n_clustered = len(clustered_indices)

    unique_clusters = [c for c in np.unique(labels) if c != -1]
    n_clusters = len(unique_clusters)

    metrics = {
        "total_incidents": total_points,
        "clustered_incidents": n_clustered,
        "noise_incidents": noise_count,
        "noise_ratio_pct": round(noise_ratio * 100, 2),
        "cluster_count": n_clusters,
        "silhouette_score": None,
        "davies_bouldin_index": None
    }

    if n_clusters < 2:
        print("[Evaluation] Less than 2 clusters found; skipping silhouette score calculation.")
        return metrics

    # Sample clustered points for silhouette evaluation
    clustered_coords = coords_rad[clustered_indices]
    clustered_labels = labels[clustered_indices]

    if len(clustered_labels) > sample_size:
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(len(clustered_labels), size=sample_size, replace=False)
        eval_coords = clustered_coords[sample_idx]
        eval_labels = clustered_labels[sample_idx]
    else:
        eval_coords = clustered_coords
        eval_labels = clustered_labels

    # Check if sampled points contain at least 2 distinct clusters
    if len(np.unique(eval_labels)) >= 2:
        try:
            print(f"[Evaluation] Computing Silhouette Score on {len(eval_labels):,} sample points (haversine metric)...")
            sil = silhouette_score(eval_coords, eval_labels, metric="haversine")
            metrics["silhouette_score"] = round(float(sil), 4)
            print(f"[Evaluation] Haversine Silhouette Score: {metrics['silhouette_score']}")
        except Exception as e:
            print(f"[Evaluation] Silhouette calculation encountered: {e}")

        try:
            db_score = davies_bouldin_score(eval_coords, eval_labels)
            metrics["davies_bouldin_index"] = round(float(db_score), 4)
            print(f"[Evaluation] Davies-Bouldin Index: {metrics['davies_bouldin_index']}")
        except Exception as e:
            print(f"[Evaluation] Davies-Bouldin calculation encountered: {e}")

    return metrics
