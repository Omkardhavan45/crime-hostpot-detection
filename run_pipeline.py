"""
Main Pipeline Runner for Crime Hotspot Detection using DBSCAN
Executes data extraction, cleaning, parameter tuning, DBSCAN clustering,
hotspot profiling, interactive map generation, and model evaluation.
"""

import os
import sys
import argparse
import time
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_raw_crime_data
from src.preprocessing import clean_and_filter_data, to_radians
from src.dbscan_model import compute_k_distance, run_dbscan, grid_search_dbscan
from src.cluster_analysis import analyze_clusters
from src.visualization import (
    plot_k_distance,
    plot_crime_type_distribution,
    plot_temporal_distribution,
    plot_cluster_distribution,
    plot_top_hotspots,
    generate_hotspot_map
)
from src.evaluation import evaluate_clustering

def main():
    parser = argparse.ArgumentParser(description="Crime Hotspot Detection Pipeline (Chicago 2016)")
    parser.add_argument("--data", type=str, default="crimes.csv", help="Path to raw crimes.csv")
    parser.add_argument("--cached", type=str, default="data/cleaned_crimes_2016.csv", help="Path to cached cleaned CSV")
    parser.add_argument("--scope", type=str, default="sample50k", choices=["sample50k", "theft", "all"],
                        help="Data scope: 'sample50k' (50,000 random sample), 'theft' (all thefts), or 'all' (full 2016 dataset)")
    parser.add_argument("--eps_km", type=float, default=0.20, help="Epsilon radius in kilometers (default: 0.20 km = 200m)")
    parser.add_argument("--min_samples", type=int, default=25, help="Minimum core samples for DBSCAN (default: 25)")
    parser.add_argument("--skip_grid", action="store_true", help="Skip hyperparameter grid search")
    args = parser.parse_args()

    start_total_time = time.time()
    print("=" * 70)
    print("        CRIME HOTSPOT DETECTION USING DBSCAN (CHICAGO 2016)        ")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. DATA LOADING & EXTRACTION
    # -------------------------------------------------------------
    print("\n>>> STEP 1: LOADING & FILTERING DATA")
    if os.path.exists(args.cached):
        print(f"Loading cleaned dataset directly from cache: {args.cached}")
        df_2016 = pd.read_csv(args.cached)
    else:
        print(f"Extracting 2016 records from raw '{args.data}'...")
        raw_df = load_raw_crime_data(filepath=args.data, year=2016, cached_path=None)
        df_2016 = clean_and_filter_data(raw_df, save_cleaned_path=args.cached)

    print(f"Dataset ready with {len(df_2016):,} clean 2016 Chicago crime incidents.")

    # -------------------------------------------------------------
    # 2. EXPLORATORY DATA ANALYSIS (EDA) CHARTS
    # -------------------------------------------------------------
    print("\n>>> STEP 2: GENERATING EXPLORATORY DATA VISUALIZATIONS")
    plot_crime_type_distribution(df_2016, top_n=12, output_path="outputs/crime_type_distribution.png")
    plot_temporal_distribution(df_2016, output_path="outputs/temporal_distribution.png")

    # -------------------------------------------------------------
    # 3. SCOPE SELECTION (Phase 4 of Blueprint)
    # -------------------------------------------------------------
    print(f"\n>>> STEP 3: APPLYING MODEL SCOPE ({args.scope.upper()})")
    if args.scope == "sample50k":
        if len(df_2016) > 50000:
            df_model = df_2016.sample(n=50000, random_state=42).reset_index(drop=True)
            print(f"Selected representative random sample of {len(df_model):,} records (random_state=42).")
        else:
            df_model = df_2016.copy()
    elif args.scope == "theft":
        df_model = df_2016[df_2016["Primary Type"] == "THEFT"].copy().reset_index(drop=True)
        print(f"Filtered for primary crime category 'THEFT': {len(df_model):,} incidents.")
    else:
        df_model = df_2016.copy().reset_index(drop=True)
        print(f"Using full 2016 dataset: {len(df_model):,} incidents.")

    # -------------------------------------------------------------
    # 4. PARAMETER SELECTION & ELBOW ANALYSIS (Phase 5)
    # -------------------------------------------------------------
    print("\n>>> STEP 4: PARAMETER SELECTION & K-DISTANCE COMPUTATION")
    coords_rad = to_radians(df_model)

    k_distances_km, _ = compute_k_distance(coords_rad, k=args.min_samples, sample_size=20000, random_state=42)
    plot_k_distance(k_distances_km, chosen_eps_km=args.eps_km, output_path="outputs/k_distance_plot.png")

    if not args.skip_grid:
        print("\nEvaluating Parameter Grid for Sensitivity Analysis...")
        grid_df = grid_search_dbscan(
            coords_rad,
            eps_km_list=[0.15, 0.20, 0.25],
            min_samples_list=[15, 25, 35]
        )
        print("\nGrid Search Evaluation Summary:")
        print(grid_df.to_string(index=False))

    # -------------------------------------------------------------
    # 5. EXECUTE DBSCAN (Phase 6)
    # -------------------------------------------------------------
    print(f"\n>>> STEP 5: RUNNING DBSCAN (eps={args.eps_km} km, min_samples={args.min_samples})")
    labels = run_dbscan(coords_rad, eps_km=args.eps_km, min_samples=args.min_samples)
    df_model["cluster"] = labels

    # -------------------------------------------------------------
    # 6. CLUSTER ANALYSIS & RISK PROFILING (Phase 7)
    # -------------------------------------------------------------
    print("\n>>> STEP 6: COMPUTING CLUSTER METRICS & RISK RANKINGS")
    clusters_df, summary_stats = analyze_clusters(df_model, cluster_col="cluster")

    # Export top hotspots to CSV
    top_csv_path = "outputs/top_hotspots.csv"
    clusters_df.to_csv(top_csv_path, index=False)
    print(f"Exported hotspot rankings to: {top_csv_path}")

    # Display Top 10 Hotspots
    print("\n" + "=" * 95)
    print("                            TOP 10 CRIME HOTSPOTS IN CHICAGO                          ")
    print("=" * 95)
    display_cols = ["risk_rank", "cluster_id", "incident_count", "dominant_crime", "peak_hour", "arrest_rate", "radius_km", "dominant_location"]
    print(clusters_df[display_cols].head(10).to_string(index=False))
    print("=" * 95)

    # -------------------------------------------------------------
    # 7. VISUALIZATIONS & FOLIUM MAP (Phase 8)
    # -------------------------------------------------------------
    print("\n>>> STEP 7: GENERATING DIAGNOSTIC CHARTS & INTERACTIVE FOLIUM MAP")
    plot_cluster_distribution(clusters_df, output_path="outputs/cluster_distribution.png")
    plot_top_hotspots(clusters_df, top_n=10, output_path="outputs/top10_hotspots_chart.png")
    generate_hotspot_map(df_model, clusters_df, output_path="outputs/hotspots_map.html", max_noise_sample=1200)

    # -------------------------------------------------------------
    # 8. EVALUATION (Phase 9)
    # -------------------------------------------------------------
    print("\n>>> STEP 8: EVALUATION & INTRINSIC QUALITY METRICS")
    eval_metrics = evaluate_clustering(coords_rad, labels, sample_size=8000, random_state=42)
    print("\nClustering Performance Summary:")
    for k, v in eval_metrics.items():
        print(f"  - {k}: {v}")

    elapsed = time.time() - start_total_time
    print(f"\nPipeline successfully completed in {elapsed:.2f} seconds.")
    print("All artifacts generated in 'outputs/' directory.")
    print("=" * 70)

if __name__ == "__main__":
    main()
