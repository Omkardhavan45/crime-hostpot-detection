"""
Visualization Module for Crime Hotspot Analysis
Generates exploratory plots, parameter tuning curves, cluster diagnostics, and interactive Folium maps.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend to prevent GUI thread conflicts
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap, MeasureControl

# Style configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15
})

def plot_k_distance(
    k_distances_km: np.ndarray,
    chosen_eps_km: float = 0.20,
    output_path: str = "outputs/k_distance_plot.png"
):
    """
    Plots the sorted k-nearest neighbor distance curve to visualize the elbow point for epsilon selection.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    x = np.arange(len(k_distances_km))
    ax.plot(x, k_distances_km, color="#1f77b4", linewidth=2.5, label="k-Distance Curve (Haversine)")

    if chosen_eps_km is not None:
        ax.axhline(y=chosen_eps_km, color="#d62728", linestyle="--", linewidth=2,
                   label=f"Selected $\\epsilon$ = {chosen_eps_km:.2f} km ({chosen_eps_km*1000:.0f} m)")
        # Find index closest to chosen_eps_km
        elbow_idx = np.searchsorted(k_distances_km, chosen_eps_km)
        if elbow_idx < len(x):
            ax.scatter([elbow_idx], [chosen_eps_km], color="#d62728", s=100, zorder=5, label="Chosen Threshold Elbow")

    ax.set_title("k-Distance Graph for Optimal DBSCAN $\\epsilon$ Selection", fontweight="bold", pad=15)
    ax.set_xlabel("Points Sorted by Distance to k-th Nearest Neighbor", fontweight="bold")
    ax.set_ylabel("Distance to k-th Neighbor (km)", fontweight="bold")
    ax.set_ylim(0, min(1.5, max(k_distances_km) * 1.05))
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved k-distance plot to: {output_path}")

def plot_crime_type_distribution(
    df: pd.DataFrame,
    top_n: int = 12,
    output_path: str = "outputs/crime_type_distribution.png"
):
    """
    Plots horizontal bar chart of the most frequent primary crime types.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    counts = df["Primary Type"].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    palette = sns.color_palette("viridis", len(counts))
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=palette[::-1], edgecolor="black", alpha=0.85)

    total_crimes = len(df)
    for bar in bars:
        w = bar.get_width()
        pct = (w / total_crimes) * 100
        ax.text(w + total_crimes * 0.005, bar.get_y() + bar.get_height()/2,
                f"{w:,.0f} ({pct:.1f}%)", va="center", ha="left", fontsize=9, fontweight="bold")

    ax.set_title(f"Top {top_n} Crime Categories in Chicago (2016)", fontweight="bold", pad=15)
    ax.set_xlabel("Number of Incidents", fontweight="bold")
    ax.set_ylabel("Crime Classification", fontweight="bold")
    ax.set_xlim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved crime distribution plot to: {output_path}")

def plot_temporal_distribution(
    df: pd.DataFrame,
    output_path: str = "outputs/temporal_distribution.png"
):
    """
    Plots temporal dynamics: Hourly distribution and Day-of-Week frequency.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # 1. Hourly distribution
    hourly_counts = df["hour"].value_counts().sort_index()
    hours = np.arange(24)
    counts_hourly = [hourly_counts.get(h, 0) for h in hours]
    colors = ["#2b5c8f" if 6 <= h < 18 else "#1a2c42" for h in hours]

    ax1.bar(hours, counts_hourly, color=colors, edgecolor="black", alpha=0.85)
    ax1.set_title("Crime Incidents by Hour of Day", fontweight="bold", pad=12)
    ax1.set_xlabel("Hour of Day (24-Hour Military Format)", fontweight="bold")
    ax1.set_ylabel("Incident Frequency", fontweight="bold")
    ax1.set_xticks(hours[::2])
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Highlight peak hour
    peak_h = int(hourly_counts.idxmax())
    peak_val = hourly_counts.max()
    ax1.annotate(f"Peak: {peak_h:02d}:00\n({peak_val:,} crimes)",
                 xy=(peak_h, peak_val), xytext=(peak_h + 1.5, peak_val * 0.92),
                 arrowprops=dict(facecolor="#d62728", shrink=0.05, width=1.5, headwidth=6),
                 fontweight="bold", color="#d62728")

    # 2. Day of Week distribution
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_counts = df["day_name"].value_counts().reindex(day_order)

    dow_palette = ["#4575b4"] * 4 + ["#d73027", "#f46d43", "#fdae61"]
    bars2 = ax2.bar(day_order, dow_counts.values, color=dow_palette, edgecolor="black", alpha=0.85)
    ax2.set_title("Crime Incidents by Day of Week", fontweight="bold", pad=12)
    ax2.set_xlabel("Day of the Week", fontweight="bold")
    ax2.set_ylabel("Incident Frequency", fontweight="bold")
    ax2.set_xticks(range(len(day_order)))
    ax2.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved temporal distribution plot to: {output_path}")

def plot_cluster_distribution(
    clusters_df: pd.DataFrame,
    output_path: str = "outputs/cluster_distribution.png"
):
    """
    Plots cluster size distribution and dispersion metrics.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Histogram of cluster sizes
    sizes = clusters_df["incident_count"]
    ax1.hist(sizes, bins=25, color="#2ca02c", edgecolor="black", alpha=0.75)
    ax1.axvline(sizes.median(), color="#d62728", linestyle="--", linewidth=2,
                label=f"Median Size: {sizes.median():.0f}")
    ax1.axvline(sizes.mean(), color="#ff7f0e", linestyle=":", linewidth=2,
                label=f"Mean Size: {sizes.mean():.1f}")
    ax1.set_title("Distribution of Hotspot Sizes (Incidents per Cluster)", fontweight="bold", pad=12)
    ax1.set_xlabel("Incidents in Hotspot", fontweight="bold")
    ax1.set_ylabel("Number of Hotspots", fontweight="bold")
    ax1.legend(frameon=True)

    # Radius vs Incident count scatter
    scatter = ax2.scatter(clusters_df["radius_km"], clusters_df["incident_count"],
                          c=clusters_df["arrest_rate"], cmap="coolwarm", s=80, edgecolors="black", alpha=0.8)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label("Arrest Rate (%)", fontweight="bold")
    ax2.set_title("Hotspot Spread (Radius) vs Incident Density", fontweight="bold", pad=12)
    ax2.set_xlabel("Hotspot Radius (90th percentile, km)", fontweight="bold")
    ax2.set_ylabel("Total Incident Count", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved cluster distribution plot to: {output_path}")

def plot_top_hotspots(
    clusters_df: pd.DataFrame,
    top_n: int = 10,
    output_path: str = "outputs/top10_hotspots_chart.png"
):
    """
    Plots ranked horizontal bar chart of the highest density crime hotspots.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    top_df = clusters_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    colors = plt.cm.plasma(np.linspace(0.2, 0.85, top_n))

    labels = [f"Rank #{row.risk_rank}: Cluster {row.cluster_id}\n({row.dominant_crime})" for _, row in top_df.iterrows()]
    bars = ax.barh(labels, top_df["incident_count"], color=colors, edgecolor="black", alpha=0.85)

    for bar, (_, row) in zip(bars, top_df.iterrows()):
        w = bar.get_width()
        text = f" {w:,} crimes | Peak: {row.peak_hour:02d}:00 | Arrest: {row.arrest_rate:.1f}%"
        ax.text(w, bar.get_y() + bar.get_height()/2, text, va="center", ha="left", fontsize=9.5, fontweight="bold")

    ax.set_title(f"Top {top_n} High-Risk Crime Hotspots in Chicago (2016)", fontweight="bold", pad=15)
    ax.set_xlabel("Total Crimes Recorded in Hotspot", fontweight="bold")
    ax.set_ylabel("Hotspot Designation & Dominant Type", fontweight="bold")
    ax.set_xlim(0, top_df["incident_count"].max() * 1.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved top hotspots chart to: {output_path}")

def generate_hotspot_map(
    df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    output_path: str = "outputs/hotspots_map.html",
    max_noise_sample: int = 1000
) -> folium.Map:
    """
    Generates an interactive Folium geospatial map with:
    - Multiple basemap tiles
    - Crime density heatmap layer
    - Centroid markers with rich HTML popup profiles
    - Boundary circles showing physical cluster radius
    - Sampled noise points layer
    - Measure tool and layer controls
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Center on Chicago coordinates
    chicago_center = [41.8781, -87.6298]
    m = folium.Map(
        location=chicago_center,
        zoom_start=11,
        tiles="OpenStreetMap",
        control_scale=True
    )

    # 1. HeatMap Layer
    heat_data = df[["Latitude", "Longitude"]].dropna().to_numpy()
    if len(heat_data) > 30000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(heat_data), size=30000, replace=False)
        heat_data = heat_data[idx]

    heat_layer = folium.FeatureGroup(name="Overall Crime Heatmap", show=False)
    HeatMap(heat_data.tolist(), radius=12, blur=18, min_opacity=0.3).add_to(heat_layer)
    heat_layer.add_to(m)

    # 2. Sampled Noise Points Layer
    noise_df = df[df["cluster"] == -1]
    if len(noise_df) > 0:
        noise_layer = folium.FeatureGroup(name=f"Noise Points (Sampled {min(len(noise_df), max_noise_sample):,})", show=False)
        sample_noise = noise_df.sample(n=min(len(noise_df), max_noise_sample), random_state=42)
        for _, row in sample_noise.iterrows():
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=2,
                color="#888888",
                fill=True,
                fill_color="#aaaaaa",
                fill_opacity=0.4,
                popup=f"Noise Crime: {row['Primary Type']} ({row.get('hour', 'N/A')}:00)"
            ).add_to(noise_layer)
        noise_layer.add_to(m)

    # 3. Hotspot Centroids & Spread Circles
    hotspots_layer = folium.FeatureGroup(name="Detected Hotspot Centroids & Radius", show=True)

    # Palette mapping for dominant crimes
    crime_colors = {
        "THEFT": "#e41a1c",
        "BATTERY": "#377eb8",
        "CRIMINAL DAMAGE": "#4daf4a",
        "ASSAULT": "#984ea3",
        "NARCOTICS": "#ff7f00",
        "BURGLARY": "#ffff33",
        "ROBBERY": "#a65628",
        "MOTOR VEHICLE THEFT": "#f781bf"
    }

    for _, row in clusters_df.iterrows():
        color = crime_colors.get(row.dominant_crime, "#999999")
        radius_m = max(float(row.radius_km * 1000), 50.0)

        # Spread Circle
        folium.Circle(
            location=[row.centroid_lat, row.centroid_lon],
            radius=radius_m,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.25,
            tooltip=f"Hotspot #{row.risk_rank} | Crimes: {row.incident_count:,} | Radius: {row.radius_km:.2f} km"
        ).add_to(hotspots_layer)

        # Popup card HTML
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; width: 260px; font-size: 13px; line-height: 1.4;">
            <div style="background-color: {color}; color: white; padding: 8px 12px; border-radius: 4px 4px 0 0; font-weight: bold; font-size: 14px;">
                Risk Rank #{row.risk_rank} (Cluster {row.cluster_id})
            </div>
            <div style="padding: 10px; border: 1px solid #ddd; border-top: none; background-color: #fafafa; border-radius: 0 0 4px 4px;">
                <p style="margin: 4px 0;"><b>Total Incidents:</b> <span style="color: #d62728; font-weight: bold;">{row.incident_count:,}</span></p>
                <p style="margin: 4px 0;"><b>Dominant Crime:</b> {row.dominant_crime} ({row.dominant_crime_pct}%)</p>
                <p style="margin: 4px 0;"><b>Peak Patrol Hour:</b> {row.peak_hour:02d}:00 ({row.peak_time_bucket})</p>
                <p style="margin: 4px 0;"><b>Arrest Rate:</b> {row.arrest_rate}%</p>
                <p style="margin: 4px 0;"><b>Effective Radius:</b> {row.radius_km:.3f} km ({radius_m:.0f} m)</p>
                <p style="margin: 4px 0;"><b>Primary Venue:</b> {row.dominant_location}</p>
                <p style="margin: 4px 0; color: #666; font-size: 11px;">Centroid: {row.centroid_lat:.4f}, {row.centroid_lon:.4f}</p>
            </div>
        </div>
        """

        # Centroid Marker
        folium.CircleMarker(
            location=[row.centroid_lat, row.centroid_lon],
            radius=6 + min(int(row.incident_count / 150), 12),
            color="#222222",
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Rank #{row.risk_rank}: {row.dominant_crime} ({row.incident_count:,} crimes)"
        ).add_to(hotspots_layer)

    hotspots_layer.add_to(m)

    # Add measurement tool & layer switch
    MeasureControl(position="bottomleft").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    m.save(output_path)
    print(f"[Visualization] Saved interactive Folium map to: {output_path}")
    return m
