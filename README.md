# Crime Hotspot Detection using DBSCAN
> **Geospatial Machine Learning & Patrol Route Optimization in Chicago**

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Machine Learning](https://img.shields.io/badge/ML-DBSCAN%20%7C%20Haversine-orange.svg)](https://scikit-learn.org)
[![Geospatial](https://img.shields.io/badge/GIS-Folium%20%7C%20BallTree-green.svg)](https://python-visualization.github.io/folium/)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)]()

---

## 📌 Project Overview

Urban crime incidents concentrate in specific geographic micro-locations known as **crime hotspots**. Efficient law enforcement requires accurately detecting these high-risk areas to optimize proactive patrols and deter offenses before they occur.

Traditional clustering methods like **k-Means** suffer from major limitations in geospatial data:
1. They require specifying the number of clusters ($k$) in advance.
2. They enforce artificial spherical cluster boundaries.
3. They are sensitive to noise and force isolated background incidents into clusters.

This project implements an unsupervised geospatial machine learning pipeline using **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** with the great-circle **Haversine metric** to:
- Discover clusters of arbitrary shapes without pre-specifying $k$.
- Filter out isolated background incidents as noise.
- Profile each hotspot with operational intelligence (centroid, effective radius, dominant crime type, peak patrol hours, and arrest yield).
- Deliver interactive mapping interfaces (`outputs/hotspots_map.html`) and dispatch-ready data tables (`outputs/top_hotspots.csv`).

---

## 📁 Repository Structure

```
crime-hotspot-dbscan/
├── data/
│   └── cleaned_crimes_2016.csv          # Preprocessed Chicago 2016 crime dataset (30.3 MB)
├── notebooks/
│   └── hotspot_analysis.ipynb           # Complete 10-phase Jupyter analysis notebook
├── outputs/
│   ├── hotspots_map.html                # Interactive Folium map with HeatMap & Hotspot Popups
│   ├── top_hotspots.csv                 # Ranked CSV registry of all detected hotspots
│   ├── k_distance_plot.png              # k-distance elbow curve for epsilon selection
│   ├── crime_type_distribution.png      # Top crime categories breakdown
│   ├── temporal_distribution.png        # Hourly and day-of-week crime distribution
│   ├── cluster_distribution.png         # Hotspot size distribution & radius diagnostics
│   └── top10_hotspots_chart.png         # Top 10 high-risk hotspots ranked chart
├── report/
│   ├── project_report.docx              # Comprehensive Word report with embedded figures
│   └── project_report.md                # Markdown version of technical report
├── src/
│   ├── __init__.py                      # Package exports
│   ├── data_loader.py                   # Chunked CSV ingestion & year filtering
│   ├── preprocessing.py                 # Coordinate validation, bounds, and date parsing
│   ├── dbscan_model.py                  # Haversine DBSCAN & hyperparameter grid search
│   ├── cluster_analysis.py              # Hotspot centroids, radii, and operational profiling
│   ├── visualization.py                 # Matplotlib diagnostic plots & Folium map generator
│   └── evaluation.py                    # Haversine silhouette & Davies-Bouldin metrics
├── run_pipeline.py                      # End-to-end command-line runner
├── requirements.txt                     # Python package dependencies
├── project_blueprint.md                 # Original mini-project blueprint specification
└── README.md                            # Comprehensive project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone or Navigate to Directory
```bash
cd "c:/Users/odhav/Downloads/archive (2)"
```

### 2. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

*Required libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `folium`, `seaborn`, `python-docx`.*

---

## 🚀 How to Run the Project

### Option A: Run the End-to-End Pipeline (CLI)
Execute the complete pipeline with default parameters (2016 dataset, sample of 50,000 incidents per blueprint Option B):
```bash
python run_pipeline.py
```

#### Pipeline Options & Flags:
```bash
# Run with a custom epsilon radius (e.g., 250 meters) and min_samples
python run_pipeline.py --eps_km 0.25 --min_samples 35

# Run specifically on all THEFT incidents
python run_pipeline.py --scope theft

# Run on the complete 2016 dataset (267k points)
python run_pipeline.py --scope all --skip_grid
```

### Option B: Interactive Jupyter Notebook
Open the notebook in VS Code or JupyterLab:
```bash
jupyter notebook notebooks/hotspot_analysis.ipynb
```

---

## 🔬 Mathematical Methodology

### 1. Haversine Distance
Because latitude and longitude represent coordinates on a spherical surface, Euclidean distance produces geographic distortion. The great-circle distance between two points $(\phi_1, \lambda_1)$ and $(\phi_2, \lambda_2)$ in radians is calculated via:

$$d = 2 R \arcsin \left( \sqrt{\sin^2\left(\frac{\phi_2 - \phi_1}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\lambda_2 - \lambda_1}{2}\right)} \right)$$

where $R = 6371.0088 \text{ km}$ is Earth's mean radius.

### 2. Epsilon Conversion
To search within a neighborhood radius of $\epsilon = 0.20 \text{ km}$ (200 meters):
$$\epsilon_{\text{rad}} = \frac{\epsilon_{\text{km}}}{6371.0088} = \frac{0.20}{6371.0088} \approx 0.000031388 \text{ radians}$$

### 3. DBSCAN Architecture
- **Core Point:** A point $p$ with at least $MinPts$ incidents within Haversine distance $\epsilon$.
- **Border Point:** A point within distance $\epsilon$ of a core point but with fewer than $MinPts$ neighbors.
- **Noise Point:** Any point that is neither a core point nor reachable from a core point (labeled `-1`).

---

## 📊 Key Results & Insights

### 1. Clustering Performance Summary

| Metric | Result | Interpretation |
|---|---|---|
| **Total Incidents Analyzed** | 50,000 | Representative sample of 2016 incidents |
| **Hotspots Detected** | **204** | Discrete high-density crime corridors |
| **Clustered Crimes** | 30,043 (60.09%) | High concentration in actionable zones |
| **Noise Incidents Filtered** | 19,957 (39.91%) | Random, non-repeating crimes isolated |
| **Haversine Silhouette Score** | **0.1223** | Positive separation on spherical geometry |
| **Davies-Bouldin Index** | **0.6414** | Strong compactness relative to cluster separation |
| **Largest Hotspot** | 5,530 crimes | Primary South/West commercial & residential corridor |

### 2. Top 10 High-Risk Crime Hotspots in Chicago

| Risk Rank | Cluster ID | Incident Count | Dominant Crime | Share (%) | Peak Hour | Patrol Shift | Arrest Rate | Effective Radius | Dominant Location |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **#1** | 2 | 5,530 | BATTERY | 27.5% | 19:00 | Evening (18-24) | 32.9% | 3.32 km | STREET |
| **#2** | 0 | 3,719 | THEFT | 46.8% | 12:00 | Afternoon (12-18) | 18.2% | 2.18 km | STREET / RETAIL |
| **#3** | 10 | 1,137 | BATTERY | 29.8% | 11:00 | Morning (06-12) | 28.9% | 1.35 km | STREET |
| **#4** | 5 | 829 | BATTERY | 28.1% | 15:00 | Afternoon (12-18) | 19.1% | 0.98 km | APARTMENT |
| **#5** | 8 | 795 | THEFT | 38.6% | 00:00 | Night (00-06) | 16.9% | 1.22 km | STREET |
| **#6** | 26 | 707 | BATTERY | 26.9% | 12:00 | Afternoon (12-18) | 18.1% | 1.26 km | APARTMENT |
| **#7** | 24 | 703 | THEFT | 41.2% | 21:00 | Evening (18-24) | 10.1% | 1.31 km | STREET |
| **#8** | 6 | 649 | BATTERY | 30.5% | 22:00 | Evening (18-24) | 29.1% | 1.38 km | STREET |
| **#9** | 14 | 636 | BATTERY | 27.4% | 18:00 | Evening (18-24) | 21.5% | 1.02 km | APARTMENT |
| **#10** | 12 | 560 | BATTERY | 31.2% | 16:00 | Afternoon (12-18) | 25.0% | 0.76 km | APARTMENT |

---

## 🗺️ Interactive Folium Geospatial Map

The interactive map is generated at `outputs/hotspots_map.html` and includes:
- **Base Tiles:** Clean OpenStreetMap basemaps with scale control.
- **Overall Crime HeatMap Layer:** Toggleable continuous density heatmap.
- **Hotspot Centroid Markers:** Circle markers scaled by crime volume with interactive popup cards displaying:
  - Hotspot Risk Rank & Cluster ID
  - Incident Count
  - Dominant Crime & Percentage
  - Peak Patrol Hour & Time Bucket
  - Arrest Rate (%)
  - Effective Radius (km & meters)
  - Dominant Physical Location
- **Spatial Spread Boundaries:** Circular boundary overlays matching the 90th percentile radius of each cluster, color-coded by dominant crime.
- **Sampled Noise Layer:** Discrete grey markers representing isolated background crimes for clear visual contrast between hotspots and noise.
- **Interactive Measurement Tool:** Measure distance directly on the map.

---

## 👮 Operational Law Enforcement Recommendations

1. **Shift Staggering:**
   - Over 65% of violent incidents in battery hotspots occur between **16:00 and 22:00**. Shift changeovers should be scheduled outside this peak window to ensure maximum patrol presence.
2. **Crime-Specific Patrol Tactics:**
   - **Theft Corridors (e.g., Hotspot #2 - Downtown Loop):** Deploy plainclothes units and coordinate with commercial loss prevention teams during midday shopping hours (11:00-14:00).
   - **Battery Hotspots (e.g., Hotspot #1, #3):** Deploy high-visibility marked patrol cruisers and foot patrols during evening and late-night hours (18:00-02:00).
3. **Computer-Aided Dispatch (CAD) Geofencing:**
   - Export cluster centroids and radii directly into dispatch systems to automatically alert responding officers when entering known high-density corridors.

---

## ⚠️ Limitations & Future Work

- **Uniform Density Threshold:** Standard DBSCAN applies a single global $\epsilon$, which can under-cluster lower-density suburban zones. Future extensions should incorporate **HDBSCAN** for multi-density clustering.
- **Block-Level Coordinates:** Anonymized block-level reporting creates duplicate coordinates at intersections.
- **Spatio-Temporal Clustering:** Future iterations can evaluate **ST-DBSCAN** to incorporate time intervals into distance calculations.

---

## 📄 License & Attribution

- **Dataset:** Chicago Police Department Crime Dataset (`crimes.csv`).
- **Blueprint:** `project_blueprint.md`.
- **Implementation:** Unsupervised Machine Learning with Scikit-Learn & Folium.
