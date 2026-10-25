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
│   └── cleaned_crimes_2016.csv          # Preprocessed dataset cache (auto-generated from crimes.csv)
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


### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

*Required libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `folium`, `seaborn`, `python-docx`.*

### 2. Dataset Download
Download the Chicago Crime dataset from Kaggle:
👉 **[Kaggle: Crimes in Chicago Dataset](https://www.kaggle.com/datasets/currie32/crimes-in-chicago)**

Download the dataset and place the CSV file in the root project folder named exactly as:
```
crimes.csv
```
> [!NOTE]
> When you run the pipeline for the first time, it automatically reads `crimes.csv` in chunks, extracts records for the target year (2016), performs coordinate boundary filtering and feature engineering, and caches the cleaned result to `data/cleaned_crimes_2016.csv`.

---

## 🚀 Pipeline Execution Guide

The core execution engine is [`run_pipeline.py`](run_pipeline.py), which handles data ingestion, spatial preprocessing, elbow evaluation, DBSCAN clustering, risk profiling, evaluation metric computation, and artifact generation.

### 1. Basic Execution Commands

```bash
# Default run: sample50k mode, eps=0.20 km (200m), min_samples=25
python run_pipeline.py
```

### 2. Command-Line Options Reference

| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--scope` | `str` | `sample50k` | Data subset: `'sample50k'` (50k sample, blueprint Option B), `'theft'` (all thefts), or `'all'` (full dataset). |
| `--eps_km` | `float` | `0.20` | DBSCAN neighborhood radius in kilometers (e.g. `0.20` = 200 meters, `0.15` = 150 meters). |
| `--min_samples`| `int` | `25` | Minimum core incident count required to anchor a dense hotspot cluster. |
| `--skip_grid` | `flag` | `False` | Skip sensitivity grid search across parameter combinations to speed up execution. |
| `--data` | `str` | `crimes.csv` | Path to the raw multi-year Chicago crime records CSV. |
| `--cached` | `str` | `data/cleaned_crimes_2016.csv` | Path to cached preprocessed CSV (auto-loaded if present; auto-created from `--data` if missing). |

### 3. Practical Execution Scenarios

```bash
# Scenario A: Fast execution with fixed parameters (skipping grid search)
python run_pipeline.py --skip_grid

# Scenario B: High-density walking patrol radius (150m, 15 incidents)
python run_pipeline.py --eps_km 0.15 --min_samples 15

# Scenario C: Broad vehicular patrol radius (300m, 40 incidents)
python run_pipeline.py --eps_km 0.30 --min_samples 40

# Scenario D: Crime-specific focus (Detecting theft & retail corridors)
python run_pipeline.py --scope theft --eps_km 0.18 --min_samples 20

# Scenario E: Full municipal dataset run (All incidents)
python run_pipeline.py --scope all --skip_grid
```

---

### 🔄 When Do You Need to Run the Pipeline?

You should run `python run_pipeline.py` whenever:

1. **Initial Setup or Fresh Clone:** Run once to generate all diagnostic charts, the CSV registry, and the interactive Folium map in `outputs/`.
2. **`outputs/` Files Are Missing or Deleted:** If any output charts, CSV, or HTML maps are deleted, running the pipeline will recreate the `outputs/` folder and regenerate all files.
3. **Hyperparameter Tuning:** Whenever you want to test different spatial radii (`--eps_km`) or density thresholds (`--min_samples`) to adapt to different policing strategies (e.g. foot patrols vs. squad car patrol zones).
4. **Scope or Crime Type Change:** When shifting focus to a specific crime category (such as `--scope theft`) or running on the complete dataset (`--scope all`).
5. **Data Refresh:** If a new raw dataset `crimes.csv` is provided, running the pipeline extracts, cleans, and updates the cache `data/cleaned_crimes_2016.csv`.
6. **Prior to Generating the Word Report:** Always ensure the pipeline has run so [`generate_report_docx.py`](generate_report_docx.py) can embed the latest figures and tables from `outputs/`.

---

### 📋 Complete Project Execution Lifecycle

```
               ┌────────────────────────────────────────────────────────┐
               │              Step 1: Execute ML Pipeline               │
               │               python run_pipeline.py                   │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   ▼                                             ▼
       ┌────────────────────────┐                   ┌────────────────────────┐
       │   data/ (Input Cache)  │                   │   outputs/ (Artifacts) │
       │ cleaned_crimes_2016.csv│                   │  • 5 PNG Visualizations│
       └────────────────────────┘                   │  • top_hotspots.csv    │
                   │                                │  • hotspots_map.html   │
                   │                                └────────────┬───────────┘
                   │                                             │
                   │ Reads from outputs/                         │
                   ▼                                             ▼
┌──────────────────────────────────────┐     ┌──────────────────────────────────────┐
│     Step 2: Generate Word Report     │     │    Step 3: Rebuild Jupyter Notebook  │
│    python generate_report_docx.py    │     │       python create_notebook.py      │
│  Outputs: report/project_report.docx │     │ Outputs: notebooks/hotspot_analysis  │
└──────────────────────────────────────┘     └──────────────────────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                     Interactive Exploration in Jupyter                            │
│                 jupyter notebook notebooks/hotspot_analysis.ipynb                 │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📤 Output Artifacts & Persistence Guide

All artifacts produced by the clustering pipeline are saved in the `outputs/` folder:

### 1. Generated Output Files

| File | Format | Description & Usage |
| :--- | :---: | :--- |
| **`outputs/hotspots_map.html`** | HTML/Leaflet | **Interactive Geospatial Map**: Features OpenStreetMap base tiles, toggleable crime density heatmap, cluster centroid markers with rich HTML popup profiles, 90th-percentile cluster radius circles, and sampled noise points. Open directly in any web browser. |
| **`outputs/top_hotspots.csv`** | CSV | **Hotspot Risk Registry**: Ranked tabular data of all detected hotspots including risk rank, cluster ID, incident count, geographic centroid (latitude/longitude), radius (km), dominant crime type, peak operational hour, and arrest yield (%). |
| **`outputs/top10_hotspots_chart.png`** | PNG (300 DPI) | **Ranked Bar Chart**: Visual comparison of the top 10 highest-density crime hotspots displaying total incidents, dominant crime, peak hour, and arrest rates. |
| **`outputs/k_distance_plot.png`** | PNG (300 DPI) | **Parameter Tuning Curve**: Sorted k-nearest neighbor Haversine distance graph displaying the elbow inflection point used to empirically choose $\epsilon$. |
| **`outputs/crime_type_distribution.png`** | PNG (300 DPI) | **EDA Incident Frequency**: Horizontal bar chart showing frequency and percentage shares of top crime classifications. |
| **`outputs/temporal_distribution.png`** | PNG (300 DPI) | **Temporal Dynamics**: Dual-panel chart detailing hourly distribution (24-hour cycle highlighting peak hour) and day-of-week incident frequencies. |
| **`outputs/cluster_distribution.png`** | PNG (300 DPI) | **Cluster Diagnostic Plots**: Hotspot size distribution histogram (median and mean sizes) alongside scatter plot of spatial radius vs. incident density color-coded by arrest rate. |

### 2. Directory Persistence & Regeneration Matrix

| Directory / File | Type | Safe to Delete? | What Happens If Deleted? | How to Regenerate |
| :--- | :---: | :---: | :--- | :--- |
| **`outputs/*`** | Generated Artifacts | **YES** | Downstream Word report will lack embedded charts/tables until regenerated. | Run `python run_pipeline.py` |
| **`data/cleaned_crimes_2016.csv`** | Cleaned Input Data | **NO** (Unless `crimes.csv` exists) | Script and notebook will fail with `FileNotFoundError` unless raw `crimes.csv` is present. | Auto-generated from `crimes.csv` if raw file is present; otherwise restore from backup. |
| **`report/project_report.docx`** | Formatted Document | **YES** | Word deliverable is removed. | Run `python generate_report_docx.py` |
| **`report/project_report.md`** | Markdown Document | **NO** | Static Markdown documentation is removed. | Restore from Git/backup. |
| **`notebooks/hotspot_analysis.ipynb`** | Jupyter Notebook | **YES** | Interactive notebook is removed. | Run `python create_notebook.py` |

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

## ⚠️ Limitations & Future Work

- **Uniform Density Threshold:** Standard DBSCAN applies a single global $\epsilon$, which can under-cluster lower-density suburban zones. Future extensions should incorporate **HDBSCAN** for multi-density clustering.
- **Block-Level Coordinates:** Anonymized block-level reporting creates duplicate coordinates at intersections.
- **Spatio-Temporal Clustering:** Future iterations can evaluate **ST-DBSCAN** to incorporate time intervals into distance calculations.

---

## 📄 License & Attribution

- **Dataset:** Chicago Police Department Crime Dataset (`crimes.csv`), available on [Kaggle: Crimes in Chicago](https://www.kaggle.com/datasets/currie32/crimes-in-chicago).
- **Blueprint:** `project_blueprint.md`.
- **Implementation:** Unsupervised Machine Learning with Scikit-Learn & Folium.
