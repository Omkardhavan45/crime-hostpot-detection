# Crime Hotspot Detection using DBSCAN

**Mini Project Blueprint**

---

## 1. Project Overview

| Item | Details |
|---|---|
| Title | Crime Hotspot Detection using DBSCAN |
| Domain | Public safety, geospatial data mining |
| Technique | Unsupervised learning (DBSCAN, haversine distance) |
| Dataset | Crimes in Chicago (Kaggle, 2001-2017) |
| File used | `Chicago_Crimes_2012_to_2017.csv`, filtered to one year (2016) |
| Language | Python |

### Problem Statement

Crime incidents often recur in specific geographical areas, making it difficult for authorities to identify high-risk locations and take preventive measures. This project detects crime-prone hotspots by applying DBSCAN to historical crime data containing latitude and longitude. Dense groups of incidents become clusters (hotspots), and isolated incidents are treated as noise.

### Proposed Solution

1. Clean and filter historical crime records.
2. Cluster incident coordinates with DBSCAN using the haversine metric.
3. Rank the resulting clusters by size and dominant crime type.
4. Show the hotspots on an interactive map to support patrol planning.

### Objectives

- Detect crime hotspots without specifying the number of clusters in advance.
- Separate dense hotspots from isolated incidents (noise).
- Describe each hotspot: location, size, dominant crime type, peak hour.
- Present results clearly for non-technical readers.

---

## 2. Dataset

**Source:** https://www.kaggle.com/datasets/currie32/crimes-in-chicago

**Backup / newer data:** City of Chicago Data Portal, "Crimes - 2001 to Present"
(https://data.cityofchicago.org/d/ijzp-q8t2)

### Columns Required

| Column | Purpose | Derived features |
|---|---|---|
| `Latitude`, `Longitude` | DBSCAN input | radians, cluster centroid, radius |
| `Date` | Time analysis | `year`, `month`, `day_of_week`, `hour`, `time_bucket` |
| `Primary Type` | Filtering and labelling | dominant crime type per cluster |
| `Arrest` (optional) | Outcome context | arrest rate per cluster |
| `Location Description` (optional) | Place context | dominant location type |

### Columns to Ignore

`ID`, `Case Number`, `Block`, `IUCR`, `Description`, `Domestic`, `Beat`, `District`, `Ward`, `Community Area`, `FBI Code`, `X Coordinate`, `Y Coordinate`, `Updated On`, `Location`, and the index column `Unnamed: 0`.

### Known Data Issues

- Locations are shown at block level, so many incidents share identical coordinates.
- Some rows have missing or invalid coordinates.
- The file is large (about 1.4 million rows). Load only the columns needed.
- The data ends in 2017, so results describe historical patterns.

---

## 3. System Architecture

```
Raw CSV
   |
   v
Data Loading (usecols, skip bad lines)
   |
   v
Cleaning (drop nulls, parse dates, bounding box filter)
   |
   v
Filtering (one year, optional crime types, optional sampling)
   |
   v
Feature Preparation (convert lat/lon to radians)
   |
   v
Parameter Selection (k-distance plot -> eps, choose min_samples)
   |
   v
DBSCAN (haversine metric, ball_tree)
   |
   v
Post-processing (cluster stats, ranking, dominant crime, peak hour)
   |
   v
Visualization (Folium map, charts) + Report
```

---

## 4. Step-by-Step Plan

### Phase 1: Setup

- Install libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `folium`, `seaborn` (optional).
- Download the dataset from Kaggle and place the CSV in the project folder.

### Phase 2: Data Loading and Cleaning

1. Read only the needed columns with `usecols`.
2. Use `on_bad_lines="skip"` and `low_memory=False` for the large file.
3. Drop rows with missing `Latitude`, `Longitude` or `Date`.
4. Parse `Date` with `pd.to_datetime(..., errors="coerce")`.
5. Keep one year (2016).
6. Remove coordinates outside Chicago (latitude 41.6 to 42.1, longitude -87.95 to -87.5).
7. Remove (0, 0) coordinates if present.

> Do not use `nrows` for the final run. The file is ordered by date, so the first N rows come from the earliest years only. Use `nrows` only for quick testing, and use `df.sample()` if you need a smaller set.

### Phase 3: Exploratory Data Analysis (EDA)

- Count incidents per `Primary Type` (bar chart).
- Incidents by hour of day and day of week.
- Scatter plot of raw coordinates to see the city shape.
- Check duplicates in coordinates.

### Phase 4: Choose Scope

- Option A: all crime types for one year.
- Option B (recommended for clarity): one or two crime types, e.g. THEFT, or THEFT vs NARCOTICS.
- If the set is still large, take a random sample of 50,000 rows (`random_state=42`).

### Phase 5: Parameter Selection

1. Convert coordinates to radians: `coords = np.radians(df[["Latitude", "Longitude"]])`.
2. Plot a k-distance graph (`NearestNeighbors`, k = `min_samples`, haversine metric) and look for the elbow to pick `eps`.
3. Convert `eps` from kilometres to radians: `eps_rad = eps_km / 6371.0088`.
4. Typical starting values: `eps` = 0.1 to 0.3 km, `min_samples` = 10 to 50 (higher for larger datasets).
5. Try a small grid of values and compare results.

### Phase 6: Run DBSCAN

```python
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=eps_rad, min_samples=min_samples,
            algorithm="ball_tree", metric="haversine")
df["cluster"] = db.fit_predict(coords)
```

- Label `-1` means noise.
- Count clusters, noise points and the noise ratio.

### Phase 7: Cluster Analysis

For each cluster compute:

| Metric | Meaning |
|---|---|
| `incident_count` | Number of crimes in the hotspot |
| `centroid_lat`, `centroid_lon` | Hotspot centre |
| `radius_km` | Spread of the hotspot |
| `dominant_crime` | Most common `Primary Type` |
| `peak_hour` | Most common hour |
| `arrest_rate` | Share of incidents with an arrest (optional) |
| `risk_rank` | Rank by incident count |

### Phase 8: Visualization

- Folium map with clusters in different colours and noise in grey.
- Markers at cluster centroids with popups (count, dominant crime, peak hour).
- Optional heatmap layer for comparison.
- Bar chart of the top 10 hotspots.
- Save the map as `hotspots_map.html`.

### Phase 9: Evaluation

No ground-truth labels exist, so evaluate with:

- Number of hotspots detected.
- Incidents per cluster and distribution of cluster sizes.
- Noise ratio (very high means `eps` is too small, one giant cluster means `eps` is too large).
- Silhouette score (haversine metric, noise excluded, computed on a sample).
- Visual check on the map against known busy areas.
- Optional validation: cluster 2015 data, then check how many 2016 incidents fall inside those hotspots.
- Usefulness: could police use the ranked list for patrol planning?

### Phase 10: Report and Presentation

Sections to write:

1. Introduction and problem statement
2. Literature or related work (brief)
3. Dataset description
4. Methodology (preprocessing, DBSCAN, parameter selection)
5. Implementation
6. Results (tables, charts, map screenshots)
7. Discussion and limitations
8. Conclusion and future work

---

## 5. Suggested Folder Structure

```
crime-hotspot-dbscan/
|-- data/
|   |-- Chicago_Crimes_2012_to_2017.csv
|-- notebooks/
|   |-- hotspot_analysis.ipynb
|-- outputs/
|   |-- hotspots_map.html
|   |-- top_hotspots.csv
|   |-- k_distance_plot.png
|-- report/
|   |-- project_report.docx
|-- README.md
|-- requirements.txt
```

`requirements.txt`:

```
pandas
numpy
scikit-learn
matplotlib
folium
seaborn
```

---

## 6. Expected Outputs

- Cleaned dataset for the chosen year.
- k-distance plot with the chosen `eps`.
- Cluster count, noise count and noise ratio.
- Table of the top hotspots (rank, centroid, size, dominant crime, peak hour).
- Interactive Folium map.
- Summary chart of the top hotspots.

---

## 7. Evaluation Criteria (from problem statement)

| Criterion | How we measure it |
|---|---|
| Number of hotspots detected | Count of clusters (excluding noise) |
| Incidents within each cluster | `incident_count` per cluster |
| Usefulness for safety planning | Ranked list, map, dominant crime and peak hour per hotspot |

---

## 8. Limitations to Mention

- A single global `eps` struggles when density varies a lot across the city.
- Results are sensitive to `eps` and `min_samples`.
- Block-level coordinates create many identical points.
- Data reflects recorded crime, not true crime. Heavily policed areas may show more records.
- Hotspots are historical and do not guarantee future events.
- DBSCAN ignores time and severity unless data is filtered or weighted beforehand.

---

## 9. Future Scope

- Compare with HDBSCAN or KDE heatmaps.
- Cluster by time slot (night vs day) or season.
- Weight incidents by severity.
- Build a Streamlit dashboard.
- Apply the same method to Indian city data (e.g. Pune) when open coordinate data is available.

---

## 10. Timeline (suggested)

| Week | Task |
|---|---|
| 1 | Setup, dataset download, loading and cleaning |
| 2 | EDA and scope selection |
| 3 | Parameter selection and DBSCAN runs |
| 4 | Cluster analysis, map and charts |
| 5 | Evaluation and experiments (different `eps`, crime types) |
| 6 | Report writing and presentation |

---

## 11. Quick Checklist

- [ ] Dataset downloaded and loaded with `usecols`
- [ ] Data cleaned (nulls, dates, coordinate range)
- [ ] One year selected (2016)
- [ ] Coordinates converted to radians
- [ ] k-distance plot drawn and `eps` chosen
- [ ] DBSCAN run with haversine metric
- [ ] Cluster statistics table created
- [ ] Folium map exported
- [ ] Evaluation metrics recorded
- [ ] Report written
