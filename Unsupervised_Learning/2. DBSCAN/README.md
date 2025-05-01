# DBSCAN Clustering – Uber Pickup Behavior in NYC

## Visualization and Interpretation

### 1. What Is DBSCAN?

<img width="630" alt="dbscan_1" src="https://github.com/user-attachments/assets/fab43c17-aab2-4110-a54d-b722673fb3ec" />

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a powerful unsupervised learning algorithm that groups data points into clusters based on density. It does **not require specifying the number of clusters** and can effectively detect **outliers** as noise.

---

### 2. Clustering Result – Projected on 2D Space

![dbscan_2](https://github.com/user-attachments/assets/8ac0c676-cb6b-4f95-91e3-dbc5971ecf67)

This scatter plot projects the clustering results onto geographic coordinates (longitude vs. latitude) to illustrate how DBSCAN separates pickup points:

- Densely packed trips are grouped into clusters (e.g., Cluster 0, 1, 2).
- Sparse or unusual pickups are labeled as noise (`-1`), shown in blue.
- Manhattan emerges as the primary dense pickup zone, while outer boroughs have more scattered demand.

---

### 3. Interactive Geomap View

![dbscan_3](https://github.com/user-attachments/assets/ae65e7b9-e2c6-4bc9-8bb6-e8fe9b43e47e)

To enhance spatial understanding, we visualize the DBSCAN output on an interactive map:

- Clusters reflect high-density pickup zones across NYC.
- Central business districts and transit hubs are clearly visible as pickup hotspots.
- Outliers can be geographically traced, revealing low-frequency or irregular travel patterns.

---

## Algorithm Description

This project implements **DBSCAN Clustering** using a custom-built version from the `rice_ml` Python package. The core logic includes:

- **Neighborhood radius (`eps`)** and **minimum sample count (`min_samples`)** are used to define dense regions.
- Points in high-density regions form core points; reachable points are added to the same cluster.
- Points that don’t fit into any cluster are marked as noise.

We used:
```python
from rice_ml.dbscan import DBSCAN
```

All logic is implemented from scratch without using scikit-learn’s DBSCAN.

---

## Dataset Summary

We use a cleaned and sampled subset of the **Uber Pickup Data** for New York City. Key steps:

- Removed invalid or missing coordinates and dates.
- Filtered extreme fare values and outliers.
- Created derived features: exact hour, day of week, Manhattan distance, weekend flag, and interaction terms.
- Final dataset contains 3,000+ sampled Uber trips.

### Features used for clustering:
- `pickup_longitude`, `pickup_latitude`
- `hour_exact`, `minute`, `dayofweek`, `is_weekend`
- `passenger_count`, `manhattan_distance`, `hour_passenger`

---

## How to Reproduce

1. Clone the repository:
```bash
git clone https://github.com/alicewang777/INDE577_ML_Rice_2025.git
```

2. Load the dataset (`uber.csv`) from the specified Google Drive path.

3. Preprocess and scale features as shown in the notebook.

4. Run the DBSCAN model from your custom package:
```python
from rice_ml.dbscan import DBSCAN

model = DBSCAN(eps=1.2, min_samples=10)
model.fit(X_scaled)
```

5. Visualize results with `matplotlib` (static) or `folium` (interactive map).
