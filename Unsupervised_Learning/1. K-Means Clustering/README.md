# K-Means Clustering – Uber Pickup Behavior

## Visualization and Interpretation

After clustering, we visualize results in two ways:

### 1. **2D Scatterplot** (Longitude vs. Latitude)

Although clustering was performed using high-dimensional behavioral and temporal data, we project the results onto a 2D geospatial plot to aid interpretation:

![kmeans_clustering_2](https://github.com/user-attachments/assets/95c39d21-f4bf-41d6-8947-eb5828c1e8ee)

This view highlights clear cluster formations across New York City, including Midtown Manhattan, the Financial District, and parts of Brooklyn and Queens. The colors indicate distinct groups of trips sharing both location and behavioral characteristics.

### 2. **Interactive Map View**

To further connect clustering results to real geography, we render the output on an interactive map using `folium`:

![kmeans_clustering_3](https://github.com/user-attachments/assets/c3e9d19d-3942-409a-b037-e9668cd9b45a)

Each dot represents a pickup point, color-coded by cluster assignment. Densely packed clusters in central Manhattan reflect high-activity areas, while scattered groups toward outer boroughs suggest more heterogeneous behavior. The clustering algorithm successfully segments trips based on both **where** and **when/how** rides occurred.

---

## Algorithm Description

This project implements the **K-Means Clustering** algorithm using a custom-built version from the `rice_ml` Python package. K-Means is an unsupervised machine learning algorithm that partitions samples into `k` distinct groups based on similarity.

The algorithm proceeds as follows:

1. Randomly initialize `k` centroids from the data.
2. Assign each data point to the closest centroid.
3. Recalculate centroids as the mean of each cluster.
4. Repeat until assignments no longer change (or convergence criteria is met).

<img width="584" alt="kmeans_clustering_1" src="https://github.com/user-attachments/assets/02a55a6c-de4b-4ed4-9749-420756a4178c" />

In this project, we use not only spatial data, but also temporal and behavioral features:

- Pickup coordinates
- Time of day (hour + minute)
- Day of week
- Weekend flag
- Passenger count
- Trip distance (Manhattan distance)
- Interaction term: hour × passenger count

This multidimensional representation captures the full context of ride behavior beyond location alone.

---

## Dataset Summary

We use a subset of the Uber NYC trip dataset (originally from Kaggle), filtered and cleaned for quality and relevance:

- **Sample size**: 5,000 rides
- **Geography**: NYC core area only (longitude/latitude trimmed)
- **Features used**: pickup location, time, passenger count, fare

The dataset is preprocessed to remove extreme fares, missing values, and invalid coordinates.

---

## Instructions to Reproduce

To reproduce this clustering pipeline:

1. Clone this repo and install dependencies:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   pip install -r requirements.txt
   ```

2. Ensure your local directory includes:
   - `uber.csv` (cleaned 5,000-sample dataset)
   - `rice_ml/` package folder with `kmeans.py`

3. Run the clustering notebook or script:
   ```python
   from rice_ml.kmeans import KMeans
   ```

4. The notebook walks through:
   - Feature engineering
   - Scaling and clustering
   - Static and interactive visualizations
