# Principal Component Analysis (PCA) – Uber Pickup Behavior

<img width="1073" alt="principal_component_analysis_1" src="https://github.com/user-attachments/assets/7d324225-e7e6-4cac-a522-59ef8116dfb2" />

## Overview

This directory implements **Principal Component Analysis (PCA)** using a custom-built algorithm from the `rice_ml` package. PCA is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving as much variance as possible. It is especially useful for visualizing complex datasets and uncovering latent structure.

The above image illustrates how PCA works: high-dimensional data (left panel) is projected onto a new coordinate system (right panel) defined by the directions of maximum variance (principal components). Each axis in the component space captures a linear combination of the original features, simplifying the analysis of patterns and clustering.

In this project, I applied PCA to the **Uber Pickup Dataset** to uncover underlying patterns in passenger ride behavior across different times, locations, and distances.

---

## Dataset Summary

We use a cleaned and preprocessed subset of the **Uber Pickup Data**, containing information on 5000 randomly sampled rides. Key features engineered include:

- **Time features**: hour, minute, weekday/weekend
- **Trip features**: pickup/dropoff coordinates, Manhattan distance
- **Interaction terms**: e.g., `hour × passenger_count`

Before applying PCA, we standardized the features and removed outliers (e.g., rides with zero coordinates or unreasonable fare amounts).

---

## Visualization of PCA Projection

After applying PCA, we projected the standardized features into 2D space using the top two principal components. The resulting scatterplot reveals clusters and patterns in Uber pickup behavior, such as groups of similar ride types or time-based patterns.

![principal_component_analysis_2](https://github.com/user-attachments/assets/82fe6d1a-00e4-453c-8ba5-618bbef5fa34)

Key takeaways:
- Distinct clusters indicate recurring behavior patterns, possibly by time or trip length.
- Outlier removal and standardization greatly improved PCA performance.
- PCA enables further downstream tasks such as clustering and anomaly detection.

---

## Instructions for Reproducing Results

1. Clone the repository and install required packages:
```bash
git clone https://github.com/alicewang777/INDE577_ML_Rice_2025.git
```

2. In your notebook or script, append the repo to your path and import the PCA model:
```python
import sys
sys.path.append('/content/INDE577_ML_Rice_2025')
from rice_ml.pca import PCA
```

3. Load and clean the Uber dataset, perform feature engineering, standardize, and apply PCA. Visualization code is included in the notebook.

> Note: The PCA implementation used here is written from scratch as part of a custom Python package (`rice_ml`). It does not rely on external libraries such as scikit-learn, and its use is intended to demonstrate a clear understanding of the algorithm’s internals.

---

## References

Im, J. (2018, December 6). Introduction to PCA (Principal Component Analysis). Medium. https://medium.com/@jamesim2077/introduction-to-pca-principal-component-analysis-c26dffe2a857​
