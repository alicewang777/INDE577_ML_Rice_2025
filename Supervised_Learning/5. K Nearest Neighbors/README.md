# K-Nearest Neighbors (KNN) - Uber Fare Classification

## a. Algorithm Description

<img width="474" alt="k_nearest_neighbors_1" src="https://github.com/user-attachments/assets/efc36945-a179-4a37-b045-a78cfb84b2f8" />

This notebook implements the **K-Nearest Neighbors (KNN)** classification algorithm using a custom-built implementation from the `rice_ml` package. The algorithm predicts whether a given Uber ride will have a **high fare (>$15)** or **low fare**, based on features such as pickup/dropoff locations, time of day, and passenger count.

The KNN algorithm classifies a test point by majority voting among its k nearest neighbors in the training set, using Euclidean distance as the similarity measure. The implementation does not rely on any built-in scikit-learn model; it is entirely constructed from scratch and accessed via:

```python
from rice_ml.knn import KNearestNeighbors
```

We also include a **balanced training version** using undersampling to address class imbalance.

### KNN Intuition

The diagram above illustrates how the K-Nearest Neighbors (KNN) algorithm classifies a new point `P`. It looks at the `k` nearest training samples—measured using distance (typically Euclidean)—and assigns `P` the majority label among those neighbors.

In this example:
- Point `P` lies near samples from three classes: **Class A (blue)**, **Class B (green)**, and **Class C (red)**.
- Based on the class labels of the nearest neighbors, the model selects the majority vote as the predicted class.

KNN is a **non-parametric**, **lazy learning** algorithm that requires no training phase. Instead, it makes predictions based on the structure of the data in feature space.

---

## b. Dataset Summary

We use a cleaned subset of the **Uber Pickup Data** (originally from Kaggle), focusing on ride-level features:

- **Input features**:
  - `pickup_longitude`, `pickup_latitude`, `dropoff_longitude`, `dropoff_latitude`
  - `passenger_count`, `hour_exact`, `dayofweek`, `is_weekend`
  - Interaction terms such as `hour_exact × passenger_count`
  - Distance feature: `manhattan_distance`

- **Target variable**:
  - `label`: A binary indicator where `1` = high fare (>$15), `0` = low fare

To ensure model efficiency, we sample **5,000 rows** and filter out records with extreme values or missing coordinates.

---

## c. Reproducing the Results

To reproduce the experiments in this notebook:

1. Clone the project repository (includes the `rice_ml` package):
    ```bash
    git clone https://github.com/alicewang777/INDE577_ML_Rice_2025.git
    ```

2. Mount Google Drive and load the Uber dataset CSV:
    - Path: `/content/drive/MyDrive/Colab Notebooks/ML_Rice_2025_Final_Project/uber.csv`

3. Run the notebook to:
    - Preprocess and standardize the dataset
    - Train the KNN model on both **raw** and **balanced** training data
    - Evaluate results using accuracy, precision, recall, and confusion matrix

4. Key imports:
    ```python
    from rice_ml.knn import KNearestNeighbors
    from rice_ml.metrics import accuracy
    ```

This project demonstrates the performance difference between a standard KNN classifier and one trained with class balancing to mitigate the effect of imbalanced datasets. It provides insights into the **accuracy-recall trade-off** in real-world classification problems.
