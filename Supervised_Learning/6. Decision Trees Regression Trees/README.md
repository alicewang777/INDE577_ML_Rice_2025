# Decision Tree Classification - Uber Fare Prediction

## a. Algorithm Description

This notebook applies a **Decision Tree Classifier**, implemented from scratch in the custom-built `rice_ml` package, to predict whether an Uber ride results in a high fare (greater than $15). The model is based on recursive binary splits that minimize Gini impurity, forming a hierarchical structure of decisions.

We directly call the classifier via:

```python
from rice_ml.decision_tree import DecisionTree
```

A decision tree consists of internal decision nodes, which test specific feature values (e.g., distance, time, passenger count), and leaf nodes, which provide the final classification (high fare or low fare). The image below illustrates a simplified decision tree structure, where a series of binary conditions guide the decision process from the root node to the final outcome.

<img width="713" alt="decision_trees_regression_trees_1" src="https://github.com/user-attachments/assets/219453e2-6540-432f-bb5c-e56684553600" />

Each internal node represents a question (e.g., "Is the pickup time during rush hour?"), and each leaf node represents a classification outcome. In our model, the structure is learned directly from the data based on maximizing information gain (or minimizing impurity).

## b. Dataset Summary

We use a cleaned and downsampled version of the **Uber trip dataset**, containing detailed ride-level information including:

- **Location Features**: pickup/dropoff coordinates
- **Time Features**: hour of the day, day of the week, weekend indicator
- **Derived Features**: manhattan distance, interaction between time and passenger count

The dataset contains **5,000 samples**, and the binary target label is defined as:

- `1`: fare_amount > 15 (high fare)
- `0`: fare_amount ≤ 15 (low fare)

All features are standardized using `StandardScaler`. The dataset is split into 80% training and 20% testing.

## c. Instructions for Reproducing Results

1. Clone the GitHub repo:
    ```bash
    !git clone https://github.com/alicewang777/INDE577_ML_Rice_2025.git
    ```
2. Mount Google Drive and load the Uber dataset.
3. Import the decision tree model:
    ```python
    from rice_ml.decision_tree import DecisionTree
    ```
4. Follow the notebook to:
    - Load and clean the data
    - Engineer features
    - Train the model with specified max depth and min sample split
    - Evaluate performance using accuracy, precision, recall
    - Visualize confusion matrix

## Model Evaluation Results

- **Accuracy**: 93.1%
- **Precision**: 88.1%
- **Recall**: 70.8%

The model performs well at identifying both high- and low-fare rides, particularly excelling at minimizing false positives. The confusion matrix confirms strong classification ability across both classes. This balance makes the model useful in pricing prediction systems or fraud detection scenarios.
