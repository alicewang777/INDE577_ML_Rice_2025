# Logistic Regression – Uber Fare Classification

## Algorithm Description

![logistic_regression_2](https://github.com/user-attachments/assets/a82fb653-2f9d-46d1-951f-2853ec9927f3)

This implementation features a **custom-built Logistic Regression classifier** developed from scratch as part of the `rice_ml` package for the INDE577/CMOR438 Final Project. The algorithm uses gradient descent to optimize weights for binary classification. Specifically, it estimates the probability that an Uber ride has a **high fare amount (>$20)** given a set of geospatial and temporal features.

Unlike library-based models, this implementation manually computes the sigmoid activation, loss function (binary cross-entropy), and gradient updates, reinforcing understanding of the mathematical underpinnings of logistic regression.

---

## Dataset Summary

- **Dataset Source**: [Uber Ride Fare Dataset](https://www.kaggle.com/datasets/yasserh/uber-fare-prediction)
- **Format**: CSV
- **Sample Size**: 5000 Uber trips (cleaned and filtered)
- **Target Variable**:  
  - `label = 1` if `fare_amount > $20` (High Fare)  
  - `label = 0` otherwise

- **Key Features Used**:
  - `pickup_longitude`, `pickup_latitude`
  - `dropoff_longitude`, `dropoff_latitude`
  - `passenger_count`
  - `hour` (from pickup time)
  - `dayofweek` (0 = Monday, 6 = Sunday)
  - `manhattan_distance` (proxy for travel distance)

---

## Instructions to Reproduce

1. **Clone the project repository** in Colab:
    ```python
    !git clone https://github.com/alicewang777/INDE577_ML_Rice_2025.git
    sys.path.append('/content/INDE577_ML_Rice_2025')
    ```

2. **Import the custom model**:
    ```python
    from rice_ml.logistic_regression import LogisticRegression
    from rice_ml.metrics import accuracy
    ```

3. **Load and preprocess the dataset**:
    - Clean zero-coordinate rows
    - Convert `pickup_datetime`
    - Create binary label (`fare_amount > 20`)
    - Standardize features

4. **Train the model**:
    ```python
    model = LogisticRegression(learning_rate=0.01, n_iters=1000)
    model.fit(X_train, y_train)
    ```

5. **Evaluate results**:
    ```python
    y_pred = model.predict(X_test)
    acc = accuracy(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    ```

6. **Visualize outcomes**:
    - Confusion matrix
    - Coefficient interpretation

---

## Note

This model is built using the `rice_ml` package, a bonus custom library created entirely from scratch. The intent is to demonstrate low-level implementation skills without relying on `scikit-learn` or similar libraries.

## Reference
Knoldus Inc. (2018, March 28). MachineX: Simplifying Logistic Regression. Medium. https://medium.com/knoldus/machinex-simplifying-logistic-regression-93b2e6d88a8a
