# AdaBoost – Ensemble Methods for Uber Fare Classification

## Algorithm Description

This directory implements the **AdaBoost (Adaptive Boosting)** classification algorithm using a custom implementation from the `rice_ml` package:

```python
from rice_ml.boosting import AdaBoost
```
<img width="595" alt="ensemble_methods_boosting_3" src="https://github.com/user-attachments/assets/39b85c5c-7bb1-4a20-b183-431a87d89f4a" />

AdaBoost is an **ensemble learning** method that combines multiple **weak learners** (typically shallow decision trees or "decision stumps") into a single strong classifier. Each weak learner is trained in sequence, with more focus on previously misclassified examples. The final prediction is a weighted vote across all learners.

<img width="924" alt="ensemble_methods_boosting_1" src="https://github.com/user-attachments/assets/ba868664-d873-4a5c-8c56-e96715771ef4" />

In class, our professor introduced this concept using the famous **"Wisdom of the Crowd"** example. We watched a BBC video clip in which many people guessed how many jelly beans were inside a jar. Surprisingly, the **average of all guesses** was remarkably close to the true number of beans—much closer than most individual guesses. This example beautifully demonstrates the core intuition behind ensemble learning: **aggregated judgments often outperform individual ones**.

<img width="884" alt="ensemble_methods_boosting_2" src="https://github.com/user-attachments/assets/49990b88-78c3-427e-8e2f-5e9b6bfb2c03" />

> Each weak model may only be slightly better than random guessing, but when combined intelligently, they form a robust and accurate ensemble model.

---

## Dataset Summary

The dataset used for training and evaluation is a cleaned version of the **Uber pickup and fare data**, originally sourced from Kaggle. Each row contains trip details such as pickup/dropoff coordinates, timestamp, passenger count, and total fare.

Key details:

- **Target Variable**: A binary label indicating whether the fare was **high (> $15)** or **low (≤ $15)**.
- **Sample Size**: 5,000 randomly sampled trips to reduce training time.
- **Features Used**:
  - Geolocation: `pickup_latitude`, `pickup_longitude`, `dropoff_latitude`, `dropoff_longitude`
  - Temporal: `hour_exact`, `dayofweek`, `is_weekend`
  - Trip metrics: `passenger_count`, `manhattan_distance`, `hour_exact × passenger_count`

---

## How to Reproduce

1. Clone the GitHub repo and import the custom `rice_ml` package:

```bash
!git clone https://github.com/alicewang777/INDE577_ML_Rice_2025.git
```

2. In your notebook, mount your Google Drive and load the Uber dataset:

```python
from rice_ml.boosting import AdaBoost
from rice_ml.metrics import accuracy
```

3. Follow the preprocessing, training, and evaluation steps as shown in the provided notebook:
   - Feature scaling using `StandardScaler`
   - Train-test split
   - Fit the `AdaBoost` model on training data
   - Evaluate accuracy, precision, recall, and plot confusion matrix

---

## Performance Highlights

| Metric     | Score   |
|------------|---------|
| Accuracy   | 92.90%  |
| Precision  | 87.94%  |
| Recall     | 69.66%  |

Confusion matrix plot:

![ensemble_methods_boosting_4](https://github.com/user-attachments/assets/5ae6e8e0-69fe-461e-9633-a7b8591f8490)

The AdaBoost model showed high precision, meaning most of the predicted high-fare rides were correct. Although the recall is slightly lower, the model still captures a large portion of actual high-fare rides, offering a solid balance between false positives and false negatives.

---

## Summary

This project demonstrates how ensemble methods like AdaBoost can significantly enhance classification performance by combining simple learners. Inspired by real-world intuition and supported by theory, boosting is a powerful addition to any machine learning toolkit.
