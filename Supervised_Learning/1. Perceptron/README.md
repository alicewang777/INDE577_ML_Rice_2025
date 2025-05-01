# Perceptron Model: Predicting High-Fare Uber Rides

## a. Algorithm Description

![Perceptron_1](https://github.com/user-attachments/assets/f9c99053-72e0-4b0b-8d7c-6239ccd9f0ef)

*Image source: neurallearner (2017), Deep Learning: An Extension of the Perceptron, Steemit. https://steemit.com/technology/@neurallearner/deep-learning-an-extension-of-the-perceptron


![Perceptron_2](https://github.com/user-attachments/assets/ba108c45-5f81-4f35-8985-30d219928526)

This module implements the **Perceptron algorithm**, a classic linear classifier for binary classification tasks. The model was implemented from scratch as part of the custom package `rice_ml`, which includes reusable components for training and evaluating machine learning models.

Key features:
- Batch Perceptron training with user-defined learning rate and number of iterations
- Custom evaluation metrics (e.g., accuracy)
- Feature weight interpretation for model explainability

## b. Dataset Summary

The model is trained and tested on a sample of the **NYC Uber trip dataset**, which includes:

- Pickup/dropoff locations (latitude & longitude)
- Pickup datetime
- Passenger count
- Fare amount (used to define the binary label)

The binary classification task is to predict whether a ride is **high-fare** (`fare_amount > 15`) or not. After cleaning and feature engineering, the dataset includes engineered features such as:
- Manhattan distance
- Hour of day (exact)
- Day of week
- Weekend indicator
- Interaction term: `hour_exact × passenger_count`

## c. Instructions to Reproduce

1. **Clone the project and set up package path** in your Colab or local environment:

```python
!git clone https://github.com/alicewang777/INDE577_ML_Rice_2025.git
import sys
sys.path.append('/content/INDE577_ML_Rice_2025')
```

2. **Import the Perceptron model from the custom package:**

```python
from rice_ml.perceptron import Perceptron
```

3. **Load and preprocess the dataset** using the included notebook. Make sure to:
   - Filter invalid coordinates
   - Convert pickup time to datetime
   - Create the binary label (`fare_amount > 15`)
   - Scale features using `StandardScaler`

4. **Train and evaluate the model**:

```python
model = Perceptron(learning_rate=0.0005, n_iters=2000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

5. **Evaluate performance** using accuracy, precision, recall, confusion matrix, and feature weights.

---

For full implementation, see the Jupyter notebook in this directory.

**Reference:**

Neurallearner. (2017, October 25). Deep learning: An extension of the perceptron. Steemit. https://steemit.com/technology/@neurallearner/deep-learning-an-extension-of-the-perceptron
