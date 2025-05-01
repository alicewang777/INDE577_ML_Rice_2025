# Neural Network - Uber Fare Classification

## Algorithm Overview

<img width="1235" alt="neural_networks_1" src="https://github.com/user-attachments/assets/8448ffc1-83aa-4d00-9ca5-201cad7a8fb9" />

This project implements a **Simple Feedforward Neural Network** from scratch using NumPy to classify Uber rides as either **high fare** (> $15) or **low fare**.

The neural network includes:
- One hidden layer with **ReLU** activation
- A final output neuron using **sigmoid** activation
- Custom **forward and backward propagation**
- Manual **gradient descent** for weight updates

The diagram above illustrates the learning workflow:
1. **Forward Propagation**: Input features pass through weighted connections and activation functions to generate predictions.
2. **Loss Function**: The model compares predictions with actual labels using a binary loss function.
3. **Backward Propagation**: Errors are backpropagated to update weights via gradient descent.
4. This iterative loop continues until the loss is minimized.

### Custom Package

The model used in this project is not from a built-in library like scikit-learn or TensorFlow.  
Instead, it was implemented **entirely from scratch** and organized into a custom Python package called `rice_ml`, which includes:

- `neural_network.py` – Manual implementation of a simple feedforward neural network using NumPy  
- `metrics.py` – Custom accuracy function and other evaluation tools

This modular package design allows for clean experimentation, reproducibility, and future extension (e.g., adding other models like logistic regression or decision trees).

> All models in this project (including this neural network) are invoked from the self-built `rice_ml` package, which reflects our ability to build and use machine learning infrastructure from the ground up.


## Dataset Summary

We use a subset of the **Uber Pickup Data**, originally sourced from Kaggle.

- **Inputs (Features)**:
  - Pickup/Dropoff Coordinates
  - Passenger Count
  - Time of Day (hour + minute)
  - Day of Week, Weekend Indicator
  - Engineered Features: `manhattan_distance`, `hour_exact_x_passenger`

- **Output (Label)**:
  - Binary variable: `1` if fare amount > $15, else `0`

- **Preprocessing**:
  - Outliers removed: `fare_amount >= 100`
  - Random sample of **5,000 rides** for training efficiency

We also explore **undersampling** the majority class (low-fare rides) to improve model performance on the minority class.

---

## Reproducing the Results in Google Colab

1. **Clone the GitHub repo and set path:**
   ```python
   !git clone https://github.com/alicewang777/INDE577_ML_Rice_2025.git
   import sys
   sys.path.append('/content/INDE577_ML_Rice_2025')
   ```

2. **Mount Google Drive and load the dataset:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   import pandas as pd
   df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ML_Rice_2025_Final_Project/uber.csv')
   ```

3. **Train the model using:**
   - Imbalanced data (raw)
   - Balanced data (via **undersampling**)

4. **Evaluate performance using:**
   - Accuracy
   - Precision
   - Recall
   - Confusion Matrix

---

## Result Comparison

| Model Type              | Accuracy | Precision | Recall  |
|-------------------------|----------|-----------|---------|
| Raw Training (Imbalanced)     | 82.2%    | 0.0000    | 0.0000  |
| Undersampled Training (Balanced) | 29.7%    | 0.1915    | 91.6%   |

> The raw model achieves high overall accuracy but fails to detect any high-fare rides.  
> The undersampled model sacrifices accuracy but dramatically improves **recall**, making it better suited for identifying costly trips in applications where missing rare events is critical.

---

## Authors

**Alice Wang**  

---

## Reference

Pramoditha, R. (2022, February 1). Overview of a Neural Network’s Learning Process. Medium. https://medium.com/data-science-365/overview-of-a-neural-networks-learning-process-61690a502fa
