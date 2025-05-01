# rice_ml

A lightweight custom machine learning library for educational use, built from scratch using NumPy. This project is part of the final project for INDE577/CMOR438 at Rice University.

---

## Package Overview

The `rice_ml` package currently includes implementations of the following machine learning algorithms:

### Supervised Learning

- `Perceptron` (`perceptron.py`)
- `Logistic Regression` (`logistic_regression.py`)
- `Linear Regression` (`linear_regression.py`)
- `Decision Tree` (`decision_tree.py`)
- `Random Forest` (`random_forest.py`)
- `Boosting` (`boosting.py`)
- `Neural Network` (`neural_network.py`)
- `K-Nearest Neighbors (KNN)` (`knn.py`)

### Unsupervised Learning

- `KMeans` (`kmeans.py`)
- `DBSCAN` (`dbscan.py`)
- `PCA` (`pca.py`)

### Evaluation Metrics

- Accuracy, MSE, MAE, R² (`metrics.py`)

Each model is implemented from scratch using only NumPy. No high-level ML libraries (e.g., scikit-learn) are used.

---

## Project Structure

```plaintext
rice_ml/
├── __init__.py                # Initializes the rice_ml package
├── perceptron.py              # Perceptron model
├── logistic_regression.py     # Logistic Regression model
├── linear_regression.py       # Linear Regression model
├── decision_tree.py           # Decision Tree model
├── random_forest.py           # Random Forest model
├── boosting.py                # AdaBoost or Gradient Boosting
├── neural_network.py          # Simple feedforward Neural Network
├── knn.py                     # K-Nearest Neighbors
├── kmeans.py                  # K-Means clustering
├── dbscan.py                  # DBSCAN clustering
├── pca.py                     # Principal Component Analysis
├── metrics.py                 # Evaluation metrics

notebooks/
└── rice_ml_package.ipynb      # Colab notebook for testing and developing models
```

---

## Notebook: `rice_ml_package.ipynb`

This notebook demonstrates how to:

- Build the `rice_ml` package dynamically in Google Colab
- Test model training and prediction on a small synthetic dataset
- Set up the package for reuse in other notebooks

It is especially useful for verifying that the models are importable and functional across files, simulating real package usage.

---

## How to Use

1. Clone the repository:

    ```bash
    git clone https://github.com/alicewang777/INDE577_ML_Rice_2025.git
    cd INDE577_ML_Rice_2025
    ```

2. Open `notebooks/rice_ml_package.ipynb` in Google Colab or JupyterLab.

3. Run the notebook to:
    - Define and write the `rice_ml` modules
    - Import and use the models like:

      ```python
      from rice_ml.logistic_regression import LogisticRegression
      ```

---

## Testing and Validation

The notebook includes toy data to verify the correctness of the logistic regression model. Future versions will include:

- Unit tests using `pytest`
- Model performance metrics on real-world datasets (e.g., Uber dataset)

---

## License & Attribution

This is a student project for educational purposes. Feel free to fork, star, or contribute!


