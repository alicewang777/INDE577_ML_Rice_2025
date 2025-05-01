# rice_ml

This is a custom machine learning package developed for the INDE577/CMOR438 final project. It includes hand-implemented versions of classical supervised and unsupervised algorithms.

## Contents

- `perceptron.py`: Binary classification using Perceptron algorithm
- `logistic_regression.py`: Logistic Regression model
- `decision_tree.py`: Decision Tree classifier
- `random_forest.py`: Random Forest ensemble
- `boosting.py`: AdaBoost implementation
- `neural_network.py`: Simple feedforward neural network
- `kmeans.py`: K-Means clustering
- `dbscan.py`: Density-based clustering (DBSCAN)
- `pca.py`: Principal Component Analysis
- `metrics.py`: Common metrics for classification and regression

## Usage

You can import any model from this package in your Colab notebooks like:

```python
from rice_ml.perceptron import Perceptron
