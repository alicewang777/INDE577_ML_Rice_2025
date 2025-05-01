# rice_ml

A lightweight custom machine learning library for educational use, built from scratch using NumPy. This project is part of the final project for INDE577/CMOR438 at Rice University.

---

## Package Overview

The `rice_ml` package currently includes implementations of the following supervised learning algorithms:

- `Perceptron`
- `LogisticRegression`
- (More models like `DecisionTree`, `RandomForest`, and `Boosting` are under development)

Each model is implemented from scratch without using high-level libraries like scikit-learn.

---

## Project Structure

```plaintext
rice_ml/
├── __init__.py              # Initializes the rice_ml package
├── perceptron.py            # Perceptron implementation
├── logistic_regression.py   # Logistic Regression implementation
├── decision_tree.py         # [Placeholder or implemented]
├── random_forest.py         # [Placeholder or implemented]
├── boosting.py              # [Placeholder or implemented]
├── metrics.py               # Evaluation metrics (accuracy, MSE, R², etc.)

notebooks/
└── rice_ml_package.ipynb    # Colab notebook for testing and developing models

## Notebook: `rice_ml_package.ipynb`

This notebook demonstrates how to:

- Build the `rice_ml` package dynamically in Google Colab
- Implement the `LogisticRegression` class using NumPy
- Test model training and prediction on a small synthetic dataset
- Set up the package for reuse in other notebooks

It is especially useful for verifying that the models are importable and functional across files, simulating real package usage.

---
