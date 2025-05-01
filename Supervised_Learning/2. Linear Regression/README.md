# Linear Regression – Uber Fare Prediction

## Description

This directory contains a custom implementation of the **Linear Regression** algorithm from scratch, developed as part of the `rice_ml` package for the INDE577/CMOR438 Machine Learning Final Project.

The model estimates the fare amount of Uber rides based on trip-related and temporal features, using standard gradient descent for parameter estimation. No external ML libraries (e.g., scikit-learn) were used in the core model.

---

## Dataset Summary

**Source**: Uber pickup data  
**Format**: CSV  
**Target variable**: `fare_amount` (continuous value in USD)

**Features used**:
- pickup and dropoff coordinates (longitude, latitude)
- passenger count
- ride time (`hour_exact`)
- day of the week, weekend indicator
- engineered features such as:
  - Manhattan distance
  - interaction term: hour × passenger count

Outliers with `fare_amount >= 100` were removed, and a random sample of 5,000 rides was used for training and evaluation.

---

## How to Reproduce

1. Clone the GitHub repo:
   ```bash
   git clone https://github.com/alicewang777/INDE577_ML_Rice_2025.git
   ```

2. Open the Colab notebook located at:
   ```
   /Supervised_Learning/Linear_Regression.ipynb
   ```

3. Mount your Google Drive and load the dataset:
   - Upload your `uber.csv` to the expected path:
     ```
     /content/drive/MyDrive/Colab Notebooks/ML_Rice_2025_Final_Project/uber.csv
     ```

4. Run the notebook step by step to:
   - Load and preprocess the data
   - Train the custom Linear Regression model
   - Evaluate using MSE and R²
   - Visualize prediction results and feature weights

---

## Dependencies

- pandas, numpy
- matplotlib, seaborn
- scikit-learn (for data preprocessing only)

---

## Notes

- The Linear Regression class is implemented in:  
  `rice_ml/linear_regression.py`
- Evaluation metrics used:
  - Mean Squared Error (MSE)
  - R² Score

---
