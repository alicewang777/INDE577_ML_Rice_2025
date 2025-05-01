# Random Forest Classification – Uber Fare Prediction

## a. Algorithm Description

This notebook implements the **Random Forest** classification algorithm using a custom-built version from the `rice_ml` package. Random Forest is an ensemble learning method that combines the output of multiple decision trees to improve classification accuracy and robustness.

In our implementation, each decision tree is trained on a different bootstrap sample of the dataset. Predictions from all trees are aggregated using **majority voting** to produce the final class label. This ensemble strategy reduces overfitting and improves generalization.

### How Random Forest Works

<img width="529" alt="random_forests_1" src="https://github.com/user-attachments/assets/cd6dfa00-c11d-4623-9c01-0ca44399afc2" />

Each decision tree independently predicts the class of a new sample. The final output is determined by majority voting across all trees, which increases stability and accuracy compared to a single decision tree.

## b. Dataset Summary

We use a cleaned subset of the **Uber Pickup Data**, originally from Kaggle. The dataset includes GPS coordinates, passenger count, and time-of-day features. A binary classification label is created:  
- `1` if the fare amount is greater than $15 (high fare)  
- `0` otherwise (low fare)

Features used for training:
- Pickup/dropoff coordinates
- Passenger count
- Hour of day
- Day of week
- Manhattan distance
- Interaction terms (e.g., hour × passenger count)

## c. Instructions for Reproducing Results

1. Clone the GitHub repository:
   ```bash
   git clone https://github.com/your-username/INDE577_ML_Rice_2025.git
   ```

2. Open the corresponding Colab notebook (`random_forests.ipynb`) and mount your Google Drive to access the dataset.

3. Make sure the dataset file (`uber.csv`) is placed in:
   ```
   /content/drive/MyDrive/Colab Notebooks/ML_Rice_2025_Final_Project/
   ```

4. The model is imported directly from the custom-built `rice_ml` package:
   ```python
   from rice_ml.random_forest import RandomForest
   ```

5. After preprocessing and training, the notebook reports model metrics and visualizations including:
   - Accuracy
   - Precision
   - Recall
   - Confusion matrix

## Model Performance Summary

The model achieved:
- **Accuracy**: 93.8%
- **Precision**: 91.43%
- **Recall**: 71.91%

It demonstrates strong classification ability, especially in minimizing false positives for high-fare predictions.

---

This implementation showcases how ensemble methods like Random Forest can provide reliable and interpretable predictions on real-world datasets.
