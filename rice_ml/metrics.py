
import numpy as np

# ✅ Classification tasks
def accuracy(y_true, y_pred):
    """Compute classification accuracy."""
    return np.mean(y_true == y_pred)

# ✅ Regression tasks
def mse(y_true, y_pred):
    """Mean Squared Error (MSE)."""
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    """Coefficient of determination (R²)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot
