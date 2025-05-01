
import numpy as np
from rice_ml.decision_tree import DecisionTree

class AdaBoost:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        y_mapped = np.where(y == 0, -1, 1)
        n_samples = len(y)
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            stump = DecisionTree(max_depth=1)
            stump.fit(X, y)  

            y_pred = stump.predict(X)
            y_pred = np.array(y_pred).astype(int)  
            y_pred_mapped = np.where(y_pred == 0, -1, 1)

            error = np.sum(weights[y_pred_mapped != y_mapped])
            error = max(error, 1e-10)
            alpha = 0.5 * np.log((1 - error) / error)

            weights *= np.exp(-alpha * y_mapped * y_pred_mapped)
            weights /= np.sum(weights)

            self.models.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        clf_preds = np.array([
            alpha * np.where(model.predict(X) == 0, -1, 1)
            for model, alpha in zip(self.models, self.alphas)
        ])
        final_pred = np.sign(np.sum(clf_preds, axis=0))
        return np.where(final_pred == -1, 0, 1)


