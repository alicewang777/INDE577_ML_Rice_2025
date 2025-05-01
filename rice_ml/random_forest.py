
import numpy as np
from rice_ml.decision_tree import DecisionTree
from collections import Counter

class RandomForest:
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, sample_ratio=0.8):
        # Initialize forest parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.sample_ratio = sample_ratio
        self.trees = []

    def _bootstrap_sample(self, X, y):
        # Randomly sample data with replacement
        n_samples = int(len(X) * self.sample_ratio)
        indices = np.random.choice(len(X), n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        # Train each tree on a bootstrap sample
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Aggregate predictions from all trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = tree_preds.T  # shape: [n_samples, n_trees]
        return np.array([Counter(row).most_common(1)[0][0] for row in tree_preds])
