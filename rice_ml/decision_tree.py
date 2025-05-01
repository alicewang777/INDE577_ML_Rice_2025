
import numpy as np

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        y = np.array(y, dtype=int).flatten()  # 强制转为整数型 numpy array
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

    def _gini(self, y):
        classes = np.unique(y)
        impurity = 1.0
        for c in classes:
            p = np.sum(y == c) / len(y)
            impurity -= p ** 2
        return impurity

    def _best_split(self, X, y):
        best_idx, best_thresh, best_gain = None, None, 0
        n_samples, n_features = X.shape
        parent_impurity = self._gini(y)

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for t in thresholds:
                left = y[X[:, feature_idx] <= t]
                right = y[X[:, feature_idx] > t]
                if len(left) < self.min_samples_split or len(right) < self.min_samples_split:
                    continue
                left_impurity = self._gini(left)
                right_impurity = self._gini(right)
                weighted_impurity = (len(left) * left_impurity + len(right) * right_impurity) / len(y)
                gain = parent_impurity - weighted_impurity
                if gain > best_gain:
                    best_gain = gain
                    best_idx = feature_idx
                    best_thresh = t
        return best_idx, best_thresh

    def _build_tree(self, X, y, depth):
        y = np.array(y, dtype=int).flatten()  # 再次强制为整数型 numpy array

        if len(np.unique(y)) == 1 or depth >= self.max_depth or len(y) < self.min_samples_split:
            values, counts = np.unique(y, return_counts=True)
            majority_class = values[np.argmax(counts)]
            return {'leaf': True, 'label': majority_class}

        feat_idx, thresh = self._best_split(X, y)
        if feat_idx is None:
            values, counts = np.unique(y, return_counts=True)
            majority_class = values[np.argmax(counts)]
            return {'leaf': True, 'label': majority_class}

        left_mask = X[:, feat_idx] <= thresh
        right_mask = ~left_mask

        return {
            'leaf': False,
            'feature_index': feat_idx,
            'threshold': thresh,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def _predict_one(self, x, node):
        if node['leaf']:
            return node['label']
        if x[node['feature_index']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])

             
