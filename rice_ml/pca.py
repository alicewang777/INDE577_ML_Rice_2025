
import numpy as np

class PCA:
    def __init__(self, n_components):
        # Number of principal components to keep
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None

    def fit(self, X):
        # Step 1: Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Step 2: Compute covariance matrix
        cov = np.cov(X_centered, rowvar=False)

        # Step 3: Eigen decomposition of covariance matrix
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Step 4: Sort eigenvectors by decreasing eigenvalues
        sorted_idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, sorted_idx]
        eigvals = eigvals[sorted_idx]

        # Step 5: Keep top n_components
        self.components_ = eigvecs[:, :self.n_components]

    def transform(self, X):
        # Project data onto principal components
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
