
import numpy as np

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        # eps: neighborhood radius
        # min_samples: minimum number of points to form a dense region
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        n = len(X)
        labels = -1 * np.ones(n, dtype=int)  # -1 means noise
        cluster_id = 0
        visited = np.zeros(n, dtype=bool)

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = self._region_query(X, i)
            if len(neighbors) < self.min_samples:
                labels[i] = -1  # mark as noise
            else:
                self._expand_cluster(X, labels, i, neighbors, cluster_id, visited)
                cluster_id += 1

        self.labels_ = labels

    def _expand_cluster(self, X, labels, i, neighbors, cluster_id, visited):
        labels[i] = cluster_id
        queue = list(neighbors)
        while queue:
            j = queue.pop(0)
            if not visited[j]:
                visited[j] = True
                j_neighbors = self._region_query(X, j)
                if len(j_neighbors) >= self.min_samples:
                    queue.extend(j_neighbors)
            if labels[j] == -1:
                labels[j] = cluster_id

    def _region_query(self, X, idx):
        # Return indices of neighbors within eps distance
        distances = np.linalg.norm(X - X[idx], axis=1)
        return list(np.where(distances <= self.eps)[0])
