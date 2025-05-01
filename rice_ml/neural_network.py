
import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_dim, hidden_dim=10, learning_rate=0.01, n_iters=1000):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.n_iters = n_iters
        self._init_weights()

    def _init_weights(self):
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * 0.01
        self.b1 = np.zeros((1, self.hidden_dim))
        self.W2 = np.random.randn(self.hidden_dim, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        for _ in range(self.n_iters):
            # Forward pass
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = self.relu(Z1)
            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = self.sigmoid(Z2)

            # Backward pass
            dZ2 = A2 - y
            dW2 = np.dot(A1.T, dZ2) / X.shape[0]
            db2 = np.mean(dZ2, axis=0, keepdims=True)

            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * self.relu_derivative(Z1)
            dW1 = np.dot(X.T, dZ1) / X.shape[0]
            db1 = np.mean(dZ1, axis=0, keepdims=True)

            # Gradient descent
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

    def predict(self, X):
        A1 = self.relu(np.dot(X, self.W1) + self.b1)
        A2 = self.sigmoid(np.dot(A1, self.W2) + self.b2)
        return (A2 >= 0.5).astype(int).flatten()
