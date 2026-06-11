from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


ActivationName = Literal["relu", "sigmoid"]
InitMethod = Literal["zeros", "random"]


def ensure_column(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        return y.reshape(-1, 1)
    if y.ndim == 2 and y.shape[1] == 1:
        return y
    raise ValueError("Expected a 1D target vector or a single-column 2D array.")


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad_from_output(sigmoid_output: np.ndarray) -> np.ndarray:
    return sigmoid_output * (1.0 - sigmoid_output)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)


def binary_cross_entropy(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    eps: float = 1e-12,
) -> float:
    y_true = ensure_column(y_true)
    y_pred = ensure_column(y_pred)
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    loss = y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)
    return float(-np.mean(loss))


@dataclass
class GradientSnapshot:
    mean_abs_dW1: float
    mean_abs_dW2: float
    mean_abs_da1: float | None = None
    mean_abs_dz1: float | None = None
    max_activation_grad: float | None = None
    mean_activation_grad: float | None = None


class TwoLayerNet:
    """Two-layer binary classifier with explicit forward and backward passes."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        init_method: InitMethod = "random",
        activation: ActivationName = "relu",
        seed: int = 42,
    ) -> None:
        if init_method not in ("zeros", "random"):
            raise ValueError("init_method must be 'zeros' or 'random'.")
        if activation not in ("relu", "sigmoid"):
            raise ValueError("activation must be 'relu' or 'sigmoid'.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.init_method = init_method
        self.activation = activation

        rng = np.random.default_rng(seed)
        if init_method == "zeros":
            self.W1 = np.zeros((input_dim, hidden_dim))
            self.W2 = np.zeros((hidden_dim, output_dim))
        else:
            self.W1 = rng.normal(0.0, 0.01, size=(input_dim, hidden_dim))
            self.W2 = rng.normal(0.0, 0.01, size=(hidden_dim, output_dim))

        self.b1 = np.zeros((1, hidden_dim))
        self.b2 = np.zeros((1, output_dim))

        self.X: np.ndarray | None = None
        self.z1: np.ndarray | None = None
        self.a1: np.ndarray | None = None
        self.z2: np.ndarray | None = None
        self.y_pred: np.ndarray | None = None

        self.dW1: np.ndarray | None = None
        self.db1: np.ndarray | None = None
        self.dW2: np.ndarray | None = None
        self.db2: np.ndarray | None = None
        self.dz2: np.ndarray | None = None
        self.da1: np.ndarray | None = None
        self.dz1: np.ndarray | None = None

    def _activation_forward(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return relu(z)
        return sigmoid(z)

    def _activation_backward(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return relu_grad(z)
        return sigmoid_grad_from_output(a)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = np.asarray(X, dtype=float)
        self.z1 = self.X @ self.W1 + self.b1
        self.a1 = self._activation_forward(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.y_pred = sigmoid(self.z2)
        return self.y_pred

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return binary_cross_entropy(y_pred, y_true)

    def backward(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        y_true = ensure_column(y_true)
        if self.y_pred is None or self.X is None or not np.array_equal(self.X, X):
            self.forward(X)

        assert self.X is not None
        assert self.z1 is not None
        assert self.a1 is not None
        assert self.y_pred is not None

        n_samples = X.shape[0]
        self.dz2 = (self.y_pred - y_true) / n_samples
        self.dW2 = self.a1.T @ self.dz2
        self.db2 = np.sum(self.dz2, axis=0, keepdims=True)

        self.da1 = self.dz2 @ self.W2.T
        self.dz1 = self.da1 * self._activation_backward(self.z1, self.a1)
        self.dW1 = self.X.T @ self.dz1
        self.db1 = np.sum(self.dz1, axis=0, keepdims=True)
        return self.dW1, self.db1, self.dW2, self.db2

    def update(self, lr: float) -> None:
        if self.dW1 is None or self.db1 is None or self.dW2 is None or self.db2 is None:
            raise RuntimeError("Run backward() before update().")
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def accuracy(self, X: np.ndarray, y_true: np.ndarray) -> float:
        y_true = ensure_column(y_true).astype(int)
        return float(np.mean(self.predict(X) == y_true))

    def gradient_snapshot(self) -> GradientSnapshot:
        if self.dW1 is None or self.dW2 is None:
            raise RuntimeError("Run backward() before requesting gradients.")

        snapshot = GradientSnapshot(
            mean_abs_dW1=float(np.mean(np.abs(self.dW1))),
            mean_abs_dW2=float(np.mean(np.abs(self.dW2))),
        )
        if self.da1 is not None:
            snapshot.mean_abs_da1 = float(np.mean(np.abs(self.da1)))
        if self.dz1 is not None:
            snapshot.mean_abs_dz1 = float(np.mean(np.abs(self.dz1)))
        if self.z1 is not None and self.a1 is not None:
            activation_grad = self._activation_backward(self.z1, self.a1)
            snapshot.max_activation_grad = float(np.max(activation_grad))
            snapshot.mean_activation_grad = float(np.mean(activation_grad))
        return snapshot

