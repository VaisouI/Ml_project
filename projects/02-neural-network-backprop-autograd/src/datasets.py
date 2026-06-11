from __future__ import annotations

import numpy as np


def generate_scaled_blobs(
    n_samples: int = 300,
    seed: int = 42,
    feature_scale: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n0 = n_samples // 2
    n1 = n_samples - n0
    cov = np.array([[0.6, 0.15], [0.15, 0.6]])

    x0 = rng.multivariate_normal(mean=[-1.5, -1.0], cov=cov, size=n0)
    x1 = rng.multivariate_normal(mean=[1.5, 1.0], cov=cov, size=n1)
    X = np.vstack([x0, x1]) * feature_scale
    y = np.vstack([np.zeros((n0, 1)), np.ones((n1, 1))])

    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


def generate_moons(
    n_samples: int = 300,
    noise: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n0 = n_samples // 2
    n1 = n_samples - n0

    theta0 = rng.uniform(0.0, np.pi, n0)
    theta1 = rng.uniform(0.0, np.pi, n1)
    moon0 = np.column_stack([np.cos(theta0), np.sin(theta0)])
    moon1 = np.column_stack([1.0 - np.cos(theta1), -np.sin(theta1) + 0.5])

    X = np.vstack([moon0, moon1])
    X += rng.normal(0.0, noise, size=X.shape)
    y = np.vstack([np.zeros((n0, 1)), np.ones((n1, 1))])

    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_samples = len(X)
    indices = rng.permutation(n_samples)
    test_n = int(round(n_samples * test_size))
    test_idx = indices[:test_n]
    train_idx = indices[test_n:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

