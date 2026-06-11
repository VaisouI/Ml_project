from __future__ import annotations

import numpy as np


class PCA:
    """Principal Component Analysis implemented with NumPy."""

    def __init__(self, n_components: int) -> None:
        if n_components <= 0:
            raise ValueError("n_components must be positive.")
        self.n_components = n_components
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.all_eigenvalues_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCA":
        X = np.asarray(X, dtype=float)
        if self.n_components > X.shape[1]:
            raise ValueError("n_components cannot exceed the number of features.")

        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        cov = (X_centered.T @ X_centered) / (len(X) - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)

        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        self.all_eigenvalues_ = eigvals.copy()
        self.components_ = eigvecs[:, : self.n_components].T
        self.explained_variance_ = eigvals[: self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / eigvals.sum()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("Fit PCA before calling transform().")
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_proj: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("Fit PCA before calling inverse_transform().")
        return np.asarray(X_proj, dtype=float) @ self.components_ + self.mean_

