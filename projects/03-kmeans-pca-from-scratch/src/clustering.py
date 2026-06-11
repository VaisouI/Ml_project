from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def generate_synthetic_clusters(
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], list[int]]:
    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [-6.0, -1.5],
            [-2.5, 3.8],
            [1.2, -2.0],
            [4.5, 3.2],
            [7.2, -0.8],
        ]
    )
    stds = [0.55, 0.85, 0.65, 1.10, 0.75]
    sizes = [100, 110, 90, 120, 80]

    parts: list[np.ndarray] = []
    labels: list[int] = []
    for cluster_id, (center, std, size) in enumerate(zip(centers, stds, sizes)):
        parts.append(rng.normal(loc=center, scale=std, size=(size, 2)))
        labels.extend([cluster_id] * size)

    X = np.vstack(parts)
    y_true = np.array(labels)
    idx = rng.permutation(len(X))
    return X[idx], y_true[idx], centers, stds, sizes


def pairwise_sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    C = np.asarray(C, dtype=float)
    x2 = np.sum(X**2, axis=1, keepdims=True)
    c2 = np.sum(C**2, axis=1)
    return np.maximum(x2 + c2 - 2 * X @ C.T, 0.0)


def pairwise_distances(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    sq = np.sum(X**2, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2 * X @ X.T, 0.0)
    return np.sqrt(d2)


def silhouette_score_manual(X: np.ndarray, labels: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) == len(X):
        return 0.0

    distances = pairwise_distances(X)
    scores = np.zeros(len(X))
    for i in range(len(X)):
        same_cluster = (labels == labels[i]).copy()
        same_cluster[i] = False
        a_i = distances[i, same_cluster].mean() if np.any(same_cluster) else 0.0

        b_i = np.inf
        for label in unique_labels:
            if label == labels[i]:
                continue
            mask = labels == label
            if np.any(mask):
                b_i = min(b_i, distances[i, mask].mean())

        denom = max(a_i, b_i)
        scores[i] = (b_i - a_i) / denom if denom > 0 else 0.0
    return float(np.mean(scores))


@dataclass
class KMeansRun:
    centers: np.ndarray
    labels: np.ndarray
    inertia: float
    n_iter: int
    init_centers: np.ndarray


class KMeans:
    """K-Means with random and K-Means++ initialization."""

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 300,
        tol: float = 1e-4,
        n_init: int = 10,
        init_method: str = "random",
        random_state: int | None = None,
    ) -> None:
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")
        if init_method not in ("random", "k-means++"):
            raise ValueError("init_method must be 'random' or 'k-means++'.")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.init_method = init_method
        self.random_state = random_state

        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int | None = None

    def _init_centroids_random(
        self,
        X: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        idx = rng.choice(len(X), size=self.n_clusters, replace=False)
        return X[idx].copy(), idx.copy()

    def _init_centroids_kmeanspp(
        self,
        X: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_samples = len(X)
        first_idx = int(rng.integers(n_samples))
        centroids = [X[first_idx].copy()]
        init_indices = [first_idx]

        while len(centroids) < self.n_clusters:
            current = np.vstack(centroids)
            distances = np.min(pairwise_sq_dists(X, current), axis=1)
            total = distances.sum()
            if total <= 0:
                remaining = list(set(range(n_samples)) - set(init_indices))
                next_idx = int(rng.choice(remaining))
            else:
                probs = distances / total
                next_idx = int(rng.choice(n_samples, p=probs))
            centroids.append(X[next_idx].copy())
            init_indices.append(next_idx)

        return np.vstack(centroids), np.array(init_indices)

    def _initialize(
        self,
        X: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.init_method == "random":
            return self._init_centroids_random(X, rng)
        return self._init_centroids_kmeanspp(X, rng)

    def _single_run(self, X: np.ndarray, rng: np.random.Generator) -> KMeansRun:
        centers, init_indices = self._initialize(X, rng)
        iteration = 0
        for iteration in range(1, self.max_iter + 1):
            distances = pairwise_sq_dists(X, centers)
            labels = np.argmin(distances, axis=1)
            new_centers = np.empty_like(centers)

            for cluster_id in range(self.n_clusters):
                mask = labels == cluster_id
                if np.any(mask):
                    new_centers[cluster_id] = X[mask].mean(axis=0)
                else:
                    new_centers[cluster_id] = X[int(rng.integers(len(X)))]

            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift < self.tol:
                break

        distances = pairwise_sq_dists(X, centers)
        labels = np.argmin(distances, axis=1)
        inertia = float(np.sum(distances[np.arange(len(X)), labels]))
        return KMeansRun(
            centers=centers.copy(),
            labels=labels.copy(),
            inertia=inertia,
            n_iter=iteration,
            init_centers=X[init_indices].copy(),
        )

    def fit(self, X: np.ndarray) -> "KMeans":
        X = np.asarray(X, dtype=float)
        base_rng = np.random.default_rng(self.random_state)

        best_run: KMeansRun | None = None
        for _ in range(self.n_init):
            seed = int(base_rng.integers(0, 1_000_000_000))
            run = self._single_run(X, np.random.default_rng(seed))
            if best_run is None or run.inertia < best_run.inertia:
                best_run = run

        assert best_run is not None
        self.cluster_centers_ = best_run.centers
        self.labels_ = best_run.labels
        self.inertia_ = best_run.inertia
        self.n_iter_ = best_run.n_iter
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise RuntimeError("Fit KMeans before calling predict().")
        distances = pairwise_sq_dists(np.asarray(X, dtype=float), self.cluster_centers_)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        assert self.labels_ is not None
        return self.labels_

