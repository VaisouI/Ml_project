from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

from clustering import KMeans, generate_synthetic_clusters, silhouette_score_manual
from dimensionality import PCA


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, float]:
    inertias = np.array([run["inertia"] for run in runs])
    iterations = np.array([run["n_iter"] for run in runs])
    return {
        "mean_inertia": float(inertias.mean()),
        "std_inertia": float(inertias.std(ddof=0)),
        "mean_iter": float(iterations.mean()),
        "std_iter": float(iterations.std(ddof=0)),
    }


def run_init_trials(
    X: np.ndarray,
    method: str,
    runs: int,
    n_clusters: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_id in range(runs):
        model = KMeans(
            n_clusters=n_clusters,
            n_init=1,
            init_method=method,
            random_state=seed + run_id,
        ).fit(X)
        rows.append(
            {
                "run": run_id + 1,
                "method": method,
                "inertia": float(model.inertia_),
                "n_iter": int(model.n_iter_),
            }
        )
    return rows


def plot_cluster_result(
    X: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    title: str,
    path: Path,
) -> None:
    plt.figure(figsize=(6, 4.5))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=22, alpha=0.8)
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        c="black",
        marker="X",
        s=150,
        edgecolor="white",
        linewidth=1.2,
    )
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def run_all_experiments(
    output_dir: Path,
    seed: int = 42,
    quick: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    X_syn, _, _, _, _ = generate_synthetic_clusters(seed=seed)
    n_init = 3 if quick else 10
    init_runs = 5 if quick else 20

    k_values = list(range(2, 7 if quick else 11))
    elbow_rows: list[dict[str, Any]] = []
    for k in k_values:
        model = KMeans(
            n_clusters=k,
            n_init=n_init,
            init_method="k-means++",
            random_state=seed,
        ).fit(X_syn)
        assert model.labels_ is not None
        elbow_rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": silhouette_score_manual(X_syn, model.labels_),
                "n_iter": int(model.n_iter_),
            }
        )

    write_csv(
        output_dir / "synthetic_k_selection.csv",
        elbow_rows,
        ["k", "inertia", "silhouette", "n_iter"],
    )

    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.plot([row["k"] for row in elbow_rows], [row["inertia"] for row in elbow_rows], marker="o")
    ax1.set_xlabel("K")
    ax1.set_ylabel("Inertia")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(
        [row["k"] for row in elbow_rows],
        [row["silhouette"] for row in elbow_rows],
        color="tab:orange",
        marker="s",
    )
    ax2.set_ylabel("Silhouette")
    plt.title("Выбор числа кластеров")
    plt.tight_layout()
    plt.savefig(output_dir / "synthetic_k_selection.png", dpi=160)
    plt.close()

    best_k = max(elbow_rows, key=lambda row: row["silhouette"])["k"]
    best_model = KMeans(
        n_clusters=int(best_k),
        n_init=10,
        init_method="k-means++",
        random_state=seed,
    ).fit(X_syn)
    assert best_model.labels_ is not None
    assert best_model.cluster_centers_ is not None
    plot_cluster_result(
        X_syn,
        best_model.labels_,
        best_model.cluster_centers_,
        f"K-Means++ result, K={best_k}",
        output_dir / "synthetic_clusters.png",
    )

    init_rows = (
        run_init_trials(X_syn, "random", init_runs, 5, seed)
        + run_init_trials(X_syn, "k-means++", init_runs, 5, seed + 10_000)
    )
    write_csv(
        output_dir / "initialization_comparison.csv",
        init_rows,
        ["run", "method", "inertia", "n_iter"],
    )

    plt.figure(figsize=(7, 4.5))
    for method in ["random", "k-means++"]:
        values = [row["inertia"] for row in init_rows if row["method"] == method]
        plt.hist(values, bins=10, alpha=0.65, label=method)
    plt.xlabel("Inertia")
    plt.ylabel("Число запусков")
    plt.title("Распределение inertia по запускам")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "initialization_inertia_hist.png", dpi=160)
    plt.close()

    digits = load_digits()
    X_digits = digits.data.astype(float)
    y_digits = digits.target
    if quick:
        X_digits = X_digits[:500]
        y_digits = y_digits[:500]

    pca_full = PCA(n_components=X_digits.shape[1]).fit(X_digits)
    assert pca_full.all_eigenvalues_ is not None
    eigenvalues = pca_full.all_eigenvalues_
    cumulative_ratio = np.cumsum(eigenvalues) / eigenvalues.sum()
    k90 = int(np.argmax(cumulative_ratio >= 0.90) + 1)

    variance_rows = [
        {
            "component": i + 1,
            "eigenvalue": float(eigenvalues[i]),
            "cumulative_explained_variance": float(cumulative_ratio[i]),
        }
        for i in range(len(eigenvalues))
    ]
    write_csv(
        output_dir / "pca_explained_variance.csv",
        variance_rows,
        ["component", "eigenvalue", "cumulative_explained_variance"],
    )

    plt.figure(figsize=(7, 4.5))
    plt.plot(np.arange(1, len(cumulative_ratio) + 1), cumulative_ratio, marker="o", markersize=3)
    plt.axhline(0.90, color="red", linestyle="--", label="90%")
    plt.axvline(k90, color="green", linestyle="--", label=f"k={k90}")
    plt.xlabel("Число компонент")
    plt.ylabel("Накопленная объясненная дисперсия")
    plt.title("PCA explained variance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "pca_explained_variance.png", dpi=160)
    plt.close()

    pca_90 = PCA(n_components=k90).fit(X_digits)
    X_90 = pca_90.transform(X_digits)
    pca_2 = PCA(n_components=2).fit(X_digits)
    X_2 = pca_2.transform(X_digits)
    datasets = {
        "64D": X_digits,
        f"{k90}D_90pct": X_90,
        "2D": X_2,
    }

    digit_rows: list[dict[str, Any]] = []
    labels_2d: np.ndarray | None = None
    centers_2d: np.ndarray | None = None
    for name, X_cur in datasets.items():
        model = KMeans(
            n_clusters=10,
            n_init=n_init,
            init_method="k-means++",
            random_state=seed,
        ).fit(X_cur)
        assert model.labels_ is not None
        digit_rows.append(
            {
                "space": name,
                "inertia": float(model.inertia_),
                "silhouette": silhouette_score_manual(X_cur, model.labels_),
                "n_iter": int(model.n_iter_),
            }
        )
        if name == "2D":
            labels_2d = model.labels_
            centers_2d = model.cluster_centers_

    write_csv(
        output_dir / "digits_clustering_results.csv",
        digit_rows,
        ["space", "inertia", "silhouette", "n_iter"],
    )

    assert labels_2d is not None
    assert centers_2d is not None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].scatter(X_2[:, 0], X_2[:, 1], c=labels_2d, cmap="tab10", s=18)
    axes[0].scatter(centers_2d[:, 0], centers_2d[:, 1], c="black", marker="X", s=150)
    axes[0].set_title("K-Means в 2D PCA-пространстве")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[1].scatter(X_2[:, 0], X_2[:, 1], c=y_digits, cmap="tab10", s=18)
    axes[1].set_title("Истинные классы в 2D PCA-пространстве")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    plt.tight_layout()
    plt.savefig(output_dir / "digits_pca_kmeans.png", dpi=160)
    plt.close()

    return {
        "k_selection": elbow_rows,
        "best_k": best_k,
        "initialization_summary": {
            "random": summarize_runs([row for row in init_rows if row["method"] == "random"]),
            "k-means++": summarize_runs([row for row in init_rows if row["method"] == "k-means++"]),
        },
        "k90": k90,
        "digits_clustering": digit_rows,
        "output_dir": str(output_dir),
        "quick": quick,
    }

