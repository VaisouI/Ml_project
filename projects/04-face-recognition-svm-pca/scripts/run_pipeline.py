#!/usr/bin/env python
"""Reproducible face-classification pipeline with saved artifacts."""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default=str(PROJECT_ROOT / "data" / "persons_pics_train.zip"),
        help="Path to dataset ZIP/CSV",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "artifacts"),
        help="Directory for saved models and reports",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=37)
    parser.add_argument("--pca-threshold", type=float, default=0.95)
    parser.add_argument("--grid-verbose", type=int, default=0)
    return parser.parse_args()


def load_dataset(path: Path) -> tuple[pd.DataFrame, str]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            csv_members = [
                name
                for name in zf.namelist()
                if name.lower().endswith(".csv") and "__macosx" not in name.lower()
            ]
            if not csv_members:
                raise FileNotFoundError(f"No CSV file found in ZIP file: {path}")
            csv_name = csv_members[0]
            with zf.open(csv_name) as f:
                return pd.read_csv(f), csv_name
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path), path.name
    raise ValueError(f"Unsupported data format: {path.suffix}")


def round3(value: float) -> float:
    return float(np.round(value, 3))


def save_class_distribution_plot(
    distribution: pd.DataFrame, out_path: Path, title: str = "Class Distribution"
) -> None:
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(111)
    ax.bar(distribution["label"], distribution["count"], color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel("label")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_mean_faces_grid(means: pd.DataFrame, out_path: Path) -> None:
    labels = list(means.index)
    cols = 4
    rows = math.ceil(len(labels) / cols)
    fig = plt.figure(figsize=(12, rows * 3))
    for idx, label in enumerate(labels, start=1):
        ax = fig.add_subplot(rows, cols, idx)
        image = means.loc[label].to_numpy().reshape(62, 47)
        ax.imshow(image, cmap="gray")
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    reports_dir = output_dir / "reports"
    plots_dir = output_dir / "plots"

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df, source_name = load_dataset(data_path)
    if "label" not in df.columns:
        raise KeyError("Dataset must contain a 'label' column")

    feature_cols = [col for col in df.columns if col != "label"]
    X = df[feature_cols]
    y = df["label"]

    counts = y.value_counts().sort_index()
    shares = (counts / len(df)).rename("share")
    distribution = (
        pd.concat([counts.rename("count"), shares], axis=1)
        .reset_index()
        .rename(columns={"index": "label"})
    )
    distribution.to_csv(reports_dir / "class_distribution.csv", index=False)
    save_class_distribution_plot(distribution, plots_dir / "class_distribution.png")

    means = df.groupby("label")[feature_cols].mean()
    means.to_csv(reports_dir / "mean_vectors.csv")
    save_mean_faces_grid(means, plots_dir / "mean_faces_grid.png")

    n_classes = int(y.nunique())
    hugo_share = float(shares.loc["Hugo Chavez"])
    jacques_coord0 = float(means.loc["Jacques Chirac", "0"])
    cos_ariel_tony = float(
        cosine_similarity(
            means.loc["Ariel Sharon"].to_numpy().reshape(1, -1),
            means.loc["Tony Blair"].to_numpy().reshape(1, -1),
        )[0, 0]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    baseline = SVC(kernel="linear", random_state=args.random_state)
    baseline.fit(X_train, y_train)
    baseline_f1 = float(f1_score(y_test, baseline.predict(X_test), average="weighted"))
    joblib.dump(baseline, models_dir / "baseline_linear_svc.joblib")

    tuned_parameters = [
        {
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "gamma": [1e-3, 1e-4],
            "C": [1, 10, 100, 1000],
            "class_weight": [None, "balanced"],
            "random_state": [args.random_state],
        }
    ]

    t0 = perf_counter()
    cv_original = GridSearchCV(SVC(), tuned_parameters, refit=True, verbose=args.grid_verbose)
    cv_original.fit(X_train, y_train)
    grid_original_time_sec = float(perf_counter() - t0)
    best_original = cv_original.best_estimator_
    original_f1 = float(f1_score(y_test, best_original.predict(X_test), average="weighted"))
    joblib.dump(best_original, models_dir / "best_original_svc.joblib")
    pd.DataFrame(cv_original.cv_results_).to_csv(
        reports_dir / "grid_original_cv_results.csv", index=False
    )

    pca_full = PCA(svd_solver="full")
    pca_full.fit(X_train)
    explained_cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_95 = int(np.searchsorted(explained_cumsum, args.pca_threshold) + 1)

    pca_95 = PCA(n_components=n_components_95, svd_solver="full")
    X_train_pca = pca_95.fit_transform(X_train)
    X_test_pca = pca_95.transform(X_test)

    t1 = perf_counter()
    cv_pca = GridSearchCV(SVC(), tuned_parameters, refit=True, verbose=args.grid_verbose)
    cv_pca.fit(X_train_pca, y_train)
    grid_pca_time_sec = float(perf_counter() - t1)
    best_pca = cv_pca.best_estimator_
    pca_f1 = float(f1_score(y_test, best_pca.predict(X_test_pca), average="weighted"))

    joblib.dump(pca_full, models_dir / "pca_full_train.joblib")
    joblib.dump(pca_95, models_dir / "pca_95_transform.joblib")
    joblib.dump(best_pca, models_dir / "best_pca_svc.joblib")
    pd.DataFrame(cv_pca.cv_results_).to_csv(reports_dir / "grid_pca_cv_results.csv", index=False)

    pipeline = Pipeline([("pca", pca_95), ("svc", best_pca)])
    joblib.dump(pipeline, models_dir / "best_pca_pipeline.joblib")

    answers_full = {
        "dataset_source": str(data_path),
        "dataset_member": source_name,
        "n_rows": int(df.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": n_classes,
        "hugo_chavez_share": hugo_share,
        "jacques_chirac_mean_coord_0": jacques_coord0,
        "cosine_similarity_ariel_sharon_tony_blair": cos_ariel_tony,
        "identified_person_on_prompt_average_image": "Jean Chretien",
        "baseline_linear_f1_weighted": baseline_f1,
        "grid_original_best_params": cv_original.best_params_,
        "grid_original_f1_weighted": original_f1,
        "pca_n_components_for_explained_variance_gt_threshold": n_components_95,
        "pca_threshold": float(args.pca_threshold),
        "grid_pca_best_params": cv_pca.best_params_,
        "grid_pca_f1_weighted": pca_f1,
        "grid_original_time_sec": grid_original_time_sec,
        "grid_pca_time_sec": grid_pca_time_sec,
        "explained_variance_cumsum_at_n_components_95": float(
            explained_cumsum[n_components_95 - 1]
        ),
    }

    answers_rounded = {
        "n_classes": n_classes,
        "hugo_chavez_share_3dp": round3(hugo_share),
        "jacques_chirac_mean_coord_0_3dp": round3(jacques_coord0),
        "cosine_similarity_ariel_sharon_tony_blair_3dp": round3(cos_ariel_tony),
        "baseline_linear_f1_weighted_3dp": round3(baseline_f1),
        "grid_original_best_C": cv_original.best_params_["C"],
        "grid_original_best_gamma": cv_original.best_params_["gamma"],
        "grid_original_best_kernel": cv_original.best_params_["kernel"],
        "grid_original_f1_weighted_3dp": round3(original_f1),
        "pca_n_components_for_explained_variance_gt_0_95": n_components_95,
        "grid_pca_best_C": cv_pca.best_params_["C"],
        "grid_pca_best_gamma": cv_pca.best_params_["gamma"],
        "grid_pca_best_kernel": cv_pca.best_params_["kernel"],
        "grid_pca_f1_weighted_3dp": round3(pca_f1),
        "identified_person_on_prompt_average_image": "Jean Chretien",
    }

    with (reports_dir / "answers_full_precision.json").open("w", encoding="utf-8") as f:
        json.dump(answers_full, f, ensure_ascii=False, indent=2)

    with (reports_dir / "answers_rounded.json").open("w", encoding="utf-8") as f:
        json.dump(answers_rounded, f, ensure_ascii=False, indent=2)

    summary_lines = [
        "Face recognition SVM/PCA pipeline summary",
        f"dataset: {data_path}",
        f"rows/features: {df.shape[0]}/{X.shape[1]}",
        f"n_classes: {n_classes}",
        f"hugo_share: {hugo_share:.6f}",
        f"jacques_coord0: {jacques_coord0:.6f}",
        f"baseline_linear_f1_weighted: {baseline_f1:.6f}",
        f"grid_original_best_params: {cv_original.best_params_}",
        f"grid_original_f1_weighted: {original_f1:.6f}",
        f"pca_components_for_>{args.pca_threshold}: {n_components_95}",
        f"grid_pca_best_params: {cv_pca.best_params_}",
        f"grid_pca_f1_weighted: {pca_f1:.6f}",
        f"cosine(Ariel Sharon, Tony Blair): {cos_ariel_tony:.6f}",
        "identified_person_on_prompt_average_image: Jean Chretien",
    ]
    (reports_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("Artifacts saved to:", output_dir)
    print("Rounded answers:")
    print(json.dumps(answers_rounded, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
