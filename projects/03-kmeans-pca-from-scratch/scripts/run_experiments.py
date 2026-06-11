from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiments import run_all_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run K-Means and PCA experiments implemented with NumPy.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "generated",
        help="Directory for generated CSV tables and plots.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a shorter smoke-test version of the experiments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_all_experiments(args.output_dir, seed=args.seed, quick=args.quick)

    print("K-Means/PCA experiment summary")
    print(f"Output directory: {summary['output_dir']}")
    print(f"Best K by silhouette: {summary['best_k']}")
    print(f"PCA components for >=90% variance: {summary['k90']}")

    print("\nInitialization summary:")
    for method, stats in summary["initialization_summary"].items():
        print(
            f"  {method:<9} "
            f"mean_inertia={stats['mean_inertia']:.3f} "
            f"std_inertia={stats['std_inertia']:.3f} "
            f"mean_iter={stats['mean_iter']:.3f}"
        )

    print("\nDigits clustering:")
    for row in summary["digits_clustering"]:
        print(
            f"  {row['space']:<10} "
            f"inertia={row['inertia']:.3f} "
            f"silhouette={row['silhouette']:.4f}"
        )


if __name__ == "__main__":
    main()

