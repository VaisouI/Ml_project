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
        description="Run manual backpropagation experiments for a two-layer network.",
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

    print("Manual backpropagation experiment summary")
    print(f"Output directory: {summary['output_dir']}")
    print(f"Gradient check passed: {all(summary['gradient_check'].values())}")

    print("\nLearning-rate sweep:")
    for row in summary["learning_rate"]:
        print(
            f"  lr={row['learning_rate']:<5} "
            f"loss={row['final_loss']:.4f} "
            f"acc={row['train_accuracy']:.4f}"
        )

    print("\nInitialization:")
    for row in summary["initialization"]:
        print(
            f"  {row['init_method']:<6} "
            f"loss={row['final_loss']:.4f} "
            f"acc={row['train_accuracy']:.4f}"
        )

    print("\nHidden width:")
    for row in summary["hidden_width"]:
        print(
            f"  hidden_dim={row['hidden_dim']:<2} "
            f"train_acc={row['train_accuracy']:.4f} "
            f"test_acc={row['test_accuracy']:.4f}"
        )

    print("\nActivation:")
    for row in summary["activation"]:
        print(
            f"  {row['activation']:<7} "
            f"loss={row['final_loss']:.4f} "
            f"acc={row['train_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()

