from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from datasets import generate_moons, generate_scaled_blobs, train_test_split
from neural_network import TwoLayerNet, binary_cross_entropy, relu, sigmoid


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    hidden_dim: int = 16,
    epochs: int = 1000,
    lr: float = 0.01,
    init_method: str = "random",
    activation: str = "relu",
    seed: int = 42,
    track_first_epoch_grad: bool = False,
) -> tuple[TwoLayerNet, np.ndarray, float, dict[str, float] | None]:
    model = TwoLayerNet(
        input_dim=X.shape[1],
        hidden_dim=hidden_dim,
        output_dim=1,
        init_method=init_method,  # type: ignore[arg-type]
        activation=activation,  # type: ignore[arg-type]
        seed=seed,
    )
    loss_history: list[float] = []
    first_epoch_grad: dict[str, float] | None = None

    for epoch in range(epochs):
        y_pred = model.forward(X)
        loss_history.append(model.compute_loss(y_pred, y))
        model.backward(X, y)

        if epoch == 0 and track_first_epoch_grad:
            snapshot = model.gradient_snapshot()
            first_epoch_grad = {
                "mean_abs_dW1": snapshot.mean_abs_dW1,
                "mean_abs_dW2": snapshot.mean_abs_dW2,
            }

        model.update(lr)

    return model, np.array(loss_history), model.accuracy(X, y), first_epoch_grad


def reference_gradient_check() -> dict[str, bool]:
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([[1.0], [0.0]])
    W1 = np.array([[0.1, 0.2], [0.3, 0.4]])
    b1 = np.array([[0.0, 0.0]])
    W2 = np.array([[0.5], [0.6]])
    b2 = np.array([[0.0]])

    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    y_hat = sigmoid(z2)
    loss = binary_cross_entropy(y_hat, y)

    n_samples = X.shape[0]
    dz2 = (y_hat - y) / n_samples
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)
    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0).astype(float)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    model = TwoLayerNet(2, 2, 1, init_method="zeros", activation="relu", seed=42)
    model.W1 = W1.copy()
    model.b1 = b1.copy()
    model.W2 = W2.copy()
    model.b2 = b2.copy()
    code_y_hat = model.forward(X)
    code_loss = model.compute_loss(code_y_hat, y)
    model.backward(X, y)

    return {
        "z1": bool(np.allclose(z1, model.z1)),
        "a1": bool(np.allclose(a1, model.a1)),
        "z2": bool(np.allclose(z2, model.z2)),
        "y_hat": bool(np.allclose(y_hat, model.y_pred)),
        "loss": bool(np.allclose(loss, code_loss)),
        "dW1": bool(np.allclose(dW1, model.dW1)),
        "db1": bool(np.allclose(db1, model.db1)),
        "dW2": bool(np.allclose(dW2, model.dW2)),
        "db2": bool(np.allclose(db2, model.db2)),
    }


def plot_decision_boundary(
    model: TwoLayerNet,
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    ax: plt.Axes,
) -> None:
    x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
    y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.forward(grid).reshape(xx.shape)

    ax.contourf(xx, yy, probs, levels=20, alpha=0.35)
    ax.contour(xx, yy, probs, levels=[0.5], linewidths=2)
    ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolors="k", s=25)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_all_experiments(
    output_dir: Path,
    seed: int = 42,
    quick: bool = False,
) -> dict[str, Any]:
    np.seterr(over="ignore", invalid="ignore")
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = {
        "lr": 150 if quick else 1000,
        "init": 150 if quick else 1000,
        "hidden": 400 if quick else 5000,
        "activation": 400 if quick else 2000,
    }

    gradient_check = reference_gradient_check()

    X_lr, y_lr = generate_scaled_blobs(n_samples=300, seed=seed)
    learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]
    lr_rows: list[dict[str, Any]] = []
    lr_histories: dict[float, np.ndarray] = {}
    for lr in learning_rates:
        _, history, acc, _ = train_model(
            X_lr,
            y_lr,
            hidden_dim=16,
            epochs=epochs["lr"],
            lr=lr,
            init_method="random",
            activation="relu",
            seed=seed,
        )
        lr_histories[lr] = history
        lr_rows.append(
            {
                "learning_rate": lr,
                "epochs": epochs["lr"],
                "final_loss": float(history[-1]),
                "train_accuracy": acc,
            }
        )

    write_csv(
        output_dir / "learning_rate_results.csv",
        lr_rows,
        ["learning_rate", "epochs", "final_loss", "train_accuracy"],
    )
    plt.figure(figsize=(10, 5))
    for lr, history in lr_histories.items():
        plt.plot(history, label=f"lr={lr}")
    plt.title("Сходимость при разных скоростях обучения")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "learning_rate_loss.png", dpi=160)
    plt.close()

    init_rows: list[dict[str, Any]] = []
    init_histories: dict[str, np.ndarray] = {}
    for init_method in ["zeros", "random"]:
        _, history, acc, first_epoch_grad = train_model(
            X_lr,
            y_lr,
            hidden_dim=16,
            epochs=epochs["init"],
            lr=0.01,
            init_method=init_method,
            activation="relu",
            seed=seed,
            track_first_epoch_grad=True,
        )
        assert first_epoch_grad is not None
        init_histories[init_method] = history
        init_rows.append(
            {
                "init_method": init_method,
                "epochs": epochs["init"],
                "final_loss": float(history[-1]),
                "train_accuracy": acc,
                "first_epoch_mean_abs_dW1": first_epoch_grad["mean_abs_dW1"],
                "first_epoch_mean_abs_dW2": first_epoch_grad["mean_abs_dW2"],
            }
        )

    write_csv(
        output_dir / "initialization_results.csv",
        init_rows,
        [
            "init_method",
            "epochs",
            "final_loss",
            "train_accuracy",
            "first_epoch_mean_abs_dW1",
            "first_epoch_mean_abs_dW2",
        ],
    )
    plt.figure(figsize=(9, 5))
    for init_method, history in init_histories.items():
        plt.plot(history, label=f"init={init_method}")
    plt.title("Влияние инициализации на сходимость")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "initialization_loss.png", dpi=160)
    plt.close()

    X_moons, y_moons = generate_moons(n_samples=150, noise=0.10, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X_moons,
        y_moons,
        test_size=0.2,
        seed=seed,
    )
    hidden_widths = [2, 4, 8, 16, 32, 64]
    hidden_rows: list[dict[str, Any]] = []
    hidden_models: dict[int, TwoLayerNet] = {}
    for hidden_dim in hidden_widths:
        model, history, train_acc, _ = train_model(
            X_train,
            y_train,
            hidden_dim=hidden_dim,
            epochs=epochs["hidden"],
            lr=0.1,
            init_method="random",
            activation="relu",
            seed=seed,
        )
        hidden_models[hidden_dim] = model
        hidden_rows.append(
            {
                "hidden_dim": hidden_dim,
                "epochs": epochs["hidden"],
                "final_loss": float(history[-1]),
                "train_accuracy": train_acc,
                "test_accuracy": model.accuracy(X_test, y_test),
            }
        )

    write_csv(
        output_dir / "hidden_width_results.csv",
        hidden_rows,
        ["hidden_dim", "epochs", "final_loss", "train_accuracy", "test_accuracy"],
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, hidden_dim in zip(axes.ravel(), hidden_widths):
        plot_decision_boundary(
            hidden_models[hidden_dim],
            X_train,
            y_train,
            title=f"hidden_dim={hidden_dim}",
            ax=ax,
        )
    plt.tight_layout()
    plt.savefig(output_dir / "hidden_width_boundaries.png", dpi=160)
    plt.close()

    X_act, y_act = generate_moons(n_samples=300, noise=0.15, seed=seed)
    activation_rows: list[dict[str, Any]] = []
    activation_histories: dict[str, np.ndarray] = {}
    for activation in ["relu", "sigmoid"]:
        model, history, train_acc, _ = train_model(
            X_act,
            y_act,
            hidden_dim=32,
            epochs=epochs["activation"],
            lr=0.1,
            init_method="random",
            activation=activation,
            seed=seed,
        )
        model.forward(X_act)
        model.backward(X_act, y_act)
        snapshot = model.gradient_snapshot()
        activation_histories[activation] = history
        activation_rows.append(
            {
                "activation": activation,
                "epochs": epochs["activation"],
                "final_loss": float(history[-1]),
                "train_accuracy": train_acc,
                "mean_abs_dW1": snapshot.mean_abs_dW1,
                "mean_abs_dW2": snapshot.mean_abs_dW2,
                "mean_abs_da1": snapshot.mean_abs_da1,
                "mean_abs_dz1": snapshot.mean_abs_dz1,
                "mean_activation_grad": snapshot.mean_activation_grad,
                "max_activation_grad": snapshot.max_activation_grad,
            }
        )

    write_csv(
        output_dir / "activation_results.csv",
        activation_rows,
        [
            "activation",
            "epochs",
            "final_loss",
            "train_accuracy",
            "mean_abs_dW1",
            "mean_abs_dW2",
            "mean_abs_da1",
            "mean_abs_dz1",
            "mean_activation_grad",
            "max_activation_grad",
        ],
    )
    plt.figure(figsize=(9, 5))
    for activation, history in activation_histories.items():
        plt.plot(history, label=activation)
    plt.title("Сходимость для разных функций активации")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "activation_loss.png", dpi=160)
    plt.close()

    return {
        "gradient_check": gradient_check,
        "learning_rate": lr_rows,
        "initialization": init_rows,
        "hidden_width": hidden_rows,
        "activation": activation_rows,
        "output_dir": str(output_dir),
        "quick": quick,
    }
