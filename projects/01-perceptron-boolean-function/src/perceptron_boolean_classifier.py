from __future__ import annotations

import itertools
import math
import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ETA = 0.3
MAX_EPOCHS = 1000
KNOWN_MINIMAL_SUBSETS = {
    "threshold": (0, 1, 7, 11, 14),
    "smooth": (1, 2, 3, 13),
}


def generate_binary_vectors() -> np.ndarray:
    """Генерирует все 16 бинарных векторов (x1, x2, x3, x4) в лексикографическом порядке."""
    vectors = list(itertools.product([0, 1], repeat=4))
    return np.array(vectors, dtype=int)


def boolean_function_variant_25(x1: int, x2: int, x3: int, x4: int) -> int:
    """
    Булева функция варианта 25:
    F(x1, x2, x3, x4) = NOT( (x1 AND x2) OR (NOT x3) OR (NOT x4) )
    """
    value = not ((x1 and x2) or (not x3) or (not x4))
    return int(value)


def compute_targets(x: np.ndarray) -> np.ndarray:
    """Вычисляет целевой вектор T из формулы (без хардкода)."""
    targets = [boolean_function_variant_25(*row) for row in x]
    return np.array(targets, dtype=int)


def activation_threshold(net: float) -> Tuple[float, int, float]:
    """
    Пороговая функция активации.
    Возвращает:
    out - непрерывный выход (здесь совпадает с y),
    y   - бинарный выход,
    dfdnet - производная (для унификации интерфейса; для пороговой не используется).
    """
    out = 1.0 if net >= 0.0 else 0.0
    y = int(out)
    dfdnet = 1.0
    return out, y, dfdnet


def activation_smooth(net: float) -> Tuple[float, int, float]:
    """
    f(net) = 1/2 * ( net / (1 + |net|) + 1 )
    df/dnet = 1 / ( 2 * (1 + |net|)^2 )
    """
    out = 0.5 * (net / (1.0 + abs(net)) + 1.0)
    y = 1 if out >= 0.5 else 0
    dfdnet = 1.0 / (2.0 * (1.0 + abs(net)) ** 2)
    return out, y, dfdnet


def get_activation(name: str):
    if name == "threshold":
        return activation_threshold
    if name == "smooth":
        return activation_smooth
    raise ValueError(f"Неизвестная функция активации: {name}")


def predict_single(x_row: np.ndarray, weights: np.ndarray, activation_name: str) -> int:
    """Прямой проход для одного вектора."""
    x_ext = np.concatenate(([1.0], x_row.astype(float)))
    net = float(np.dot(weights, x_ext))
    _, y, _ = get_activation(activation_name)(net)
    return y


def evaluate_on_all(
    x_all: np.ndarray, t_all: np.ndarray, weights: np.ndarray, activation_name: str
) -> Tuple[np.ndarray, int]:
    """Оценивает модель на всех 16 векторах и считает ошибку Хэмминга E(k)."""
    y_pred = np.array([predict_single(row, weights, activation_name) for row in x_all], dtype=int)
    error = int(np.sum(y_pred != t_all))
    return y_pred, error


def bits_to_string(bits: Sequence[int]) -> str:
    return "".join(str(int(b)) for b in bits)


def train_network(
    x_all: np.ndarray,
    t_all: np.ndarray,
    train_indices: Sequence[int],
    activation_name: str,
    eta: float = ETA,
    max_epochs: int = MAX_EPOCHS,
) -> Dict[str, object]:
    """
    Обучает один нейрон по правилу Видроу-Хоффа на указанном подмножестве.
    После каждой эпохи оценивает качество на всех 16 векторах.
    """
    weights = np.zeros(5, dtype=float)  # [w0, w1, w2, w3, w4]
    activation = get_activation(activation_name)

    history_rows: List[Dict[str, object]] = []
    converged = False
    y_final = np.zeros_like(t_all)
    e_final = int(len(t_all))

    for epoch in range(1, max_epochs + 1):
        for idx in train_indices:
            x_row = x_all[idx]
            target = int(t_all[idx])
            x_ext = np.concatenate(([1.0], x_row.astype(float)))
            net = float(np.dot(weights, x_ext))
            out, y, dfdnet = activation(net)
            delta = target - y

            if activation_name == "threshold":
                delta_w = eta * delta * x_ext
            else:
                delta_w = eta * delta * dfdnet * x_ext

            weights += delta_w

        y_eval, e_eval = evaluate_on_all(x_all, t_all, weights, activation_name)
        history_rows.append(
            {
                "epoch": epoch,
                "w0": weights[0],
                "w1": weights[1],
                "w2": weights[2],
                "w3": weights[3],
                "w4": weights[4],
                "Y(k)": bits_to_string(y_eval),
                "E(k)": e_eval,
            }
        )

        y_final = y_eval
        e_final = e_eval
        if e_eval == 0:
            converged = True
            break

    history_df = pd.DataFrame(history_rows)
    return {
        "weights": weights.copy(),
        "history": history_df,
        "converged": converged,
        "epochs": int(len(history_rows)),
        "y_final": y_final.copy(),
        "e_final": int(e_final),
        "train_indices": tuple(int(i) for i in train_indices),
    }


def find_minimal_training_subset(
    x_all: np.ndarray,
    t_all: np.ndarray,
    activation_name: str,
    eta: float = ETA,
    max_epochs: int = MAX_EPOCHS,
) -> Optional[Dict[str, object]]:
    """
    Детерминированно ищет минимальное обучающее подмножество:
    по возрастанию размера и в лексикографическом порядке индексов.
    """
    n = len(x_all)
    for subset_size in range(1, n + 1):
        for subset in itertools.combinations(range(n), subset_size):
            result = train_network(
                x_all=x_all,
                t_all=t_all,
                train_indices=subset,
                activation_name=activation_name,
                eta=eta,
                max_epochs=max_epochs,
            )
            if result["converged"]:
                return result
    return None


def build_truth_table(x_all: np.ndarray, t_all: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(x_all, columns=["x1", "x2", "x3", "x4"])
    df["F"] = t_all
    return df


def build_verification_table(
    x_all: np.ndarray, t_all: np.ndarray, y_pred: np.ndarray
) -> pd.DataFrame:
    df = pd.DataFrame(x_all, columns=["x1", "x2", "x3", "x4"])
    df["T"] = t_all
    df["Y"] = y_pred
    df["match"] = (df["T"] == df["Y"]).astype(int)
    return df


def plot_error(history_df: pd.DataFrame, title: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["E(k)"], marker="o")
    plt.xlabel("Номер эпохи")
    plt.ylabel("Суммарная ошибка E(k)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def format_weights(weights: np.ndarray) -> str:
    return "[" + ", ".join(f"{w:.6f}" for w in weights) + "]"


def activation_label(activation_name: str) -> str:
    if activation_name == "threshold":
        return "Пороговая"
    if activation_name == "smooth":
        return "f(net)=1/2*(net/(1+|net|)+1)"
    return activation_name


def save_result_tables(
    output_dir: Path,
    prefix: str,
    result: Dict[str, object],
    x_all: np.ndarray,
    t_all: np.ndarray,
) -> None:
    history_df = result["history"]
    history_df.to_csv(output_dir / f"{prefix}_history.csv", index=False, encoding="utf-8-sig")

    verification_df = build_verification_table(x_all, t_all, result["y_final"])
    verification_df.to_csv(
        output_dir / f"{prefix}_verification.csv", index=False, encoding="utf-8-sig"
    )


def build_comparison_table(
    full_results: Dict[str, Dict[str, object]],
    min_results: Dict[str, Optional[Dict[str, object]]],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for activation_name, result in full_results.items():
        rows.append(
            {
                "Функция активации": activation_label(activation_name),
                "Режим обучения": "Полная выборка",
                "Размер обучающей выборки": len(result["train_indices"]),
                "Число эпох": result["epochs"],
                "Итоговые веса [w0..w4]": format_weights(result["weights"]),
                "Достигнута E=0": "Да" if result["converged"] else "Нет",
            }
        )

    for activation_name, result in min_results.items():
        if result is None:
            rows.append(
                {
                    "Функция активации": activation_label(activation_name),
                    "Режим обучения": "Минимальная выборка",
                    "Размер обучающей выборки": "не найдено",
                    "Число эпох": "не найдено",
                    "Итоговые веса [w0..w4]": "не найдено",
                    "Достигнута E=0": "Нет",
                }
            )
        else:
            rows.append(
                {
                    "Функция активации": activation_label(activation_name),
                    "Режим обучения": "Минимальная выборка",
                    "Размер обучающей выборки": len(result["train_indices"]),
                    "Число эпох": result["epochs"],
                    "Итоговые веса [w0..w4]": format_weights(result["weights"]),
                    "Достигнута E=0": "Да" if result["converged"] else "Нет",
                }
            )

    return pd.DataFrame(rows)


def build_report_text(
    truth_df: pd.DataFrame,
    full_results: Dict[str, Dict[str, object]],
    min_results: Dict[str, Optional[Dict[str, object]]],
    comparison_df: pd.DataFrame,
) -> str:
    ones_count = int(truth_df["F"].sum())
    zeros_count = int(len(truth_df) - ones_count)

    full_thr = full_results["threshold"]
    full_smooth = full_results["smooth"]
    min_thr = min_results["threshold"]
    min_smooth = min_results["smooth"]

    min_thr_subset = min_thr["train_indices"] if min_thr is not None else "не найдено"
    min_smooth_subset = min_smooth["train_indices"] if min_smooth is not None else "не найдено"

    min_thr_epochs = min_thr["epochs"] if min_thr is not None else "не найдено"
    min_smooth_epochs = min_smooth["epochs"] if min_smooth is not None else "не найдено"

    lines = [
        "1. Цель работы",
        (
            "Целью работы является исследование однослойной нейронной сети (одного нейрона) "
            "для моделирования булевой функции варианта 25, а также сравнение влияния двух "
            "функций активации на процесс обучения и способность к обобщению."
        ),
        "",
        "2. Постановка задачи",
        (
            "Требовалось реализовать обучение нейрона по правилу Видроу—Хоффа для функции "
            "F(x1,x2,x3,x4)=NOT((x1 AND x2) OR (NOT x3) OR (NOT x4)) при бинарных входах "
            "x1..x4 и смещении x0=1. Обучение выполнялось в двух режимах: на полной выборке "
            "(16 векторов) и на минимальном подмножестве. Критерий качества: ошибка Хэмминга "
            "E(k) на всех 16 комбинациях."
        ),
        "",
        "3. Теоретические сведения",
        (
            "Однослойная сеть с одним нейроном вычисляет net = w0 + w1*x1 + w2*x2 + w3*x3 + w4*x4, "
            "после чего к net применяется функция активации. Рассмотрены: (1) пороговая функция и "
            "(2) f(net)=1/2*(net/(1+|net|)+1). Для второй функции использована аналитическая производная "
            "df/dnet = 1/(2*(1+|net|)^2). Коррекция весов выполнялась по правилу Видроу—Хоффа с eta=0.3."
        ),
        "",
        "4. Практическая часть",
        (
            f"По таблице истинности получено: единичных выходов {ones_count}, нулевых выходов {zeros_count}. "
            "При обучении на полной выборке обе функции активации достигли E=0."
        ),
        (
            f"Пороговая ФА (полная выборка): {full_thr['epochs']} эпох, итоговые веса "
            f"{format_weights(full_thr['weights'])}."
        ),
        (
            f"Гладкая ФА (полная выборка): {full_smooth['epochs']} эпох, итоговые веса "
            f"{format_weights(full_smooth['weights'])}."
        ),
        (
            f"Минимальное подмножество для пороговой ФА: {min_thr_subset}, эпох до E=0: {min_thr_epochs}."
        ),
        (
            f"Минимальное подмножество для гладкой ФА: {min_smooth_subset}, эпох до E=0: {min_smooth_epochs}."
        ),
        "",
        "5. Выводы",
        (
            "Обе исследованные функции активации позволили реализовать булеву функцию варианта 25 "
            "с нулевой ошибкой на полном множестве входов. Результаты показывают, что при корректно "
            "подобранном (минимальном) обучающем подмножестве сеть сохраняет способность правильно "
            "классифицировать все 16 комбинаций. По скорости сходимости и итоговым весам между ФА "
            "наблюдаются количественные различия, что подтверждает влияние формы функции активации "
            "на динамику обучения."
        ),
        "",
        "Сводная таблица (для отчета):",
        comparison_df.to_string(index=False),
    ]
    return "\n".join(lines)


def build_control_answers() -> str:
    return "\n".join(
        [
            "1. Определение персептрона и алгоритм функционирования",
            (
                "Персептрон — это базовая модель искусственного нейрона, которая формирует выход "
                "на основе взвешенной суммы входов. Для входного вектора x вычисляется net = w^T x + w0. "
                "Далее net подается на функцию активации, которая формирует выход y. В задаче бинарной "
                "классификации обычно используют пороговую функцию, и тогда y принадлежит {0,1}. "
                "Алгоритм функционирования: (1) получить входы, (2) вычислить net, (3) применить ФА, "
                "(4) выдать y, (5) при обучении скорректировать веса по выбранному правилу."
            ),
            "",
            "2. Функции активации НС и их производные",
            (
                "В данной работе использованы две функции. "
                "Пороговая: f(net)=1 при net>=0 и f(net)=0 при net<0; она недифференцируема в нуле, "
                "поэтому в правиле коррекции для нее применяется форма Delta_wi=eta*delta*xi. "
                "Гладкая функция: f(net)=1/2*(net/(1+|net|)+1), ее аналитическая производная "
                "df/dnet=1/(2*(1+|net|)^2). Для нее обновление выполняется как "
                "Delta_wi=eta*delta*(df/dnet)*xi."
            ),
            "",
            "3. Правило обучения Видроу—Хоффа",
            (
                "Правило Видроу—Хоффа (дельта-правило) корректирует веса пропорционально ошибке на текущем "
                "примере. Сначала вычисляется ошибка delta=t-y, где t — целевой выход, y — выход сети. "
                "Далее вес изменяется в направлении, уменьшающем ошибку: "
                "Delta_wi=eta*delta*xi (для порогового случая) или "
                "Delta_wi=eta*delta*(df/dnet)*xi (для дифференцируемой ФА). "
                "Коэффициент eta задает шаг обучения. Итеративное применение правила по эпохам приводит "
                "к уменьшению суммарной ошибки на наборе данных."
            ),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated tables, plots, and text blocks.",
    )
    parser.add_argument(
        "--exhaustive-minimal-search",
        action="store_true",
        help="Search all subsets instead of using the known minimal subsets from the report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or (project_root / "artifacts" / "generated")
    output_dir.mkdir(parents=True, exist_ok=True)

    x_all = generate_binary_vectors()
    t_all = compute_targets(x_all)
    truth_df = build_truth_table(x_all, t_all)
    truth_df.to_csv(output_dir / "truth_table_variant25.csv", index=False, encoding="utf-8-sig")

    full_results: Dict[str, Dict[str, object]] = {}
    minimal_results: Dict[str, Optional[Dict[str, object]]] = {}

    for activation_name in ["threshold", "smooth"]:
        full_results[activation_name] = train_network(
            x_all=x_all,
            t_all=t_all,
            train_indices=tuple(range(len(x_all))),
            activation_name=activation_name,
            eta=ETA,
            max_epochs=MAX_EPOCHS,
        )

    if args.exhaustive_minimal_search:
        for activation_name in ["threshold", "smooth"]:
            minimal_results[activation_name] = find_minimal_training_subset(
                x_all=x_all,
                t_all=t_all,
                activation_name=activation_name,
                eta=ETA,
                max_epochs=MAX_EPOCHS,
            )
    else:
        for activation_name, subset in KNOWN_MINIMAL_SUBSETS.items():
            minimal_results[activation_name] = train_network(
                x_all=x_all,
                t_all=t_all,
                train_indices=subset,
                activation_name=activation_name,
                eta=ETA,
                max_epochs=MAX_EPOCHS,
            )

    # Сохранение таблиц и графиков для полной выборки
    save_result_tables(
        output_dir=output_dir,
        prefix="full_threshold",
        result=full_results["threshold"],
        x_all=x_all,
        t_all=t_all,
    )
    save_result_tables(
        output_dir=output_dir,
        prefix="full_smooth",
        result=full_results["smooth"],
        x_all=x_all,
        t_all=t_all,
    )

    plot_error(
        full_results["threshold"]["history"],
        "Зависимость E(k) от эпохи (ФА 1, полная выборка)",
        output_dir / "E_vs_epoch_full_threshold.png",
    )
    plot_error(
        full_results["smooth"]["history"],
        "Зависимость E(k) от эпохи (ФА 2, полная выборка)",
        output_dir / "E_vs_epoch_full_smooth.png",
    )

    # Сохранение таблиц и графиков для минимальных подмножеств
    if minimal_results["threshold"] is not None:
        save_result_tables(
            output_dir=output_dir,
            prefix="minimal_threshold",
            result=minimal_results["threshold"],
            x_all=x_all,
            t_all=t_all,
        )
        plot_error(
            minimal_results["threshold"]["history"],
            "Зависимость E(k) от эпохи (ФА 1, минимальная выборка)",
            output_dir / "E_vs_epoch_min_threshold.png",
        )

    if minimal_results["smooth"] is not None:
        save_result_tables(
            output_dir=output_dir,
            prefix="minimal_smooth",
            result=minimal_results["smooth"],
            x_all=x_all,
            t_all=t_all,
        )
        plot_error(
            minimal_results["smooth"]["history"],
            "Зависимость E(k) от эпохи (ФА 2, минимальная выборка)",
            output_dir / "E_vs_epoch_min_smooth.png",
        )

    comparison_df = build_comparison_table(full_results, minimal_results)
    comparison_df.to_csv(output_dir / "comparison_table.csv", index=False, encoding="utf-8-sig")

    report_text = build_report_text(truth_df, full_results, minimal_results, comparison_df)
    (output_dir / "report_text_blocks.txt").write_text(report_text, encoding="utf-8")

    control_answers = build_control_answers()
    (output_dir / "control_questions_answers.txt").write_text(control_answers, encoding="utf-8")

    # Краткий консольный отчет для проверки
    print("=== Boolean perceptron experiment, variant 25 ===")
    print("Функция: F = NOT( (x1 AND x2) OR (NOT x3) OR (NOT x4) )")
    print("Целевой вектор T:", bits_to_string(t_all))
    print("\nТаблица истинности:")
    print(truth_df.to_string(index=True))

    for activation_name in ["threshold", "smooth"]:
        res = full_results[activation_name]
        print(f"\n[Полная выборка] {activation_label(activation_name)}")
        print(f"Эпох до E=0: {res['epochs']}, converged={res['converged']}")
        print("Итоговые веса:", format_weights(res["weights"]))
        print("Финальный Y:", bits_to_string(res["y_final"]))
        print("Финальная E:", res["e_final"])

    for activation_name in ["threshold", "smooth"]:
        res = minimal_results[activation_name]
        print(f"\n[Минимальная выборка] {activation_label(activation_name)}")
        if res is None:
            print("Подходящее подмножество не найдено в пределах max_epochs.")
        else:
            print("Индексы:", res["train_indices"])
            print("Размер:", len(res["train_indices"]))
            print("Эпох до E=0:", res["epochs"])
            print("Итоговые веса:", format_weights(res["weights"]))
            print("Финальный Y:", bits_to_string(res["y_final"]))
            print("Финальная E:", res["e_final"])

    print("\nСводная таблица:")
    print(comparison_df.to_string(index=False))
    print(f"\nАртефакты сохранены в: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
