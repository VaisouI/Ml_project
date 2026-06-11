# K-Means и PCA с нуля

Проект по обучению без учителя: ручная реализация K-Means и PCA на NumPy с проверкой на синтетических кластерах и датасете `digits`.

## Что внутри

- Реализация K-Means со случайной и K-Means++ инициализацией.
- Подсчет inertia, silhouette score, повторные запуски и обработка пустых кластеров.
- Реализация PCA через центрирование, ковариационную матрицу, eigendecomposition, projection и reconstruction.
- Анализ PCA + K-Means на высокоразмерных векторах рукописных цифр.

## Результаты

| Эксперимент | Результат |
| --- | --- |
| Метод локтя и silhouette на синтетических данных | Лучшее число кластеров - `K=5`; silhouette достиг `0.7332` |
| Случайная инициализация против K-Means++ | K-Means++ снизил среднюю inertia с `1274.230` до `829.022` и среднее число итераций с `8.800` до `4.300` |
| PCA на `digits` | `21` компонента сохраняет `90.32%` дисперсии |
| PCA + K-Means | Silhouette вырос с `0.1828` в 64D до `0.2102` в 21D-пространстве с 90% дисперсии |

## Запуск

```powershell
python projects\03-kmeans-pca-from-scratch\scripts\run_experiments.py --quick
```

Полный воспроизводимый запуск:

```powershell
python projects\03-kmeans-pca-from-scratch\scripts\run_experiments.py
```

CSV-таблицы и графики сохраняются в `projects/03-kmeans-pca-from-scratch/artifacts/generated/`.

## Структура

- `src/clustering.py` - K-Means, K-Means++, расстояния и silhouette score.
- `src/dimensionality.py` - PCA через NumPy.
- `src/experiments.py` - воспроизводимые эксперименты и сохранение артефактов.
- `scripts/run_experiments.py` - CLI-точка запуска.

## Что показывает проект

Проект хорошо подходит для обсуждения роли инициализации в не выпуклой кластеризации, компромисса PCA между компактностью и потерей информации, а также различий между метриками без учителя и настоящими метками классов.
