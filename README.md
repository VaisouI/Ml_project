# Проекты по машинному обучению

Набор проектов по машинному обучению с воспроизводимым кодом, краткими результатами и отдельными точками запуска для каждого эксперимента.

## Проекты

| Проект | Направление | Что демонстрирует |
| --- | --- | --- |
| [01 - Перцептрон для булевой функции](projects/01-perceptron-boolean-function) | Однонейронный классификатор, обученный правилом Видроу-Хоффа | Динамика обучения, функции активации, поиск минимальной обучающей выборки |
| [02 - Backpropagation и autograd для нейросети](projects/02-neural-network-backprop-autograd) | Двухслойная нейронная сеть и эксперименты с autograd | Ручной расчет градиентов, проверка backpropagation, инициализация, функции активации, оптимизаторы |
| [03 - K-Means и PCA с нуля](projects/03-kmeans-pca-from-scratch) | Алгоритмы обучения без учителя | Собственная реализация K-Means, K-Means++, silhouette score, PCA, PCA + кластеризация |
| [04 - Распознавание лиц с SVM и PCA](projects/04-face-recognition-svm-pca) | Классическое ML-решение для классификации изображений | Подбор SVM, снижение размерности через PCA, воспроизводимый pipeline и артефакты эксперимента |

## Быстрый старт

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Основные проекты запускаются обычными Python-скриптами:

```powershell
python projects\01-perceptron-boolean-function\src\perceptron_boolean_classifier.py
python projects\02-neural-network-backprop-autograd\scripts\run_experiments.py --quick
python projects\03-kmeans-pca-from-scratch\scripts\run_experiments.py --quick
python projects\04-face-recognition-svm-pca\scripts\run_pipeline.py --data-path path\to\persons_pics_train.zip
```

## Примечания

- Большие обученные модели, исходные датасеты, локальные кэши и временные артефакты исключены через `.gitignore`.
- Основная логика проектов вынесена в Python-модули и воспроизводимые CLI-скрипты.
- Исходные датасеты, PDF-материалы и локальные рабочие файлы не коммитятся.
