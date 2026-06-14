# mephi-2026-classical-ml-course-project

Курсовая работа по классическому машинному обучению. Используется набор данных
по химическим соединениям и показателям эффективности против вируса гриппа.

## Структура

```text
mephi-2026-classical-ml-course-project/
  data/raw/coursework_data.xlsx      # локально, в git не добавляется
  src/data_utils.py
  eda.ipynb
  task_regression_ic50.py
  task_regression_cc50.py
  task_regression_si.py
  task_classification_ic50_median.py
  task_classification_cc50_median.py
  task_classification_si_median.py
  task_classification_si_gt_8.py
  run_all.py
  attachments/
  models/                            # локально, в git не добавляется
  reports/
  requirements.txt
```

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

EDA:

```bash
jupyter notebook eda.ipynb
```

Все модели:

```bash
PYTHONPATH=. python run_all.py
```

## Задачи

Решены три задачи регрессии:

- `IC50, mM`;
- `CC50, mM`;
- `SI`.

Решены четыре задачи классификации:

- `IC50, mM` больше медианы;
- `CC50, mM` больше медианы;
- `SI` больше медианы;
- `SI` больше 8.

Для каждой задачи сравнивались несколько моделей с небольшим подбором
гиперпараметров. Лучшие модели сохраняются локально в `models/`.

Данные курса нужно положить в `data/raw/coursework_data.xlsx`. Файл с данными,
таблицы с результатами и обученные модели не добавлялись в репозиторий.

## Основные результаты

Лучшие результаты по регрессии:

```text
regression_ic50: random_forest, MAE 220.2404, R2 0.3134
regression_cc50: random_forest, MAE 297.2899, R2 0.4846
regression_si: random_forest, MAE 177.8079, R2 0.0027
```

Лучшие результаты по классификации:

```text
classification_ic50_median: gradient_boosting, F1 0.7431, ROC-AUC 0.7686
classification_cc50_median: random_forest, F1 0.7421, ROC-AUC 0.8226
classification_si_median: logistic_regression, F1 0.6275, ROC-AUC 0.6558
classification_si_gt_8: gradient_boosting, F1 0.5500, ROC-AUC 0.7409
```

Отчёт сохранён в файле:

- `reports/classical_ml_course_report.docx`.
