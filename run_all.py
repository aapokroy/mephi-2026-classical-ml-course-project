from __future__ import annotations

from task_classification_cc50_median import run as run_cc50_median
from task_classification_ic50_median import run as run_ic50_median
from task_classification_si_gt_8 import run as run_si_gt_8
from task_classification_si_median import run as run_si_median
from task_regression_cc50 import run as run_regression_cc50
from task_regression_ic50 import run as run_regression_ic50
from task_regression_si import run as run_regression_si


def main() -> None:
    tasks = [
        run_regression_ic50,
        run_regression_cc50,
        run_regression_si,
        run_ic50_median,
        run_cc50_median,
        run_si_median,
        run_si_gt_8,
    ]
    for task in tasks:
        result = task()
        print(result.head(1).to_string(index=False))


if __name__ == "__main__":
    main()

