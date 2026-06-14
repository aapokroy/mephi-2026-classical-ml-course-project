from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_utils import MODELS_DIR, REPORTS_DIR, SI_COLUMN, ensure_dirs, feature_columns, load_data


RANDOM_STATE = 42
TASK_NAME = "regression_si"


def make_models() -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    return {
        "ridge": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", Ridge()),
                ]
            ),
            {"model__alpha": [1.0, 10.0]},
        ),
        "random_forest": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)),
                ]
            ),
            {"model__n_estimators": [120], "model__max_depth": [6, None], "model__min_samples_leaf": [3]},
        ),
        "gradient_boosting": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
                ]
            ),
            {"model__n_estimators": [100], "model__learning_rate": [0.05, 0.1], "model__max_depth": [2]},
        ),
    }


def run() -> pd.DataFrame:
    ensure_dirs()
    data = load_data()
    columns = feature_columns(data)
    x_train, x_test, y_train, y_test = train_test_split(
        data[columns],
        np.log1p(data[SI_COLUMN]),
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    rows: list[dict[str, Any]] = []
    best_model: GridSearchCV | None = None
    best_score = -np.inf
    for model_name, (model, params) in make_models().items():
        search = GridSearchCV(model, params, cv=5, scoring="neg_mean_absolute_error", n_jobs=1)
        search.fit(x_train, y_train)
        predictions = np.expm1(search.predict(x_test))
        actual = np.expm1(y_test)
        rows.append(
            {
                "task": TASK_NAME,
                "model": model_name,
                "mae": mean_absolute_error(actual, predictions),
                "rmse": mean_squared_error(actual, predictions) ** 0.5,
                "r2": r2_score(actual, predictions),
                "best_params": json.dumps(search.best_params_, ensure_ascii=False),
            }
        )
        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_model = search

    result = pd.DataFrame(rows).sort_values("mae")
    result.to_csv(REPORTS_DIR / f"{TASK_NAME}_results.csv", index=False)
    if best_model is not None:
        joblib.dump(best_model.best_estimator_, MODELS_DIR / f"{TASK_NAME}.joblib")
    return result


if __name__ == "__main__":
    print(run().to_string(index=False))

