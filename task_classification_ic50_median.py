from __future__ import annotations

import json
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_utils import IC50_COLUMN, MODELS_DIR, REPORTS_DIR, ensure_dirs, feature_columns, load_data


RANDOM_STATE = 42
TASK_NAME = "classification_ic50_median"


def make_models() -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    return {
        "logistic_regression": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
                ]
            ),
            {"model__C": [0.5, 1.0]},
        ),
        "random_forest": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight="balanced")),
                ]
            ),
            {"model__n_estimators": [120], "model__max_depth": [6, None], "model__min_samples_leaf": [3]},
        ),
        "gradient_boosting": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
                ]
            ),
            {"model__n_estimators": [100], "model__learning_rate": [0.05, 0.1], "model__max_depth": [2]},
        ),
    }


def run() -> pd.DataFrame:
    ensure_dirs()
    data = load_data()
    columns = feature_columns(data)
    target = (data[IC50_COLUMN] > data[IC50_COLUMN].median()).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        data[columns],
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    rows: list[dict[str, Any]] = []
    best_model: GridSearchCV | None = None
    best_score = -1.0
    for model_name, (model, params) in make_models().items():
        search = GridSearchCV(model, params, cv=5, scoring="f1", n_jobs=1)
        search.fit(x_train, y_train)
        predictions = search.predict(x_test)
        probabilities = search.predict_proba(x_test)[:, 1]
        rows.append(
            {
                "task": TASK_NAME,
                "model": model_name,
                "accuracy": accuracy_score(y_test, predictions),
                "f1": f1_score(y_test, predictions),
                "roc_auc": roc_auc_score(y_test, probabilities),
                "positive_rate": float(target.mean()),
                "best_params": json.dumps(search.best_params_, ensure_ascii=False),
            }
        )
        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_model = search

    result = pd.DataFrame(rows).sort_values("f1", ascending=False)
    result.to_csv(REPORTS_DIR / f"{TASK_NAME}_results.csv", index=False)
    if best_model is not None:
        joblib.dump(best_model.best_estimator_, MODELS_DIR / f"{TASK_NAME}.joblib")
    return result


if __name__ == "__main__":
    print(run().to_string(index=False))

