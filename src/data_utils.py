from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "coursework_data.xlsx"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
ATTACHMENTS_DIR = PROJECT_ROOT / "attachments"

IC50_COLUMN = "IC50, mM"
CC50_COLUMN = "CC50, mM"
SI_COLUMN = "SI"
TARGET_COLUMNS = [IC50_COLUMN, CC50_COLUMN, SI_COLUMN]
INDEX_COLUMNS = ["Unnamed: 0"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    data = pd.read_excel(path)
    return data


def feature_columns(data: pd.DataFrame) -> list[str]:
    dropped_columns = set(TARGET_COLUMNS + INDEX_COLUMNS)
    return [column for column in data.columns if column not in dropped_columns]


def make_classification_target(data: pd.DataFrame, target_column: str, threshold: float) -> pd.Series:
    return (data[target_column] > threshold).astype(int)


def ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)
