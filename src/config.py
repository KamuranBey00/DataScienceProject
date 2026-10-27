"""Project-wide constants shared by every notebook (team contract, see README)."""

from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_PATH = DATA_DIR / "splits.csv"
MODELS_DIR = ROOT / "models"
PREDS_DIR = MODELS_DIR / "preds"
IMPORTANCE_DIR = MODELS_DIR / "importance"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = ROOT / "figures"

ID_COL = "row_id"
TARGET = "default"

NOMINAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
CAT_COLS = NOMINAL_COLS + PAY_COLS

BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
NUM_COLS = ["LIMIT_BAL", "AGE"] + BILL_COLS + PAY_AMT_COLS

# Original UCI column order; every processed file keeps this order.
FEATURES = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + PAY_COLS + BILL_COLS + PAY_AMT_COLS
)

# Every code that may appear in the raw data. Fixed so TabNet's cat_dims and
# OneHotEncoder categories never depend on which rows a split happened to get.
CAT_LEVELS = {
    "SEX": [1, 2],
    "EDUCATION": [0, 1, 2, 3, 4, 5, 6],
    "MARRIAGE": [0, 1, 2, 3],
    **{col: list(range(-2, 9)) for col in PAY_COLS},
}

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLITS = tuple(SPLIT_FRACTIONS)
