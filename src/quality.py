"""Data-quality checks, invalid-code fixes and the constraint table (Student 1)."""

import numpy as np
import pandas as pd

from src.config import BILL_COLS, CAT_COLS, FEATURES, NUM_COLS, PAY_AMT_COLS, PAY_COLS, TARGET

# Codes outside the UCI documentation; both are folded into the "other" level.
INVALID_CODES = {"EDUCATION": ([0, 5, 6], 4), "MARRIAGE": ([0], 3)}


def constraint_report(df: pd.DataFrame) -> pd.DataFrame:
    """The assignment's lower bounds checked against df (FEATURES + TARGET)."""
    counts = df[TARGET].value_counts()
    imbalance = counts.max() / counts.min()
    cat_share = len(CAT_COLS) / len(FEATURES)
    missing = float(df[FEATURES].isna().to_numpy().mean())
    rows = [
        ("Rows", ">= 20,000", f"{len(df):,}", len(df) >= 20_000),
        ("Features", ">= 20", str(len(FEATURES)), len(FEATURES) >= 20),
        (
            "Categorical share",
            ">= 30%",
            f"{cat_share:.1%} ({len(CAT_COLS)}/{len(FEATURES)})",
            cat_share >= 0.30,
        ),
        ("Missing cells", ">= 5%", f"{missing:.2%}", missing >= 0.05),
        ("Imbalance (majority:minority)", ">= 3:1", f"{imbalance:.2f}:1", imbalance >= 3),
    ]
    return pd.DataFrame(rows, columns=["constraint", "required", "value", "ok"])


def duplicate_report(df: pd.DataFrame) -> dict:
    exact = df.duplicated(subset=FEATURES + [TARGET])
    same_features = df.duplicated(subset=FEATURES, keep=False)
    conflicting = df[same_features].groupby(FEATURES)[TARGET].nunique().gt(1).sum()
    return {
        "exact_duplicate_rows": int(exact.sum()),
        "feature_duplicates_with_conflicting_target": int(conflicting),
    }


def validity_report(df: pd.DataFrame) -> pd.DataFrame:
    """Rule violations per check. Negative bills are counted but are valid (credit balance)."""
    checks = {
        "LIMIT_BAL <= 0": df["LIMIT_BAL"].le(0),
        "AGE outside 18-100": ~df["AGE"].between(18, 100),
        "SEX not in {1,2}": ~df["SEX"].isin([1, 2]),
        "EDUCATION in {0,5,6} (undocumented)": df["EDUCATION"].isin(INVALID_CODES["EDUCATION"][0]),
        "MARRIAGE == 0 (undocumented)": df["MARRIAGE"].isin(INVALID_CODES["MARRIAGE"][0]),
        "PAY_x outside -2..8": ~df[PAY_COLS].apply(lambda s: s.between(-2, 8)).all(axis=1),
        "PAY_AMT negative": df[PAY_AMT_COLS].lt(0).any(axis=1),
        "BILL_AMT negative (valid: credit balance)": df[BILL_COLS].lt(0).any(axis=1),
    }
    return pd.DataFrame(
        {
            "check": list(checks),
            "rows": [int(mask.sum()) for mask in checks.values()],
            "share": [round(float(mask.mean()), 4) for mask in checks.values()],
        }
    )


def fix_invalid_codes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Map undocumented EDUCATION/MARRIAGE codes to "other"; returns (copy, changed counts)."""
    out = df.copy()
    changed = {}
    for col, (bad, other) in INVALID_CODES.items():
        hit = out[col].isin(bad)
        changed[col] = int(hit.sum())
        out.loc[hit, col] = other
    return out, changed


def iqr_outliers(df: pd.DataFrame, cols=NUM_COLS, k: float = 1.5) -> pd.DataFrame:
    q1, q3 = df[cols].quantile(0.25), df[cols].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    outside = df[cols].lt(lower) | df[cols].gt(upper)
    return pd.DataFrame(
        {
            "q1": q1,
            "q3": q3,
            "lower": lower,
            "upper": upper,
            "n_outliers": outside.sum(),
            "pct_outliers": (100 * outside.mean()).round(2),
            "max": df[cols].max(),
        }
    ).sort_values("pct_outliers", ascending=False)


def signed_log1p(values) -> np.ndarray:
    """Log scale that keeps the sign, for plotting money columns with negatives."""
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))
