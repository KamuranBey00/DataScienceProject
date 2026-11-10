"""Missing-value injection, the three imputers and imputation error (Student 1)."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import RobustScaler

from src.config import CAT_COLS, FEATURES, NUM_COLS, PAY_AMT_COLS, SEED

# Never masked: they are the observed drivers of the MAR mechanism.
PROTECTED_COLS = ["LIMIT_BAL", "AGE", "PAY_0"]
# Older clients miss more bill statements; low-limit clients miss more payment records.
MAR_AGE_COLS = [f"BILL_AMT{i}" for i in range(2, 7)]
MAR_LIMIT_COLS = PAY_AMT_COLS[1:]
METHODS = ("median_mode", "knn", "mice")


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std()


def inject_missing(
    df: pd.DataFrame, mcar_rate: float = 0.05, mar_rate: float = 0.03, seed: int = SEED
) -> pd.DataFrame:
    """Boolean mask over FEATURES (True = set to NaN).

    mcar_rate and mar_rate are shares of *all* feature cells, so the expected
    total is mcar_rate + mar_rate. MAR cells are drawn only among cells that
    MCAR left observed, which keeps the two shares additive.
    """
    rng = np.random.default_rng(seed)
    n, n_cells = len(df), len(df) * len(FEATURES)
    mask = pd.DataFrame(False, index=df.index, columns=FEATURES)

    eligible = [c for c in FEATURES if c not in PROTECTED_COLS]
    p_mcar = mcar_rate * n_cells / (n * len(eligible))
    mask[eligible] = rng.random((n, len(eligible))) < p_mcar

    mar_cols = MAR_AGE_COLS + MAR_LIMIT_COLS
    p_mar_mean = mar_rate * n_cells / (n * len(mar_cols))
    drivers = (
        (MAR_AGE_COLS, _zscore(df["AGE"])),
        (MAR_LIMIT_COLS, -_zscore(np.log(df["LIMIT_BAL"]))),
    )
    for cols, driver in drivers:
        weight = _sigmoid(2 * driver)
        p_row = np.clip(p_mar_mean * weight / weight.mean() / (1 - p_mcar), 0, 0.95)
        draw = rng.random((n, len(cols))) < p_row[:, None]
        mask[cols] = mask[cols].to_numpy() | (draw & ~mask[cols].to_numpy())
    return mask


def apply_mask(df: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Copy of df with masked feature cells set to NaN (feature columns become float)."""
    out = df.copy()
    out[FEATURES] = out[FEATURES].astype(float).mask(mask.to_numpy())
    return out


class Imputer(BaseEstimator, TransformerMixin):
    """One interface for the three strategies; works on FEATURES and returns a DataFrame.

    median_mode  baseline: median for numeric, most frequent code for categorical
    knn          KNNImputer on RobustScaler-scaled data (distances need equal scales)
    mice         IterativeImputer (chained BayesianRidge regressions)

    Categorical outputs are rounded and clipped to the code range seen in fit, so
    KNN averages and MICE regressions always become valid codes.
    """

    def __init__(self, method: str = "median_mode", n_neighbors: int = 5, max_iter: int = 10):
        self.method = method
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter

    def fit(self, X: pd.DataFrame, y=None):
        X = X[FEATURES]
        self.cat_low_, self.cat_high_ = X[CAT_COLS].min(), X[CAT_COLS].max()
        if self.method == "median_mode":
            self.imputer_ = ColumnTransformer(
                [
                    ("num", SimpleImputer(strategy="median"), NUM_COLS),
                    ("cat", SimpleImputer(strategy="most_frequent"), CAT_COLS),
                ],
                verbose_feature_names_out=False,
            ).set_output(transform="pandas")
            self.imputer_.fit(X)
        elif self.method == "knn":
            self.scaler_ = RobustScaler().fit(X)
            self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors).fit(self.scaler_.transform(X))
        elif self.method == "mice":
            self.imputer_ = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=self.max_iter,
                initial_strategy="median",
                min_value=X.min().to_numpy(),
                max_value=X.max().to_numpy(),
                random_state=SEED,
            ).fit(X)
        else:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[FEATURES]
        if self.method == "median_mode":
            filled = self.imputer_.transform(X)[FEATURES]
        elif self.method == "knn":
            scaled = self.imputer_.transform(self.scaler_.transform(X))
            filled = pd.DataFrame(self.scaler_.inverse_transform(scaled), columns=FEATURES)
        else:
            filled = pd.DataFrame(self.imputer_.transform(X), columns=FEATURES)
        filled.index = X.index
        filled[CAT_COLS] = filled[CAT_COLS].round().clip(self.cat_low_, self.cat_high_, axis=1)
        return filled


def imputation_errors(
    truth: pd.DataFrame, imputed: pd.DataFrame, mask: pd.DataFrame
) -> pd.DataFrame:
    """Per-column error on masked cells only.

    numeric      NRMSE = RMSE / std of the true column (comparable across columns)
    categorical  share of masked cells restored to the exact true code
    """
    rows = []
    for col in FEATURES:
        hit = mask[col].to_numpy()
        if not hit.any():
            continue
        true_vals = truth[col].to_numpy(dtype=float)[hit]
        pred_vals = imputed[col].to_numpy(dtype=float)[hit]
        if col in CAT_COLS:
            rows.append(
                (col, "categorical", int(hit.sum()), np.nan, (true_vals == pred_vals).mean())
            )
        else:
            rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
            nrmse = rmse / truth[col].to_numpy(dtype=float).std()
            rows.append((col, "numeric", int(hit.sum()), nrmse, np.nan))
    return pd.DataFrame(rows, columns=["column", "kind", "n_masked", "nrmse", "accuracy"])
