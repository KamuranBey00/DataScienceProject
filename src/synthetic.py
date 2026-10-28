"""Statistical fidelity of synthetic data against real data (Student 1)."""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from src.config import CAT_COLS, FEATURES, NUM_COLS


def total_variation(real: pd.Series, synth: pd.Series) -> float:
    """0.5 * sum |p - q| over the union of categories (0 = identical, 1 = disjoint)."""
    p = real.value_counts(normalize=True)
    q = synth.value_counts(normalize=True)
    p, q = p.align(q, fill_value=0)
    return float(0.5 * (p - q).abs().sum())


def fidelity_table(
    real: pd.DataFrame, synth: pd.DataFrame, num_cols=NUM_COLS, cat_cols=CAT_COLS
) -> pd.DataFrame:
    """KS and Wasserstein for numeric columns, total variation for categorical ones.

    Wasserstein is computed on values standardised with the real column's mean and
    std, so distances are comparable between columns measured in different units.
    """
    rows = []
    for col in num_cols:
        r = real[col].to_numpy(dtype=float)
        s = synth[col].to_numpy(dtype=float)
        mu, sd = r.mean(), r.std() or 1.0
        ks = ks_2samp(r, s)
        wd = wasserstein_distance((r - mu) / sd, (s - mu) / sd)
        rows.append((col, "numeric", ks.statistic, ks.pvalue, wd, np.nan))
    for col in cat_cols:
        rows.append(
            (col, "categorical", np.nan, np.nan, np.nan, total_variation(real[col], synth[col]))
        )
    return pd.DataFrame(
        rows, columns=["column", "kind", "ks_stat", "ks_pvalue", "wasserstein_std", "tvd"]
    )


def corr_diff(real: pd.DataFrame, synth: pd.DataFrame, cols=FEATURES) -> pd.DataFrame:
    """|Corr_real - Corr_synth| (Pearson); constant columns count as zero correlation."""
    c_real = real[cols].corr().fillna(0)
    c_synth = synth[cols].corr().fillna(0)
    return (c_real - c_synth).abs()


def summarize_fidelity(table: pd.DataFrame, diff: pd.DataFrame) -> dict:
    upper = diff.to_numpy()[np.triu_indices_from(diff, k=1)]
    return {
        "mean_ks": table["ks_stat"].mean(),
        "mean_wasserstein_std": table["wasserstein_std"].mean(),
        "mean_tvd": table["tvd"].mean(),
        "mean_abs_corr_diff": upper.mean(),
        "max_abs_corr_diff": upper.max(),
    }
