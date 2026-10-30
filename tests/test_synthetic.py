import pandas as pd
import pytest

from src.config import FEATURES
from src.synthetic import corr_diff, fidelity_table, summarize_fidelity, total_variation


def test_identical_data_has_perfect_fidelity(toy_df):
    table = fidelity_table(toy_df, toy_df.copy())
    summary = summarize_fidelity(table, corr_diff(toy_df, toy_df.copy()))
    assert summary["mean_ks"] == 0
    assert summary["mean_wasserstein_std"] == 0
    assert summary["mean_tvd"] == 0
    assert summary["max_abs_corr_diff"] == 0


def test_shifted_column_is_detected(toy_df):
    shifted = toy_df.copy()
    shifted["BILL_AMT1"] += 2 * toy_df["BILL_AMT1"].std()
    table = fidelity_table(toy_df, shifted).set_index("column")
    assert table.loc["BILL_AMT1", "wasserstein_std"] == pytest.approx(2, rel=0.01)
    assert table.loc["BILL_AMT1", "ks_pvalue"] < 1e-6
    assert table.loc["BILL_AMT2", "ks_stat"] == 0


def test_total_variation_bounds():
    assert total_variation(pd.Series([1, 1, 2]), pd.Series([1, 1, 2])) == 0
    assert total_variation(pd.Series([1, 1]), pd.Series([2, 2])) == 1


def test_corr_diff_shape(toy_df):
    assert corr_diff(toy_df, toy_df).shape == (len(FEATURES), len(FEATURES))
