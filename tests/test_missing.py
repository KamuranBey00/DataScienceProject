import numpy as np
import pytest

from src.config import CAT_COLS, FEATURES, NUM_COLS
from src.missing import (
    METHODS,
    PROTECTED_COLS,
    Imputer,
    apply_mask,
    imputation_errors,
    inject_missing,
)
from tests.conftest import make_toy_df


@pytest.fixture(scope="module")
def big_df():
    return make_toy_df(20_000)


def test_inject_missing_hits_target_rate(big_df):
    mask = inject_missing(big_df, mcar_rate=0.05, mar_rate=0.03)
    assert mask.to_numpy().mean() == pytest.approx(0.08, abs=0.003)


def test_inject_missing_never_touches_protected_columns(big_df):
    mask = inject_missing(big_df)
    assert not mask[PROTECTED_COLS].to_numpy().any()


def test_inject_missing_is_reproducible(big_df):
    assert inject_missing(big_df, seed=1).equals(inject_missing(big_df, seed=1))
    assert not inject_missing(big_df, seed=1).equals(inject_missing(big_df, seed=2))


def test_mar_mechanism_depends_on_age(big_df):
    mask = inject_missing(big_df, mcar_rate=0.0, mar_rate=0.03)
    old = big_df["AGE"] > big_df["AGE"].median()
    assert mask.loc[old, "BILL_AMT3"].mean() > 1.5 * mask.loc[~old, "BILL_AMT3"].mean()


@pytest.mark.parametrize("method", METHODS)
def test_imputer_fills_everything_with_valid_codes(toy_df, method):
    holed = apply_mask(toy_df, inject_missing(toy_df))
    train, test = holed.iloc[:1500], holed.iloc[1500:]
    filled = Imputer(method=method).fit(train).transform(test)
    assert not filled.isna().any().any()
    assert list(filled.columns) == FEATURES
    cats = filled[CAT_COLS]
    assert (cats == cats.round()).all().all()
    assert (cats.ge(train[CAT_COLS].min()) & cats.le(train[CAT_COLS].max())).all().all()


def test_baseline_uses_train_statistics_only(toy_df):
    holed = apply_mask(toy_df, inject_missing(toy_df))
    train, test = holed.iloc[:1500], holed.iloc[1500:].copy()
    test["BILL_AMT1"] = np.nan
    filled = Imputer("median_mode").fit(train).transform(test)
    assert (filled["BILL_AMT1"] == train["BILL_AMT1"].median()).all()


def test_imputation_errors_zero_for_perfect_imputation(toy_df):
    mask = inject_missing(toy_df)
    errors = imputation_errors(toy_df[FEATURES], toy_df[FEATURES], mask)
    assert errors.loc[errors["kind"] == "numeric", "nrmse"].eq(0).all()
    assert errors.loc[errors["kind"] == "categorical", "accuracy"].eq(1).all()
    assert set(errors["column"]) <= set(NUM_COLS + CAT_COLS) - set(PROTECTED_COLS)
