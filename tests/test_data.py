import numpy as np
import pandas as pd
import pytest

from src.config import CAT_LEVELS, FEATURES, ID_COL, NOMINAL_COLS, NUM_COLS, PAY_COLS, TARGET
from src.data import processed_path, split_ids
from src.features import make_preprocessor
from src.metrics import evaluate, pick_threshold


def test_split_ids_is_deterministic_and_disjoint(toy_df):
    first, second = split_ids(toy_df), split_ids(toy_df)
    pd.testing.assert_frame_equal(first, second)
    assert first[ID_COL].is_unique
    assert set(first[ID_COL]) == set(toy_df[ID_COL])


def test_split_ids_fractions_and_stratification(toy_df):
    splits = split_ids(toy_df).merge(toy_df[[ID_COL, TARGET]], on=ID_COL)
    shares = splits["split"].value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.70, abs=0.01)
    assert shares["val"] == pytest.approx(0.15, abs=0.01)
    rates = splits.groupby("split")[TARGET].mean()
    assert rates.max() - rates.min() < 0.01


def test_processed_path_rejects_unknown_inputs():
    with pytest.raises(ValueError):
        processed_path("holdout", "clean")
    with pytest.raises(ValueError):
        processed_path("test", "smote")
    with pytest.raises(ValueError):
        processed_path("train", "raw")


@pytest.mark.parametrize(
    "kind, n_cols",
    [
        ("linear", sum(len(CAT_LEVELS[c]) for c in NOMINAL_COLS) + len(NUM_COLS + PAY_COLS)),
        ("tree", len(FEATURES)),
        ("tabnet", len(FEATURES)),
    ],
)
def test_make_preprocessor_output_width(toy_df, kind, n_cols):
    out = make_preprocessor(kind).fit_transform(toy_df[FEATURES])
    assert out.shape == (len(toy_df), n_cols)


def test_pick_threshold_separates_perfect_scores():
    y = np.array([0] * 80 + [1] * 20)
    score = np.where(y == 1, 0.9, 0.1)
    threshold = pick_threshold(y, score)
    assert evaluate(y, score, (score >= threshold).astype(int))["macro_f1"] == 1.0
