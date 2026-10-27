import numpy as np
import pandas as pd
import pytest

from src.config import CAT_LEVELS, FEATURES, ID_COL, NOMINAL_COLS, NUM_COLS, PAY_COLS, TARGET


def make_toy_df(n: int, seed: int = 0) -> pd.DataFrame:
    """Offline stand-in with the real schema: row_id + FEATURES + TARGET."""
    rng = np.random.default_rng(seed)
    data = {col: rng.choice(CAT_LEVELS[col], n) for col in NOMINAL_COLS + PAY_COLS}
    data |= {col: rng.normal(1000, 300, n).round() for col in NUM_COLS}
    data["AGE"] = rng.integers(21, 75, n)
    data["LIMIT_BAL"] = rng.integers(10, 500, n) * 1000
    df = pd.DataFrame(data)[FEATURES]
    df.insert(0, ID_COL, np.arange(n))
    df[TARGET] = (rng.random(n) < 0.22).astype(int)
    return df


@pytest.fixture
def toy_df():
    return make_toy_df(2000)
