"""Machine-learning utility of a training set, used by MVI and TSTR/TRTR (Student 1)."""

import pandas as pd
from lightgbm import LGBMClassifier

from src.config import FEATURES, SEED, TARGET
from src.metrics import evaluate, pick_threshold


def lgbm_utility(train: pd.DataFrame, holdout: pd.DataFrame) -> dict:
    """Fixed-parameter LightGBM fit on `train`, scored on real `holdout` data.

    The parameters never change between calls, so differences in the scores come
    from the training data alone. The macro-F1 threshold is picked on `holdout`
    itself: absolute F1 is slightly optimistic, but the comparison stays fair.
    """
    model = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=SEED, verbose=-1
    )
    model.fit(train[FEATURES], train[TARGET])
    score = model.predict_proba(holdout[FEATURES])[:, 1]
    threshold = pick_threshold(holdout[TARGET], score)
    return evaluate(holdout[TARGET], score, (score >= threshold).astype(int))
