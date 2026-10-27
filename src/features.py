"""Encoders and scalers. Always fit on the training split only."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.config import CAT_COLS, CAT_LEVELS, FEATURES, NOMINAL_COLS, NUM_COLS, PAY_COLS

KINDS = ("linear", "tree", "tabnet")


def make_preprocessor(kind: str) -> ColumnTransformer:
    """Unfitted column transformer for one model family.

    linear  one-hot nominal codes, RobustScaler on numeric and PAY_x (Perceptron)
    tree    the 23 raw columns unchanged (LightGBM)
    tabnet  RobustScaler on numeric, categorical codes passed through (TabNet)
    """
    if kind == "linear":
        onehot = OneHotEncoder(
            categories=[CAT_LEVELS[col] for col in NOMINAL_COLS],
            handle_unknown="ignore",
            sparse_output=False,
        )
        transformers = [
            ("onehot", onehot, NOMINAL_COLS),
            ("scale", RobustScaler(), NUM_COLS + PAY_COLS),
        ]
    elif kind == "tree":
        transformers = [("keep", "passthrough", FEATURES)]
    elif kind == "tabnet":
        transformers = [("scale", RobustScaler(), NUM_COLS), ("keep", "passthrough", CAT_COLS)]
    else:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return ColumnTransformer(
        transformers, remainder="drop", verbose_feature_names_out=False
    ).set_output(transform="pandas")
