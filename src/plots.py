"""Shared figure style and helpers."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.config import FIGURES_DIR, ID_COL
from src.data import load_split, to_xy
from src.features import make_preprocessor
from src.metrics import pick_threshold

# Categorical slots in fixed order (blue, orange, aqua). The first three are
# colour-blind safe as a set, so a figure never shows more than three series.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
# Single-hue sequential ramp for magnitudes (heatmaps): near-zero recedes to the surface.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"]
)
# Diverging ramp for signed values in [-1, 1]: red <- neutral gray -> blue.
DIVERGING = LinearSegmentedColormap.from_list("div_red_blue", ["#e34948", "#f0efec", "#2a78d6"])


def set_style() -> None:
    sns.set_theme(context="notebook", style="whitegrid", palette=SERIES)
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2,
            "legend.frameon": False,
        }
    )


def save_fig(fig, name: str, subdir: str | None = None):
    """Save to figures/[subdir/]name.png and return the path."""
    folder = FIGURES_DIR / subdir if subdir else FIGURES_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path)
    return path


def make_dummy_preds(variant: str = "v0") -> pd.DataFrame:
    """Logistic-regression test predictions in the shared format.

    Stand-in for models/preds/*.csv while plotting code is developed; it is not
    written to disk so it can never be mistaken for a real model's output.
    """
    X_train, y_train = to_xy(load_split("train", variant))
    val, test = load_split("val", variant), load_split("test", variant)
    model = make_pipeline(make_preprocessor("linear"), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    X_val, y_val = to_xy(val)
    threshold = pick_threshold(y_val, model.predict_proba(X_val)[:, 1])
    X_test, y_test = to_xy(test)
    score = model.predict_proba(X_test)[:, 1]
    return pd.DataFrame(
        {
            ID_COL: test[ID_COL],
            "y_true": y_test,
            "y_score": score,
            "y_pred": (score >= threshold).astype(int),
            "threshold": threshold,
        }
    )
