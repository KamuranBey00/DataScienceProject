"""Shared figure style and helpers."""

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
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


def plot_roc_pr(models: dict[str, tuple]):
    """Grafik 5: ROC (left) and precision-recall (right) for several models.

    models: {name: (y_true, y_score)}. Each series gets its AUC / AP in the
    legend; the ROC panel shows the random-classifier diagonal, the PR panel
    the positive rate as a baseline.
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.5))
    for (name, (y_true, y_score)), color in zip(models.items(), SERIES):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color)

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax_pr.plot(recall, precision, label=f"{name} (AP={ap:.3f})", color=color)

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color=INK_MUTED, label="rastgele")
    ax_roc.set(title="ROC", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax_roc.legend(loc="lower right")

    pos_rate = np.mean(next(iter(models.values()))[0])
    ax_pr.axhline(pos_rate, linestyle="--", color=INK_MUTED, label=f"pozitif oran={pos_rate:.2f}")
    ax_pr.set(title="Precision-Recall", xlabel="Recall", ylabel="Precision")
    ax_pr.legend(loc="upper right")

    fig.tight_layout()
    return fig


def plot_cm_grid(models: dict[str, tuple]):
    """Grafik 6: one normalised confusion matrix per model, shared colour scale.

    models: {name: (y_true, y_pred, threshold)}.
    """
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), squeeze=False)
    axes = axes[0]
    for ax, (name, (y_true, y_pred, threshold)) in zip(axes, models.items()):
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        im = ax.imshow(cm, cmap=SEQUENTIAL, vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color=INK)
        ax.set(
            title=f"{name} (eşik={threshold:.2f})",
            xlabel="Tahmin",
            ylabel="Gerçek",
            xticks=[0, 1],
            yticks=[0, 1],
        )
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label="oran")
    return fig


def plot_importance_and_masks(step_masks, tabnet_importance, lgbm_importance, feature_names):
    """Grafik 4: TabNet step masks, TabNet global importance, LGBM gain importance.

    step_masks: array (n_steps, n_features). tabnet_importance, lgbm_importance:
    arrays aligned with feature_names, same order in all three panels (caller
    decides the order, e.g. sorted by tabnet_importance).
    """
    lgbm_norm = np.asarray(lgbm_importance, dtype=float)
    lgbm_norm = lgbm_norm / lgbm_norm.sum()

    fig, (ax_mask, ax_tabnet, ax_lgbm) = plt.subplots(1, 3, figsize=(15, 5))

    im = ax_mask.imshow(step_masks, cmap=SEQUENTIAL, aspect="auto")
    ax_mask.set(
        title="TabNet adım maskeleri",
        xlabel="özellik",
        ylabel="adım",
        xticks=range(len(feature_names)),
        yticks=range(len(step_masks)),
    )
    ax_mask.set_xticklabels(feature_names, rotation=90)
    fig.colorbar(im, ax=ax_mask, fraction=0.046, pad=0.04)

    y_pos = range(len(feature_names))
    ax_tabnet.barh(y_pos, tabnet_importance, color=SERIES[1])
    ax_tabnet.set(title="TabNet global önem", yticks=y_pos)
    ax_tabnet.set_yticklabels(feature_names)
    ax_tabnet.invert_yaxis()

    ax_lgbm.barh(y_pos, lgbm_norm, color=SERIES[2])
    ax_lgbm.set(title="LGBM gain önemi (normalize)", yticks=y_pos)
    ax_lgbm.set_yticklabels(feature_names)
    ax_lgbm.invert_yaxis()

    fig.tight_layout()
    return fig


def plot_missingness(df: pd.DataFrame):
    """Grafik 1 (final hali): msno matrix + heatmap + dendrogram, 1x3."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    msno.matrix(df, ax=axes[0], sparkline=False)
    axes[0].set_title("Eksiklik matrisi")
    msno.heatmap(df, ax=axes[1], cmap=DIVERGING)
    axes[1].set_title("Eksiklik korelasyonu")
    msno.dendrogram(df, ax=axes[2])
    axes[2].set_title("Eksiklik dendrogramı")
    fig.tight_layout()
    return fig


def plot_kde_grid(real: pd.DataFrame, smote: pd.DataFrame, ctgan: pd.DataFrame, columns):
    """Grafik 2 (final hali): 2x3 KDE grid, real/SMOTE/CTGAN overlaid per column."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, columns):
        for label, data, color in zip(["real", "smote", "ctgan"], [real, smote, ctgan], SERIES):
            sns.kdeplot(data[col], ax=ax, label=label, color=color, fill=False)
        ax.set_title(col)
    axes.flat[0].legend()
    fig.tight_layout()
    return fig


def plot_corr_diff(corr_real: pd.DataFrame, corr_smote: pd.DataFrame, corr_ctgan: pd.DataFrame):
    """Grafik 3 (final hali): 1x2 heatmap of |corr_real - corr_synth|, shared colour scale."""
    diff_smote = (corr_real - corr_smote).abs()
    diff_ctgan = (corr_real - corr_ctgan).abs()
    vmax = max(diff_smote.to_numpy().max(), diff_ctgan.to_numpy().max())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    im = None
    for ax, diff, title in zip(
        axes, [diff_smote, diff_ctgan], ["|Corr_real - Corr_SMOTE|", "|Corr_real - Corr_CTGAN|"]
    ):
        im = ax.imshow(diff, cmap=SEQUENTIAL, vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(range(len(diff.columns)))
        ax.set_xticklabels(diff.columns, rotation=90)
        ax.set_yticks(range(len(diff.index)))
        ax.set_yticklabels(diff.index)
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    return fig
