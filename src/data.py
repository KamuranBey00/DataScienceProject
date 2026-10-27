"""Loading the raw UCI data, the frozen split, and every processed variant."""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    FEATURES,
    ID_COL,
    PROCESSED_DIR,
    RAW_DIR,
    SEED,
    SPLIT_FRACTIONS,
    SPLITS,
    SPLITS_PATH,
    TARGET,
)

UCI_ID = 350
RAW_CACHE = RAW_DIR / "credit_default.parquet"
VARIANTS = ("v0", "clean", "smote", "ctgan")

# ucimlrepo names the columns X1..X23 / Y, in the same order as FEATURES.
_UCI_RENAME = {f"X{i}": col for i, col in enumerate(FEATURES, start=1)} | {"Y": TARGET}


def _fetch_uci() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=UCI_ID)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    df = df.rename(columns=_UCI_RENAME)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"UCI download is missing expected columns: {sorted(missing)}")
    df = df[FEATURES + [TARGET]].astype("int64")
    # row_id is the position in the original UCI file, so it never shifts.
    df.insert(0, ID_COL, range(len(df)))
    return df


def load_raw(drop_duplicates: bool = True) -> pd.DataFrame:
    """Raw data with standard column names, cached under data/raw/.

    Exact duplicates (all features + target) are dropped by default, keeping the
    first occurrence; pass drop_duplicates=False to analyse them.
    """
    if RAW_CACHE.exists():
        df = pd.read_parquet(RAW_CACHE)
    else:
        df = _fetch_uci()
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(RAW_CACHE, index=False)
    if drop_duplicates:
        df = df[~df.duplicated(subset=FEATURES + [TARGET])].reset_index(drop=True)
    return df


def split_ids(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Stratified train/val/test assignment for every row_id in df."""
    test_size = SPLIT_FRACTIONS["test"]
    rest, test = train_test_split(
        df[[ID_COL, TARGET]], test_size=test_size, stratify=df[TARGET], random_state=seed
    )
    val_share = SPLIT_FRACTIONS["val"] / (1 - test_size)
    train, val = train_test_split(
        rest, test_size=val_share, stratify=rest[TARGET], random_state=seed
    )
    parts = [part[[ID_COL]].assign(split=name) for name, part in zip(SPLITS, (train, val, test))]
    return pd.concat(parts).sort_values(ID_COL).reset_index(drop=True)


def make_splits(force: bool = False) -> pd.DataFrame:
    """Write data/splits.csv once. The file is committed and must not change."""
    if SPLITS_PATH.exists() and not force:
        raise FileExistsError(f"{SPLITS_PATH} already exists; pass force=True to rebuild it.")
    splits = split_ids(load_raw())
    splits.to_csv(SPLITS_PATH, index=False, lineterminator="\n")
    return splits


def load_splits() -> pd.DataFrame:
    if not SPLITS_PATH.exists():
        raise FileNotFoundError(f"{SPLITS_PATH} not found; pull main or run make_splits().")
    return pd.read_csv(SPLITS_PATH)


def processed_path(split: str, variant: str):
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if variant == "clean":
        return PROCESSED_DIR / f"clean_{split}.parquet"
    if variant in ("smote", "ctgan"):
        if split != "train":
            raise ValueError(f"variant {variant!r} only exists for the train split")
        return PROCESSED_DIR / f"train_{variant}.parquet"
    raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")


def load_split(split: str, variant: str = "v0") -> pd.DataFrame:
    """One split of one data variant: row_id + FEATURES + TARGET (+ is_synthetic).

    v0    raw data (no NaN), usable before Student 1's files exist
    clean imputed and cleaned, unscaled (Student 1, checkpoint CP-A)
    smote / ctgan  augmented train set (Student 1, checkpoint CP-B)
    """
    if variant == "v0":
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        splits = load_splits()
        keep = splits.loc[splits["split"] == split, ID_COL]
        df = load_raw()
        return df[df[ID_COL].isin(keep)].reset_index(drop=True)
    path = processed_path(split, variant)
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found: its checkpoint is not merged yet. Use variant='v0' meanwhile."
        )
    return pd.read_parquet(path)


def to_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES], df[TARGET]
