"""Tabular data loading + deterministic train/holdout split.

Loads a CSV and produces a reproducible, stratified split as a `TabularDataset`.
The split is leakage-safe by construction: it happens here, *before* any
preprocessing (which is the model's job, fit on train only). A content hash of
the data is recorded so any result is traceable to the exact data + split that
produced it.

The split is reproducible via a fixed seed (the data is static within a run).
Hash-based splitting — robust when the data itself evolves between runs — is a
later concern (see IDEAS / Week 8 discovery).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_TEST_SIZE = 0.2
DEFAULT_SEED = 42
# Above this many distinct target values, a numeric target is treated as regression.
_MAX_CLASSES_FOR_STRATIFY = 20


@dataclass(frozen=True)
class TabularDataset:
    """A loaded tabular dataset, split into train + a sealed holdout.

    Holds live pandas objects (not a persisted contract), so it's a dataclass —
    not a Pydantic schema. The holdout stays untouched until scoring.
    """

    train_features: pd.DataFrame
    train_target: pd.Series
    test_features: pd.DataFrame
    test_target: pd.Series
    target: str
    features: list[str]
    seed: int
    test_size: float
    data_hash: str  # content fingerprint of the full dataset (a data version)
    user_split: bool = False  # the caller supplied train and holdout; nothing was shuffled

    @property
    def n_train(self) -> int:
        return len(self.train_features)

    @property
    def n_test(self) -> int:
        return len(self.test_features)


def _content_hash(frame: pd.DataFrame) -> str:
    """A stable content fingerprint of a dataframe — lightweight data versioning."""
    row_hashes = pd.util.hash_pandas_object(frame, index=True).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()[:16]


def looks_like_classification(target: pd.Series) -> bool:
    """Discrete target → classification; a continuous target → regression.

    The distinct-value cap applies to INTEGER targets as well as float ones. An
    integer column was previously always read as a class label, so a price, a count
    or a year became thousands of "classes" and the stratified split raised before
    the run could start: the diamonds dataset, with 11,602 distinct integer prices,
    could not be loaded at all. Every target with few enough distinct values is
    unaffected, so nothing that worked before changes behaviour.
    """
    if pd.api.types.is_bool_dtype(target) or not pd.api.types.is_numeric_dtype(target):
        return True
    return bool(target.nunique(dropna=True) <= _MAX_CLASSES_FOR_STRATIFY)


log = logging.getLogger(__name__)


def _read_csv_any_encoding(path: str | Path) -> pd.DataFrame:
    """Read a CSV without demanding it be UTF-8.

    Plain `pd.read_csv` assumes UTF-8 and raises `UnicodeDecodeError` on anything
    else. That is not an edge case: any European export with a euro sign or an
    accented product name is latin-1, and the user's first contact with the tool
    would be a decoding traceback on a file that opens fine in every spreadsheet.
    Try UTF-8 first, then latin-1, which is byte-complete and cannot itself fail.
    """
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        log.info("%s is not UTF-8; re-reading as latin-1", path)
        return pd.read_csv(path, encoding="latin-1")


def split_frame(
    frame: pd.DataFrame,
    target: str,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
    stratify: bool = True,
) -> TabularDataset:
    """A deterministic train/holdout split of one frame.

    ``stratify`` keeps the class balance identical in train and holdout for a
    classification target; it is ignored for a continuous (regression) target.
    """
    if target not in frame.columns:
        raise ValueError(f"target column {target!r} not in columns {list(frame.columns)}")

    features = [col for col in frame.columns if col != target]
    feature_frame = frame[features]
    target_col = frame[target]

    stratify_on = target_col if (stratify and looks_like_classification(target_col)) else None
    train_feat, test_feat, train_tgt, test_tgt = train_test_split(
        feature_frame,
        target_col,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_on,
    )

    return TabularDataset(
        train_features=train_feat,
        train_target=train_tgt,
        test_features=test_feat,
        test_target=test_tgt,
        target=target,
        features=features,
        seed=seed,
        test_size=test_size,
        data_hash=_content_hash(frame),
    )


def dataset_from_frames(
    train: pd.DataFrame, holdout: pd.DataFrame, target: str, *, seed: int = DEFAULT_SEED
) -> TabularDataset:
    """A dataset from a split the CALLER made. Membership and labels are untouched.

    Rows are shuffled with the fixed seed, pairing kept, because a file sorted by
    label would otherwise reach the kernel with its order encoding the answer. The
    holdout is reordered to the training frame's columns so the kernel sees one
    layout, and its index is kept disjoint from train.
    """
    for name, frame in (("train", train), ("holdout", holdout)):
        if target not in frame.columns:
            raise ValueError(
                f"target column {target!r} not in the {name} columns {list(frame.columns)}"
            )
        if frame.empty:
            raise ValueError(f"the {name} file has no rows")
        if frame[target].isna().any():
            raise ValueError(
                f"the {name} file has {int(frame[target].isna().sum())} empty {target!r} value(s)"
            )
    features = [col for col in train.columns if col != target]
    missing = sorted(set(features) - set(holdout.columns))
    extra = sorted(set(holdout.columns) - set(train.columns))
    if missing or extra:
        raise ValueError(
            f"train and holdout columns differ: the holdout is missing {missing} "
            f"and has extra {extra}"
        )
    holdout = holdout[[*features, target]]
    holdout.index = pd.RangeIndex(len(train), len(train) + len(holdout))
    train = train.sample(frac=1, random_state=seed)
    holdout = holdout.sample(frac=1, random_state=seed)

    if looks_like_classification(train[target]):
        unseen = set(holdout[target].dropna().unique()) - set(train[target].dropna().unique())
        if unseen:
            log.warning(
                "holdout has %d class(es) absent from train: %s",
                len(unseen),
                sorted(str(v) for v in unseen)[:5],
            )

    n_train, n_test = len(train), len(holdout)
    return TabularDataset(
        train_features=train[features],
        train_target=train[target],
        test_features=holdout[features],
        test_target=holdout[target],
        target=target,
        features=features,
        seed=seed,
        test_size=n_test / (n_train + n_test),
        data_hash=_content_hash(
            pd.concat([train.sort_index(), holdout.sort_index()], ignore_index=True)
        ),
        user_split=True,
    )


def load_csv(
    path: str | Path,
    target: str,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
    stratify: bool = True,
) -> TabularDataset:
    """Load a CSV and return a deterministic train/holdout split."""
    return split_frame(
        _read_csv_any_encoding(path), target, test_size=test_size, seed=seed, stratify=stratify
    )


def load_split(
    train_path: str | Path, holdout_path: str | Path, target: str, *, seed: int = DEFAULT_SEED
) -> TabularDataset:
    """Two CSVs the user split themselves; the holdout is sealed as it stands."""
    return dataset_from_frames(
        _read_csv_any_encoding(train_path), _read_csv_any_encoding(holdout_path), target, seed=seed
    )


def with_smaller_holdout(dataset: TabularDataset, n: int) -> TabularDataset:
    """The same dataset with its holdout cut to the first ``n`` rows.

    Both loaders shuffle rows with the fixed seed, so the first ``n`` is a random
    subset and the SAME subset for every candidate, which keeps the comparison
    paired. The winner is re-scored on the whole holdout at the end.
    """
    if n >= len(dataset.test_features) or n <= 0:
        return dataset
    return replace(
        dataset,
        test_features=dataset.test_features.head(n),
        test_target=dataset.test_target.head(n),
    )


__all__ = [
    "DEFAULT_SEED",
    "DEFAULT_TEST_SIZE",
    "TabularDataset",
    "dataset_from_frames",
    "load_csv",
    "load_split",
    "split_frame",
    "with_smaller_holdout",
]
