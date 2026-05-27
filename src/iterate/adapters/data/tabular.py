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
from dataclasses import dataclass
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


def _looks_like_classification(target: pd.Series) -> bool:
    """Discrete target → classification; a continuous float target → regression."""
    if not pd.api.types.is_float_dtype(target):
        return True
    return bool(target.nunique(dropna=True) <= _MAX_CLASSES_FOR_STRATIFY)


def load_csv(
    path: str | Path,
    target: str,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
    stratify: bool = True,
) -> TabularDataset:
    """Load a CSV and return a deterministic train/holdout split.

    ``stratify`` keeps the class balance identical in train and holdout for a
    classification target; it is ignored for a continuous (regression) target.
    """
    frame = pd.read_csv(path)
    if target not in frame.columns:
        raise ValueError(f"target column {target!r} not in CSV columns {list(frame.columns)}")

    features = [col for col in frame.columns if col != target]
    feature_frame = frame[features]
    target_col = frame[target]

    stratify_on = target_col if (stratify and _looks_like_classification(target_col)) else None
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


__all__ = ["DEFAULT_SEED", "DEFAULT_TEST_SIZE", "TabularDataset", "load_csv"]
