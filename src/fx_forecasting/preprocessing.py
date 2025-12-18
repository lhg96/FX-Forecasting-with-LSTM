"""Reusable preprocessing helpers for the FX forecasting pipeline."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def forward_fill_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by date, forward-fill missing values, and return a copy."""

    if "date" not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column.")

    ordered = df.sort_values("date").copy()
    ordered["date"] = pd.to_datetime(ordered["date"])
    ordered.set_index("date", inplace=True)
    
    # Forward fill
    filled = ordered.ffill()
    # Backward fill for remaining NaNs
    filled = filled.bfill()
    # Fill any remaining NaNs with column means
    filled = filled.fillna(filled.mean())
    
    return filled


def build_minmax_scaler(
    df: pd.DataFrame,
    feature_range: Tuple[float, float] = (0, 1),
) -> Tuple[MinMaxScaler, pd.DataFrame]:
    """Fit a scaler on ``df`` and return both the scaler and transformed frame."""

    scaler = MinMaxScaler(feature_range=feature_range)
    scaled = pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=df.columns,
    )
    return scaler, scaled


def create_supervised_sequences(
    df: pd.DataFrame,
    *,
    sequence_length: int,
    target_column: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a feature frame into overlapping sequences."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    if target_column not in df.columns:
        raise ValueError(f"{target_column} not found in DataFrame.")

    values = df.values
    X, y = [], []
    target_index = df.columns.get_loc(target_column)

    for idx in range(sequence_length, len(df)):
        X.append(values[idx - sequence_length : idx])
        y.append(values[idx, target_index])

    if not X:
        raise ValueError("Not enough rows to build a single sequence.")

    return np.array(X), np.array(y)


def split_sequences(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split sequences into train/test partitions."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    split_idx = int(len(X) * train_ratio)
    if split_idx == 0 or split_idx == len(X):
        raise ValueError("train_ratio results in an empty split.")

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test
