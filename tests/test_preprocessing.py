import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fx_forecasting.preprocessing import (
    build_minmax_scaler,
    create_supervised_sequences,
    forward_fill_by_date,
    split_sequences,
)


def test_forward_fill_by_date_orders_and_fills():
    df = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-01", "2024-01-02"],
            "USD/KRW": [1230.0, 1210.0, np.nan],
            "달러지수": [103.0, np.nan, 102.0],
        }
    )
    filled = forward_fill_by_date(df)
    assert filled.index[0].strftime("%Y-%m-%d") == "2024-01-01"
    assert filled.iloc[1]["USD/KRW"] == 1210.0
    assert filled.iloc[1]["달러지수"] == 102.0


def test_create_supervised_sequences_returns_expected_shape():
    df = pd.DataFrame(
        {
            "USD/KRW": [1, 2, 3, 4, 5],
            "달러지수": [10, 11, 12, 13, 14],
        },
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
    )
    X, y = create_supervised_sequences(df, sequence_length=2, target_column="USD/KRW")
    assert X.shape == (3, 2, 2)
    assert np.array_equal(y, np.array([3, 4, 5]))


def test_build_minmax_scaler_scales_between_bounds():
    df = pd.DataFrame(
        {
            "USD/KRW": [1.0, 2.0],
            "달러지수": [10.0, 20.0],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    scaler, scaled = build_minmax_scaler(df)
    assert scaled.min().min() == 0.0
    assert scaled.max().max() == 1.0
    restored = pd.DataFrame(
        scaler.inverse_transform(scaled),
        index=df.index,
        columns=df.columns,
    )
    pd.testing.assert_frame_equal(restored, df)


def test_split_sequences_respects_ratio():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    X_train, X_test, y_train, y_test = split_sequences(X, y, train_ratio=0.6)
    assert X_train.shape[0] == 6
    assert X_test.shape[0] == 4
    assert y_train[-1] == 5
    assert y_test[0] == 6


def main() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-01", "2024-01-02"],
            "USD/KRW": [1230.0, 1210.0, np.nan],
            "달러지수": [103.0, np.nan, 102.0],
        }
    )
    print("\n=== forward_fill_by_date ===")
    filled = forward_fill_by_date(df)
    print(filled)

    series_df = pd.DataFrame(
        {
            "USD/KRW": [1, 2, 3, 4, 5],
            "달러지수": [10, 11, 12, 13, 14],
        },
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
    )
    print("\n=== create_supervised_sequences ===")
    X, y = create_supervised_sequences(
        series_df, sequence_length=2, target_column="USD/KRW"
    )
    print(f"X shape: {X.shape}, y: {y}")

    print("\n=== build_minmax_scaler ===")
    scaler, scaled = build_minmax_scaler(series_df)
    print(scaled)

    print("\n=== split_sequences ===")
    X_train, X_test, y_train, y_test = split_sequences(X, y, train_ratio=0.6)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"y_train: {y_train}, y_test: {y_test}")


if __name__ == "__main__":
    main()
