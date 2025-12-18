"""Core package for the FX forecasting project.

This module exposes utilities that transform the original notebook workflow
into importable Python modules that can be unit tested.
"""

from .data_collection import fetch_investing_history, merge_indicator_frames
from .preprocessing import (
    forward_fill_by_date,
    create_supervised_sequences,
    split_sequences,
    build_minmax_scaler,
)
from .modeling import build_lstm_model, predict_sequences

__all__ = [
    "fetch_investing_history",
    "merge_indicator_frames",
    "forward_fill_by_date",
    "create_supervised_sequences",
    "split_sequences",
    "build_minmax_scaler",
    "build_lstm_model",
    "predict_sequences",
]
