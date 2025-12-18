"""Model helpers built on top of TensorFlow/Keras."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

try:  # pragma: no cover - handled by unit tests through importorskip.
    import tensorflow as tf
except ImportError:  # pragma: no cover
    tf = None


def _require_tensorflow() -> None:
    if tf is None:
        raise ImportError(
            "TensorFlow is not installed. Install the 'tensorflow' extra "
            "defined in requirements.txt to use modeling utilities."
        )


def build_lstm_model(
    input_shape: Iterable[int],
    lstm_units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> "tf.keras.Model":
    """Create and compile a small LSTM regression model."""

    _require_tensorflow()
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=tuple(input_shape)),
            tf.keras.layers.LSTM(lstm_units, return_sequences=False),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def predict_sequences(
    model: "tf.keras.Model",
    sequences: np.ndarray,
    *,
    inverse_transformer: Optional[object] = None,
) -> np.ndarray:
    """Run inference and optionally invert the scaling."""

    _require_tensorflow()
    preds = model.predict(sequences, verbose=0)
    if inverse_transformer is not None:
        preds = inverse_transformer.inverse_transform(preds)
    return preds.squeeze(-1)
