import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

if __name__ != "__main__":
    pytest.importorskip("tensorflow")

from fx_forecasting.modeling import build_lstm_model, predict_sequences


def test_build_lstm_model_produces_predictions():
    model = build_lstm_model((3, 2))
    dummy_sequences = np.random.rand(4, 3, 2).astype(np.float32)
    preds = predict_sequences(model, dummy_sequences)
    assert preds.shape == (4,)


def main() -> None:
    if importlib.util.find_spec("tensorflow") is None:
        print("TensorFlow is not installed; install it to run this demo.")
        return
    model = build_lstm_model((3, 2))
    dummy_sequences = np.random.rand(4, 3, 2).astype(np.float32)
    preds = predict_sequences(model, dummy_sequences)
    print("Predictions:", preds)


if __name__ == "__main__":
    main()
