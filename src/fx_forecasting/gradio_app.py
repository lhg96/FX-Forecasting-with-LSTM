"""Gradio app that mirrors the LSTM workflow from ``최종 LSTM.ipynb``.

The notebook pulled USD/KRW, 달러지수, and CRB data, normalized them,
and trained an LSTM to forecast the next USD/KRW point.  This script
wraps the same idea behind a small UI:

- Upload a CSV/XLSX with a ``date`` column plus numeric feature columns
  (target defaults to ``USD/KRW``).
- Choose sequence length, split ratio, epochs, etc.
- Train a quick LSTM and see test-set MAE/R2 plus the next-day forecast.

If TensorFlow is not installed the app will short-circuit with a
friendly message.
"""

from __future__ import annotations

import io
from pathlib import Path
import sys
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gradio as gr

ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fx_forecasting.modeling import build_lstm_model, predict_sequences
from fx_forecasting.preprocessing import (
    build_minmax_scaler,
    create_supervised_sequences,
    forward_fill_by_date,
    split_sequences,
)

TARGET_COLUMN = "USD/KRW"


def _load_frame(file_obj) -> pd.DataFrame:
    """Load a CSV/XLSX upload into a DataFrame."""

    if file_obj is None:
        return _synthetic_frame()

    # Gradio 4.x: file_obj가 str(경로)일 수 있음
    if isinstance(file_obj, str):
        name = Path(file_obj).name.lower()
        with open(file_obj, "rb") as f:
            data = f.read()
        buf = io.BytesIO(data)
    else:
        name = Path(getattr(file_obj, "name", "upload")).name.lower()
        data = file_obj.read()
        buf = io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else file_obj

    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(buf)
    else:
        df = pd.read_csv(buf)
    return df


def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.copy()
    for col in numeric.columns:
        if col == "date":
            continue
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
    return numeric


def _synthetic_frame(rows: int = 200) -> pd.DataFrame:
    """Small trending dataset to let the UI run without external files."""

    idx = pd.date_range("2020-01-01", periods=rows, freq="D")
    usd = 1100 + np.cumsum(np.random.normal(0, 1.5, size=rows))
    dxy = 100 + np.cumsum(np.random.normal(0, 0.3, size=rows))
    crb = 180 + np.cumsum(np.random.normal(0, 0.5, size=rows))
    return pd.DataFrame({"date": idx, "USD/KRW": usd, "달러지수": dxy, "crb": crb})


def _inverse_scale_target(
    values: np.ndarray, scaler, columns: Iterable[str], target_column: str
) -> np.ndarray:
    cols = list(columns)
    if target_column not in cols:
        raise ValueError(f"Target '{target_column}' not found in columns.")
    idx = cols.index(target_column)
    data_min = scaler.data_min_[idx]
    data_max = scaler.data_max_[idx]
    return values * (data_max - data_min) + data_min


def train_and_forecast(
    file_obj,
    sequence_length: int = 10,
    train_ratio: float = 0.9,
    lstm_units: int = 200,
    epochs: int = 10,
    batch_size: int = 32,
) -> Tuple[str, plt.Figure]:
    """Core Gradio handler that trains and returns metrics/plot."""

    try:
        df_raw = _load_frame(file_obj)
    except Exception as exc:  # pragma: no cover - UI guardrail
        return f"❌ 파일을 불러오지 못했습니다: {exc}", None

    if "date" not in df_raw.columns:
        return "❌ 데이터에 'date' 컬럼이 없습니다.", None
    if TARGET_COLUMN not in df_raw.columns:
        return f"❌ '{TARGET_COLUMN}' 컬럼을 찾을 수 없습니다.", None

    df_clean = _ensure_numeric(df_raw)
    try:
        filled = forward_fill_by_date(df_clean)
    except Exception as exc:
        return f"❌ 날짜 정렬/결측 처리 중 오류: {exc}", None

    try:
        scaler, scaled = build_minmax_scaler(filled)
        X, y = create_supervised_sequences(
            scaled, sequence_length=sequence_length, target_column=TARGET_COLUMN
        )
        X_train, X_test, y_train, y_test = split_sequences(
            X, y, train_ratio=train_ratio
        )
    except Exception as exc:
        return f"❌ 전처리/시퀀스 생성 중 오류: {exc}", None

    try:
        model = build_lstm_model(
            (sequence_length, scaled.shape[1]), lstm_units=lstm_units
        )
    except ImportError:
        return "❌ TensorFlow가 설치되어 있지 않습니다. requirements.txt의 tensorflow를 설치하세요.", None

    callbacks = []
    try:  # pragma: no cover - only when TF is present
        import tensorflow as tf

        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )
        )
    except Exception:
        callbacks = []

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=int(epochs),
        batch_size=int(batch_size),
        verbose=0,
        callbacks=callbacks,
    )

    # 학습 후 loss 확인
    final_loss = history.history['loss'][-1] if 'loss' in history.history else 'N/A'
    final_val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else 'N/A'

    preds_scaled = predict_sequences(model, X_test)
    preds = _inverse_scale_target(preds_scaled, scaler, scaled.columns, TARGET_COLUMN)
    y_true = _inverse_scale_target(y_test, scaler, scaled.columns, TARGET_COLUMN)

    next_input = scaled.tail(sequence_length).values.reshape(1, sequence_length, -1)
    next_pred_scaled = predict_sequences(model, next_input)[0]
    next_pred = _inverse_scale_target(
        np.array([next_pred_scaled]), scaler, scaled.columns, TARGET_COLUMN
    )[0]

    mae = float(np.mean(np.abs(preds - y_true)))
    ss_res = float(np.sum((y_true - preds) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")

    fig, ax = plt.subplots(figsize=(8, 4))
    date_index = filled.index[-len(y_true):]
    ax.plot(date_index, y_true, label="Actual")
    ax.plot(date_index, preds, label="Predicted")
    ax.set_title("Test Set Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD/KRW")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # 진단용 정보 추가
    debug_info = (
        f"X_test.shape: {X_test.shape}\n"
        f"y_test.shape: {y_test.shape}\n"
        f"preds.shape: {preds.shape}\n"
        f"y_true.shape: {y_true.shape}\n"
        f"final_loss: {final_loss}\n"
        f"final_val_loss: {final_val_loss}\n"
        f"scaled에 nan 있음: {scaled.isna().any().any()}\n"
        f"X_test에 nan 있음: {np.isnan(X_test).any()}\n"
        f"y_test에 nan 있음: {np.isnan(y_test).any()}\n"
    )
    msg = (
        f"✅ 학습 완료\n"
        f"- 데이터 행 수: {len(filled)}\n"
        f"- 테스트 MAE: {mae:.4f}\n"
        f"- 테스트 R2: {r2:.4f}\n"
        f"- 다음 예측 USD/KRW: {next_pred:.4f}\n"
        f"\n[DEBUG INFO - 결측치 처리 확인]\n{debug_info}"
    )
    return msg, fig


def build_interface() -> gr.Blocks:
    # queue=False for compatibility with older Gradio (avoids queue/fn_index errors).
    with gr.Blocks(title="FX LSTM Forecaster") as demo:
        gr.Markdown(
            "### 환율 LSTM 예측\n"
            "- CSV/XLSX 업로드 또는 입력이 없으면 예제 데이터를 사용합니다.\n"
            "- `date`, `USD/KRW`, 그 외 지표(예: 달러지수, crb) 컬럼을 포함하세요."
        )
        with gr.Row():
            file_in = gr.File(label="데이터 파일 (CSV/XLSX)")
            seq = gr.Slider(5, 30, value=10, step=1, label="시퀀스 길이")
            train_ratio = gr.Slider(0.6, 0.95, value=0.9, step=0.01, label="학습 비율")
        with gr.Row():
            units = gr.Slider(32, 256, value=200, step=16, label="LSTM 유닛 수")
            epochs = gr.Slider(1, 50, value=10, step=1, label="에포크")
            batch = gr.Slider(8, 128, value=32, step=8, label="배치 크기")
        run_btn = gr.Button("학습 및 예측 실행")
        msg_out = gr.Markdown()
        plot_out = gr.Plot()
        run_btn.click(
            train_and_forecast,
            inputs=[file_in, seq, train_ratio, units, epochs, batch],
            outputs=[msg_out, plot_out],
        )
    return demo


if __name__ == "__main__":  # pragma: no cover
    app = build_interface()
    try:
        app.launch(enable_queue=False)
    except TypeError:
        # Older Gradio versions may not support enable_queue; fall back to defaults.
        app.launch()
