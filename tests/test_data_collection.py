from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fx_forecasting import data_collection
from fx_forecasting.data_collection import (
    fetch_financedatareader_history,
    merge_indicator_frames,
)


def test_fetch_financedatareader_history_normalizes(monkeypatch):
    calls = []

    def fake_reader(symbol, start=None, end=None):
        calls.append((symbol, start, end))
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        return pd.DataFrame({"Close": [1210.0, 1220.10, 1230.55]}, index=idx)

    monkeypatch.setattr(data_collection, "DataReader", fake_reader)
    df = fetch_financedatareader_history(
        "USD/KRW", start="2024-01-01", end="2024-01-03"
    )
    assert calls[0] == ("USD/KRW", "2024-01-01", "2024-01-03")
    assert list(df["date"].dt.strftime("%Y-%m-%d")) == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
    ]
    assert df["price"].tolist() == [1210.0, 1220.10, 1230.55]


def test_merge_indicator_frames_aligns_on_date():
    usd = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "price": [1210.0, 1220.1],
        }
    )
    idx = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "price": [102.5, 103.1],
        }
    )
    merged = merge_indicator_frames([("usd", usd), ("dxy", idx)])
    assert merged.shape == (3, 3)
    assert merged.columns.tolist() == ["date", "usd", "dxy"]
    assert merged.iloc[1]["usd"] == 1220.1
    assert merged.iloc[1]["dxy"] == 102.5


def main() -> None:
    print("\n=== fetch_financedatareader_history demo (mock) ===")
    try:
        df = fetch_financedatareader_history(
            "USD/KRW", start="2024-01-01", end="2024-01-03"
        )
        print(df)
    except ImportError:
        print("FinanceDataReader not installed; install to run this demo.")

    print("\n=== merge_indicator_frames demo ===")
    usd = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "price": [1210.0, 1220.1],
        }
    )
    idx = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "price": [102.5, 103.1],
        }
    )
    merged = merge_indicator_frames([("usd", usd), ("dxy", idx)])
    print(merged)


if __name__ == "__main__":
    main()
