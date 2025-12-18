"""Utilities for collecting macro-economic indicators from HTML tables.

The original notebook relied on copy-pasted CSS selectors to obtain values
from Investing.com.  Those selectors change frequently, which made the
workflow brittle and impossible to test.  The helpers below focus on parsing
the rendered HTML table structure instead so we can run deterministic unit
tests with canned HTML fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
try:  # FinanceDataReader is the preferred source for FX data.
    from FinanceDataReader import DataReader
except ImportError:  # pragma: no cover - handled at runtime
    DataReader = None  # type: ignore

DEFAULT_HEADERS: MutableMapping[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/118.0 Safari/537.36"
    )
}


def _clean_numeric(value: str) -> float:
    """Convert a numeric string that may contain thousands separators."""
    cleaned = value.replace(",", "").replace("−", "-")
    return pd.to_numeric(cleaned, errors="coerce")


def parse_investing_history(html: str, date_parser: Optional[Callable[[str], str]] = None) -> pd.DataFrame:
    """Parse Investing.com style history tables into a DataFrame.

    Parameters
    ----------
    html:
        Raw HTML string that contains a single historical data table.
    date_parser:
        Optional callable that receives the date string and should return a
        normalized date representation (string or datetime).
    """

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError("HTML snippet does not contain a <table> element.")

    body = table.find("tbody")
    rows = body.find_all("tr") if body else table.find_all("tr")

    parsed_rows = []
    for row in rows:
        cols = row.find_all(["td", "th"])
        if len(cols) < 2:
            continue
        raw_date = cols[0].get_text(strip=True)
        raw_price = cols[1].get_text(strip=True)
        if not raw_date or not raw_price:
            continue
        parsed_rows.append((raw_date, raw_price))

    if not parsed_rows:
        raise ValueError("Table did not contain any parsable rows.")

    df = pd.DataFrame(parsed_rows, columns=["date", "price"])
    if date_parser:
        df["date"] = df["date"].apply(date_parser)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)

    df["price"] = df["price"].apply(_clean_numeric)
    df.dropna(subset=["price"], inplace=True)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def fetch_investing_history(
    url: str,
    *,
    session: Optional[requests.Session] = None,
    date_parser: Optional[Callable[[str], str]] = None,
) -> pd.DataFrame:
    """Download and parse Investing.com data while injecting default headers.

    Note: legacy path kept for compatibility; FinanceDataReader is preferred.
    """

    sess = session or requests.Session()
    response = sess.get(url, headers=DEFAULT_HEADERS, timeout=30)
    response.raise_for_status()
    return parse_investing_history(response.text, date_parser)


def merge_indicator_frames(frames: Sequence[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Merge indicator frames on their ``date`` column using outer joins."""

    merged: Optional[pd.DataFrame] = None
    for name, frame in frames:
        if "date" not in frame.columns:
            raise ValueError(f"Frame for {name} is missing a 'date' column.")
        if frame.empty:
            continue
        candidate = frame.copy()
        value_col = "price" if "price" in candidate.columns else candidate.columns[-1]
        candidate = candidate[["date", value_col]].rename(columns={value_col: name})
        candidate["date"] = pd.to_datetime(candidate["date"], errors="coerce")
        candidate.dropna(subset=["date"], inplace=True)
        if merged is None:
            merged = candidate
        else:
            merged = pd.merge(merged, candidate, on="date", how="outer")

    if merged is None:
        raise ValueError("No frames supplied to merge.")

    merged.sort_values("date", inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged


@dataclass
class IndicatorConfig:
    """Configuration for an indicator fetched via FinanceDataReader."""

    name: str
    symbol: str
    start: Optional[str] = None
    end: Optional[str] = None


def fetch_financedatareader_history(
    symbol: str, start: Optional[str] = None, end: Optional[str] = None
) -> pd.DataFrame:
    """Fetch history via FinanceDataReader and normalize columns."""

    if DataReader is None:
        raise ImportError("FinanceDataReader is not installed.")
    df = DataReader(symbol, start, end)
    if df.empty:
        raise ValueError(f"No data returned for symbol '{symbol}'.")
    date_col = df.index.name or "index"
    df = df.reset_index().rename(
        columns={
            date_col: "date",
            "Date": "date",
            "Close": "price",
        }
    )
    if "price" not in df.columns:
        # If Close not present, try the last numeric column.
        numeric_cols = [c for c in df.columns if c != "date"]
        if not numeric_cols:
            raise ValueError("No numeric columns found to use as price.")
        df = df.rename(columns={numeric_cols[-1]: "price"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df = df[["date", "price"]].copy()
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def fetch_many_indicators(configs: Iterable[IndicatorConfig]) -> pd.DataFrame:
    """Fetch and merge several indicators described by ``configs`` using FDR."""

    frames = []
    for cfg in configs:
        frame = fetch_financedatareader_history(
            cfg.symbol, start=cfg.start, end=cfg.end
        )
        frames.append((cfg.name, frame))
    return merge_indicator_frames(frames)


def save_indicator_data(
    configs: Iterable[IndicatorConfig],
    output_path: str,
    *,
    to_excel: bool = False,
    target_column: str = "USD/KRW",
) -> pd.DataFrame:
    """Fetch indicators, merge, and persist to disk.

    Parameters
    ----------
    configs:
        Iterable of IndicatorConfig entries (name, url, optional date parser).
    output_path:
        Destination filepath. ``.csv`` or ``.xlsx`` are supported.
    to_excel:
        Force Excel output even if the extension is not ``.xlsx``.
    target_column:
        Name of the primary target column; rows missing this value are dropped.

    Returns
    -------
    pd.DataFrame
        The merged indicator frame that was saved.
    """

    df = fetch_many_indicators(configs)
    if target_column in df.columns:
        df = df.dropna(subset=[target_column])
    path = pd.io.common.stringify_path(output_path)
    if to_excel or str(path).lower().endswith(".xlsx"):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)
    return df


if __name__ == "__main__":  # pragma: no cover - convenience CLI
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch Investing.com indicators and save merged data."
    )
    parser.add_argument(
        "--usd-krw-symbol",
        default="USD/KRW",
        help="FinanceDataReader symbol for USD/KRW history.",
    )
    parser.add_argument(
        "--dxy-symbol",
        default="DX-Y.NYB",
        help="FinanceDataReader symbol for Dollar Index (DXY).",
    )
    parser.add_argument(
        "--crb-symbol",
        default="CRB",
        help="FinanceDataReader symbol for CRB index.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="indicators.csv",
        help="Output file path (.csv or .xlsx).",
    )
    parser.add_argument(
        "--excel",
        action="store_true",
        help="Force Excel output regardless of extension.",
    )
    args = parser.parse_args()

    configs = [
        IndicatorConfig(name="USD/KRW", symbol=args.usd_krw_symbol),
        IndicatorConfig(name="달러지수", symbol=args.dxy_symbol),
        IndicatorConfig(name="crb", symbol=args.crb_symbol),
    ]
    saved = save_indicator_data(
        configs, args.output, to_excel=args.excel, target_column="USD/KRW"
    )
    print(f"Saved {len(saved)} rows to {args.output}")
