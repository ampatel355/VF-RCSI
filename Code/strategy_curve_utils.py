"""Helpers for loading saved strategy curves or reconstructing them when needed."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from plot_config import data_clean_dir, load_csv_checked
except ModuleNotFoundError:
    from Code.plot_config import data_clean_dir, load_csv_checked


def strategy_curve_path(ticker: str, agent_name: str) -> Path:
    """Return the optional saved curve path for one strategy."""
    return data_clean_dir() / f"{ticker.upper()}_{agent_name}_curve.csv"


def load_saved_strategy_curve(ticker: str, agent_name: str) -> pd.DataFrame | None:
    """Load a saved strategy curve when it exists and contains the expected columns."""
    input_path = strategy_curve_path(ticker, agent_name)
    if not input_path.exists():
        return None

    df = load_csv_checked(
        input_path,
        required_columns=[
            "Date",
            "Close",
            "equity",
            "wealth_index",
            "daily_return",
            "cumulative_return",
            "rolling_peak",
            "drawdown",
        ],
    )
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    numeric_columns = [
        "Close",
        "equity",
        "wealth_index",
        "daily_return",
        "cumulative_return",
        "rolling_peak",
        "drawdown",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "bar_return" not in df.columns:
        df["bar_return"] = df["daily_return"]

    return df.dropna(subset=["Date", "equity", "cumulative_return"]).sort_values("Date").reset_index(drop=True)
