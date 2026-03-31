"""Shared helpers for single-ticker strategy wrapper scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from regimes import main as create_regimes
except ModuleNotFoundError:
    from Code.regimes import main as create_regimes


def load_regime_data(input_path: Path, required_columns: list[str]) -> pd.DataFrame:
    """Load a regime-tagged market table, creating it first when needed."""
    if not input_path.exists():
        create_regimes()

    df = pd.read_csv(input_path)
    if df.empty or any(column not in df.columns for column in required_columns):
        create_regimes()
        df = pd.read_csv(input_path)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    numeric_columns = [column for column in required_columns if column not in {"Date", "regime"}]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    cleaned_df = (
        df.dropna(subset=required_columns)
        .sort_values("Date", ascending=True)
        .reset_index(drop=True)
    )
    return cleaned_df
