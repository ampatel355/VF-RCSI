"""Assign forward-safe volatility regimes to one ticker's feature table."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from features import build_features_for_ticker
    from strategy_config import (
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOW_COLUMN,
        BREAKOUT_MOMENTUM_RETURN_COLUMN,
        MEAN_REVERSION_STOP_LOW_COLUMN,
        RELATIVE_STRENGTH_RETURN_COLUMN,
        TREND_PULLBACK_STOP_LOW_COLUMN,
        TREND_PULLBACK_TARGET_HIGH_COLUMN,
        VOL_MANAGED_TSMOM_ENTRY_RETURN_COLUMN,
        VOL_MANAGED_TSMOM_EXIT_RETURN_COLUMN,
        VOL_MANAGED_TSMOM_VOL_COLUMN,
    )
except ModuleNotFoundError:
    from Code.features import build_features_for_ticker
    from Code.strategy_config import (
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOW_COLUMN,
        BREAKOUT_MOMENTUM_RETURN_COLUMN,
        MEAN_REVERSION_STOP_LOW_COLUMN,
        RELATIVE_STRENGTH_RETURN_COLUMN,
        TREND_PULLBACK_STOP_LOW_COLUMN,
        TREND_PULLBACK_TARGET_HIGH_COLUMN,
        VOL_MANAGED_TSMOM_ENTRY_RETURN_COLUMN,
        VOL_MANAGED_TSMOM_EXIT_RETURN_COLUMN,
        VOL_MANAGED_TSMOM_VOL_COLUMN,
    )


ticker = os.environ.get("TICKER", "SPY")
MIN_REGIME_HISTORY = int(os.environ.get("MIN_REGIME_HISTORY", "252"))


def resolve_data_clean_dir(project_root: Path) -> Path:
    """Return the project's clean-data folder, supporting either naming style."""
    lowercase_dir = project_root / "data_clean"
    uppercase_dir = project_root / "Data_Clean"

    if lowercase_dir.exists():
        return lowercase_dir
    if uppercase_dir.exists():
        return uppercase_dir

    lowercase_dir.mkdir(parents=True, exist_ok=True)
    return lowercase_dir


def load_feature_data(input_path: Path, current_ticker: str | None = None) -> pd.DataFrame:
    """Load one feature table, creating it first when needed."""
    current_ticker = current_ticker or ticker
    if not input_path.exists():
        build_features_for_ticker(current_ticker, save_output=True)

    df = pd.read_csv(input_path)
    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "daily_return",
        "ma_20",
        "ma_50",
        "ma_100",
        "ma_200",
        "std_20",
        "price_std_20",
        "avg_volume_20",
        "volume_ratio_20",
        "rsi_14",
        "atr_14",
        "atr_percent",
        "bollinger_mid",
        "bollinger_upper",
        "bollinger_lower",
        "bollinger_width",
        "macd_line",
        "macd_signal",
        "macd_hist",
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOW_COLUMN,
        BREAKOUT_MOMENTUM_RETURN_COLUMN,
        VOL_MANAGED_TSMOM_ENTRY_RETURN_COLUMN,
        VOL_MANAGED_TSMOM_EXIT_RETURN_COLUMN,
        VOL_MANAGED_TSMOM_VOL_COLUMN,
        RELATIVE_STRENGTH_RETURN_COLUMN,
        TREND_PULLBACK_STOP_LOW_COLUMN,
        TREND_PULLBACK_TARGET_HIGH_COLUMN,
        MEAN_REVERSION_STOP_LOW_COLUMN,
        "atr_percent_ratio_60",
        "bollinger_width_ratio_60",
        "distance_from_ma_20",
        "distance_from_ma_50",
        "ma_50_over_ma_200",
        "zscore_20",
    ]
    if df.empty or any(column not in df.columns for column in required_columns):
        df = build_features_for_ticker(current_ticker, save_output=True)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    numeric_columns = [column for column in required_columns if column != "Date"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return (
        df.dropna(subset=["Date", *numeric_columns])
        .sort_values("Date", ascending=True)
        .reset_index(drop=True)
    )


def add_regime_labels(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Add calm / neutral / stressed labels using only past information."""
    df = feature_df.copy()

    # We use expanding quantiles of recent close-to-close volatility. Each day's
    # thresholds are shifted by one bar, so the label for today depends only on
    # history that was already known before today's close.
    lower_quantile = (
        df["std_20"]
        .expanding(min_periods=MIN_REGIME_HISTORY)
        .quantile(1 / 3)
        .shift(1)
    )
    upper_quantile = (
        df["std_20"]
        .expanding(min_periods=MIN_REGIME_HISTORY)
        .quantile(2 / 3)
        .shift(1)
    )

    df["regime"] = pd.Series(np.nan, index=df.index, dtype="object")
    valid_rows = ~(lower_quantile.isna() | upper_quantile.isna())
    df.loc[valid_rows, "regime"] = "neutral"
    df.loc[valid_rows & (df["std_20"] <= lower_quantile), "regime"] = "calm"
    df.loc[valid_rows & (df["std_20"] >= upper_quantile), "regime"] = "stressed"
    return df.dropna(subset=["regime"]).reset_index(drop=True)


def build_regime_dataframe_for_ticker(
    current_ticker: str,
    *,
    project_root: Path | None = None,
    save_output: bool = True,
) -> pd.DataFrame:
    """Build the regime table for one ticker and optionally save it."""
    project_root = project_root or Path(__file__).resolve().parents[1]
    output_dir = resolve_data_clean_dir(project_root)
    input_path = output_dir / f"{current_ticker}_features.csv"
    output_path = output_dir / f"{current_ticker}_regimes.csv"

    feature_df = load_feature_data(input_path, current_ticker=current_ticker)
    regime_df = add_regime_labels(feature_df)

    if save_output:
        output_dir.mkdir(parents=True, exist_ok=True)
        regime_df.to_csv(output_path, index=False)

    return regime_df


def main() -> None:
    """Create the regime-tagged dataset for the active ticker."""
    regime_df = build_regime_dataframe_for_ticker(ticker)
    print(regime_df[["Date", "Close", "std_20", "regime"]].head(10))
    print("\nRows in each regime:")
    print(regime_df["regime"].value_counts().reindex(["calm", "neutral", "stressed"]))


if __name__ == "__main__":
    main()
