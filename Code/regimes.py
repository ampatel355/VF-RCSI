"""Assign volatility regimes to one ticker's feature data."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from features import main as create_features
    from strategy_config import BREAKOUT_HIGH_COLUMN, BREAKOUT_LOW_COLUMN, MOMENTUM_RETURN_COLUMN
except ModuleNotFoundError:
    from Code.features import main as create_features
    from Code.strategy_config import (
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOW_COLUMN,
        MOMENTUM_RETURN_COLUMN,
    )


# Read the active ticker from the environment, or fall back to SPY.
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


def load_feature_data(input_path: Path) -> pd.DataFrame:
    """Load feature data, creating it first if the CSV does not exist yet."""
    if not input_path.exists():
        create_features()

    df = pd.read_csv(input_path)
    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "ma_20",
        "ma_50",
        "std_20",
        "price_std_20",
        "avg_volume_20",
        "zscore_20",
        MOMENTUM_RETURN_COLUMN,
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOW_COLUMN,
    ]
    if any(column not in df.columns for column in required_columns):
        create_features()
        df = pd.read_csv(input_path)

    # Convert Date into real datetime values so dates are handled correctly.
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Make sure important numeric columns are stored as numbers, not text.
    numeric_columns = [column for column in required_columns if column != "Date"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.dropna(subset=["Date", *numeric_columns]).reset_index(drop=True)


def main() -> None:
    # Find the project root folder so the script works from any starting directory.
    project_root = Path(__file__).resolve().parents[1]
    output_dir = resolve_data_clean_dir(project_root)

    # Build the input and output file paths using the shared ticker setting.
    input_path = output_dir / f"{ticker}_features.csv"
    output_path = output_dir / f"{ticker}_regimes.csv"

    # Create the output folder if it does not already exist.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the feature data that already contains std_20.
    df = load_feature_data(input_path)

    # Use only information available before each day when labeling regimes.
    # This avoids the forward-looking leakage created by full-sample quantiles.
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
    df = df.dropna(subset=["regime"]).reset_index(drop=True)

    # Save the finished data with the new regime column.
    df.to_csv(output_path, index=False)

    # Print the first 10 rows with the columns the user asked to see.
    print(df[["Date", "Close", "std_20", "regime"]].head(10))

    # Count how many rows fall into each regime.
    regime_counts = df["regime"].value_counts().reindex(
        ["calm", "neutral", "stressed"]
    )

    print("\nRows in each regime:")
    print(regime_counts)


if __name__ == "__main__":
    main()
