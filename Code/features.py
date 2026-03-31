"""Create basic feature columns from a ticker's closing prices."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from data_loader import main as download_raw_data
    from strategy_config import (
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOOKBACK_DAYS,
        BREAKOUT_LOW_COLUMN,
        MOMENTUM_LOOKBACK_DAYS,
        MOMENTUM_RETURN_COLUMN,
    )
except ModuleNotFoundError:
    from Code.data_loader import main as download_raw_data
    from Code.strategy_config import (
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOOKBACK_DAYS,
        BREAKOUT_LOW_COLUMN,
        MOMENTUM_LOOKBACK_DAYS,
        MOMENTUM_RETURN_COLUMN,
    )


# Read the active ticker from the environment, or fall back to SPY.
ticker = os.environ.get("TICKER", "SPY")


def resolve_data_raw_dir(project_root: Path) -> Path:
    """Return the project's raw-data folder, supporting either naming style."""
    lowercase_dir = project_root / "data_raw"
    uppercase_dir = project_root / "Data_Raw"

    if lowercase_dir.exists():
        return lowercase_dir
    if uppercase_dir.exists():
        return uppercase_dir

    lowercase_dir.mkdir(parents=True, exist_ok=True)
    return lowercase_dir


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


def load_raw_prices(input_path: Path) -> pd.DataFrame:
    """Load raw prices, downloading them first if the CSV does not exist yet."""
    if not input_path.exists():
        # If the raw CSV is missing, download data for the chosen ticker first.
        download_raw_data(ticker)

    # If the file still does not exist, stop with a clear message.
    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find or create the raw price file: {input_path}"
        )

    df = pd.read_csv(input_path)

    # Older CSVs may contain an extra ticker row beneath the header; coerce bad rows away.
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.dropna(subset=["Date", "Close"]).reset_index(drop=True)


def main() -> None:
    # Find the project root folder so the script works no matter where it is run from.
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = resolve_data_raw_dir(project_root)
    output_dir = resolve_data_clean_dir(project_root)

    # Build the input and output file paths using the shared ticker setting.
    input_path = raw_dir / f"{ticker}.csv"
    output_path = output_dir / f"{ticker}_features.csv"

    # Create the output folder if it does not already exist.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the raw price data from the CSV file for the chosen ticker.
    df = load_raw_prices(input_path)

    # Sort the rows from oldest date to newest date.
    df = df.sort_values("Date", ascending=True)

    # daily_return compares today's close to yesterday's close.
    df["daily_return"] = df["Close"].pct_change()

    # Rolling means calculate the average Close over the last 20 or 50 trading days.
    df["ma_20"] = df["Close"].rolling(window=20).mean()
    df["ma_50"] = df["Close"].rolling(window=50).mean()

    # std_20 measures how much daily returns have been moving around over 20 days.
    df["std_20"] = df["daily_return"].rolling(window=20).std(ddof=0)

    # price_std_20 measures the rolling dispersion of the price itself.
    # This is the correct scale to use for Bollinger-style mean-reversion bands.
    df["price_std_20"] = df["Close"].rolling(window=20).std(ddof=0)

    # avg_volume_20 is a simple liquidity proxy known at the close.
    df["avg_volume_20"] = df["Volume"].rolling(window=20).mean()

    # zscore_20 measures how far price sits above or below its 20-day mean.
    df["zscore_20"] = np.where(
        df["price_std_20"] > 0,
        (df["Close"] - df["ma_20"]) / df["price_std_20"],
        np.nan,
    )

    # Time-series momentum compares today's close with the close from the
    # chosen lookback window. This uses only historical information.
    df[MOMENTUM_RETURN_COLUMN] = (df["Close"] / df["Close"].shift(MOMENTUM_LOOKBACK_DAYS)) - 1.0

    # Breakout thresholds use the prior rolling high/low, shifted by one day,
    # so the signal never includes the current bar in the threshold itself.
    df[BREAKOUT_HIGH_COLUMN] = (
        df["High"].rolling(window=BREAKOUT_LOOKBACK_DAYS).max().shift(1)
    )
    df[BREAKOUT_LOW_COLUMN] = (
        df["Low"].rolling(window=BREAKOUT_LOOKBACK_DAYS).min().shift(1)
    )

    # Remove the early rows that do not have enough past data for the rolling windows.
    df = df.dropna().reset_index(drop=True)

    # Save the finished feature data to a new CSV file.
    df.to_csv(output_path, index=False)

    # Show the first 10 rows so it is easy to check the result.
    print(df.head(10))


if __name__ == "__main__":
    main()
