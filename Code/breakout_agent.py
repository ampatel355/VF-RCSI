"""Simulate a volatility breakout trading agent with next-bar execution."""

import os
from pathlib import Path

import pandas as pd

try:
    from regimes import main as create_regimes
    from strategy_config import (
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOOKBACK_DAYS,
        BREAKOUT_LOW_COLUMN,
    )
    from strategy_simulator import resolve_data_clean_dir, run_strategy
except ModuleNotFoundError:
    from Code.regimes import main as create_regimes
    from Code.strategy_config import (
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOOKBACK_DAYS,
        BREAKOUT_LOW_COLUMN,
    )
    from Code.strategy_simulator import resolve_data_clean_dir, run_strategy


ticker = os.environ.get("TICKER", "SPY")


def load_regime_data(input_path: Path) -> pd.DataFrame:
    """Load regime data, creating it first if the CSV does not exist yet."""
    if not input_path.exists():
        create_regimes()

    df = pd.read_csv(input_path)
    required_columns = [
        "Date",
        "Open",
        "Close",
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOW_COLUMN,
        "avg_volume_20",
        "regime",
    ]
    if any(column not in df.columns for column in required_columns):
        create_regimes()
        df = pd.read_csv(input_path)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df[BREAKOUT_HIGH_COLUMN] = pd.to_numeric(df[BREAKOUT_HIGH_COLUMN], errors="coerce")
    df[BREAKOUT_LOW_COLUMN] = pd.to_numeric(df[BREAKOUT_LOW_COLUMN], errors="coerce")
    df["avg_volume_20"] = pd.to_numeric(df["avg_volume_20"], errors="coerce")

    return (
        df.dropna(subset=required_columns)
        .sort_values("Date", ascending=True)
        .reset_index(drop=True)
    )


def main() -> None:
    """Run the breakout strategy and save its completed trades."""
    project_root = Path(__file__).resolve().parents[1]
    data_clean_dir = resolve_data_clean_dir(project_root)
    input_path = data_clean_dir / f"{ticker}_regimes.csv"
    output_path = data_clean_dir / f"{ticker}_breakout_trades.csv"

    df = load_regime_data(input_path)
    trades_df = run_strategy("breakout", df)
    trades_df.to_csv(output_path, index=False)

    print(f"Breakout lookback days: {BREAKOUT_LOOKBACK_DAYS}")
    print(f"Total number of trades: {len(trades_df)}")
    print("\nFirst 10 trades:")
    print(trades_df.head(10))


if __name__ == "__main__":
    main()
