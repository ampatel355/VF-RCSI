"""Simulate a simple mean-reversion trading agent with next-bar execution."""

import os
from pathlib import Path

import pandas as pd

try:
    from regimes import main as create_regimes
    from strategy_simulator import resolve_data_clean_dir, run_strategy
except ModuleNotFoundError:
    from Code.regimes import main as create_regimes
    from Code.strategy_simulator import resolve_data_clean_dir, run_strategy


# Read the active ticker from the environment, or fall back to SPY.
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
        "ma_20",
        "price_std_20",
        "zscore_20",
        "avg_volume_20",
        "regime",
    ]
    if any(column not in df.columns for column in required_columns):
        create_regimes()
        df = pd.read_csv(input_path)

    # Convert the Date column into datetime values so dates sort correctly
    # and are saved in a clean, consistent format.
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Make sure the columns used in the trading rules are numeric values.
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["ma_20"] = pd.to_numeric(df["ma_20"], errors="coerce")
    df["price_std_20"] = pd.to_numeric(df["price_std_20"], errors="coerce")
    df["zscore_20"] = pd.to_numeric(df["zscore_20"], errors="coerce")
    df["avg_volume_20"] = pd.to_numeric(df["avg_volume_20"], errors="coerce")

    # Remove rows with missing values in important columns and sort oldest first.
    return (
        df.dropna(
            subset=[
                "Date",
                "Open",
                "Close",
                "ma_20",
                "price_std_20",
                "zscore_20",
                "avg_volume_20",
                "regime",
            ]
        )
        .sort_values("Date", ascending=True)
        .reset_index(drop=True)
    )


def main() -> None:
    # Find the project root so the script works from any starting directory.
    project_root = Path(__file__).resolve().parents[1]
    data_clean_dir = resolve_data_clean_dir(project_root)

    # Build the input and output file paths using the shared ticker setting.
    input_path = data_clean_dir / f"{ticker}_regimes.csv"
    output_path = data_clean_dir / f"{ticker}_mean_reversion_trades.csv"

    # Load the daily data that already includes the moving average,
    # volatility, and regime columns.
    df = load_regime_data(input_path)

    trades_df = run_strategy("mean_reversion", df)

    # Save only completed trades. If a trade is still open at the end,
    # it is not included because there is no exit yet.
    trades_df.to_csv(output_path, index=False)

    # Print a summary so the results are easy to inspect.
    print(f"Total number of trades: {len(trades_df)}")
    print("\nFirst 10 trades:")
    print(trades_df.head(10))


if __name__ == "__main__":
    main()
