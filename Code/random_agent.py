"""Simulate a simple random trading agent with next-bar execution."""

import os
from pathlib import Path

import pandas as pd

try:
    from regimes import main as create_regimes
    from strategy_simulator import (
        ENTRY_PROBABILITY,
        EXIT_PROBABILITY,
        RANDOM_AGENT_REPRODUCIBLE,
        RANDOM_AGENT_SEED,
        resolve_data_clean_dir,
        run_strategy,
    )
except ModuleNotFoundError:
    from Code.regimes import main as create_regimes
    from Code.strategy_simulator import (
        ENTRY_PROBABILITY,
        EXIT_PROBABILITY,
        RANDOM_AGENT_REPRODUCIBLE,
        RANDOM_AGENT_SEED,
        resolve_data_clean_dir,
        run_strategy,
    )


# Read the active ticker from the environment, or fall back to SPY.
ticker = os.environ.get("TICKER", "SPY")


def load_regime_data(input_path: Path) -> pd.DataFrame:
    """Load regime data, creating it first if the CSV does not exist yet."""
    if not input_path.exists():
        create_regimes()

    df = pd.read_csv(input_path)
    required_columns = ["Date", "Open", "Close", "avg_volume_20", "regime"]
    if any(column not in df.columns for column in required_columns):
        create_regimes()
        df = pd.read_csv(input_path)

    # Convert Date to datetime so the dates are handled correctly.
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Make sure the price columns are numeric so returns can be calculated safely.
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["avg_volume_20"] = pd.to_numeric(df["avg_volume_20"], errors="coerce")

    # Remove rows missing important values, then sort from oldest to newest.
    return (
        df.dropna(subset=["Date", "Open", "Close", "avg_volume_20", "regime"])
        .sort_values("Date", ascending=True)
        .reset_index(drop=True)
    )

def main() -> None:
    # Find the project root so the script works from any starting directory.
    project_root = Path(__file__).resolve().parents[1]
    data_clean_dir = resolve_data_clean_dir(project_root)

    # Build the input and output file paths using the shared ticker setting.
    input_path = data_clean_dir / f"{ticker}_regimes.csv"
    output_path = data_clean_dir / f"{ticker}_random_trades.csv"

    # Load the regime data that includes Date, Close, and regime columns.
    df = load_regime_data(input_path)

    decision_seed = RANDOM_AGENT_SEED if RANDOM_AGENT_REPRODUCIBLE else None
    trades_df = run_strategy("random", df, random_decision_seed=decision_seed)

    # Save only completed trades. If a trade is still open at the end,
    # it is not saved because there is no exit yet.
    trades_df.to_csv(output_path, index=False)

    # Print a simple summary so the result is easy to inspect.
    print(f"Total number of trades: {len(trades_df)}")
    print(f"Random agent reproducible mode: {RANDOM_AGENT_REPRODUCIBLE}")
    print(f"Random agent seed used: {RANDOM_AGENT_SEED if RANDOM_AGENT_REPRODUCIBLE else 'None'}")
    print(f"Entry probability: {ENTRY_PROBABILITY:.3f}")
    print(f"Exit probability: {EXIT_PROBABILITY:.3f}")
    print("\nFirst 10 trades:")
    print(trades_df.head(10))


if __name__ == "__main__":
    main()
