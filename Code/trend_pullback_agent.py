"""Run the Trend + Pullback strategy for the active ticker."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from single_ticker_agent_common import load_regime_data
    from strategy_config import (
        TREND_PULLBACK_STOP_LOW_COLUMN,
        TREND_PULLBACK_TARGET_HIGH_COLUMN,
    )
    from strategy_simulator import resolve_data_clean_dir, run_strategy
except ModuleNotFoundError:
    from Code.single_ticker_agent_common import load_regime_data
    from Code.strategy_config import (
        TREND_PULLBACK_STOP_LOW_COLUMN,
        TREND_PULLBACK_TARGET_HIGH_COLUMN,
    )
    from Code.strategy_simulator import resolve_data_clean_dir, run_strategy


ticker = os.environ.get("TICKER", "SPY")


def main() -> None:
    """Load regime-tagged data, run the strategy, and save the trade log."""
    project_root = Path(__file__).resolve().parents[1]
    data_clean_dir = resolve_data_clean_dir(project_root)
    input_path = data_clean_dir / f"{ticker}_regimes.csv"
    output_path = data_clean_dir / f"{ticker}_trend_pullback_trades.csv"

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "ma_20",
        "ma_50",
        "ma_200",
        "rsi_14",
        "atr_14",
        TREND_PULLBACK_STOP_LOW_COLUMN,
        TREND_PULLBACK_TARGET_HIGH_COLUMN,
        "avg_volume_20",
        "regime",
    ]
    df = load_regime_data(input_path, required_columns)
    df.attrs["ticker"] = ticker

    trades_df = run_strategy("trend_pullback", df, ticker=ticker)
    trades_df.to_csv(output_path, index=False)

    print(f"Total number of trades: {len(trades_df)}")
    print("\nFirst 10 trades:")
    print(trades_df.head(10))


if __name__ == "__main__":
    main()
