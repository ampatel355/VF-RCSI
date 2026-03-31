"""Run the Breakout + Volume + Momentum strategy for the active ticker."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from single_ticker_agent_common import load_regime_data
    from strategy_config import (
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOW_COLUMN,
        BREAKOUT_MOMENTUM_RETURN_COLUMN,
    )
    from strategy_simulator import resolve_data_clean_dir, run_strategy
except ModuleNotFoundError:
    from Code.single_ticker_agent_common import load_regime_data
    from Code.strategy_config import (
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOW_COLUMN,
        BREAKOUT_MOMENTUM_RETURN_COLUMN,
    )
    from Code.strategy_simulator import resolve_data_clean_dir, run_strategy


ticker = os.environ.get("TICKER", "SPY")


def main() -> None:
    """Load regime-tagged data, run the strategy, and save the trade log."""
    project_root = Path(__file__).resolve().parents[1]
    data_clean_dir = resolve_data_clean_dir(project_root)
    input_path = data_clean_dir / f"{ticker}_regimes.csv"
    output_path = data_clean_dir / f"{ticker}_breakout_volume_momentum_trades.csv"

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "avg_volume_20",
        "volume_ratio_20",
        "macd_line",
        "macd_signal",
        "atr_14",
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOW_COLUMN,
        BREAKOUT_MOMENTUM_RETURN_COLUMN,
        "regime",
    ]
    df = load_regime_data(input_path, required_columns)
    df.attrs["ticker"] = ticker

    trades_df = run_strategy("breakout_volume_momentum", df, ticker=ticker)
    trades_df.to_csv(output_path, index=False)

    print(f"Total number of trades: {len(trades_df)}")
    print("\nFirst 10 trades:")
    print(trades_df.head(10))


if __name__ == "__main__":
    main()
