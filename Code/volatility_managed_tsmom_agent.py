"""Run the Volatility-Managed TSMOM strategy for the active ticker."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from single_ticker_agent_common import load_regime_data
    from strategy_config import (
        VOL_MANAGED_TSMOM_ENTRY_RETURN_COLUMN,
        VOL_MANAGED_TSMOM_EXIT_RETURN_COLUMN,
        VOL_MANAGED_TSMOM_VOL_COLUMN,
    )
    from strategy_simulator import resolve_data_clean_dir, run_strategy
except ModuleNotFoundError:
    from Code.single_ticker_agent_common import load_regime_data
    from Code.strategy_config import (
        VOL_MANAGED_TSMOM_ENTRY_RETURN_COLUMN,
        VOL_MANAGED_TSMOM_EXIT_RETURN_COLUMN,
        VOL_MANAGED_TSMOM_VOL_COLUMN,
    )
    from Code.strategy_simulator import resolve_data_clean_dir, run_strategy


ticker = os.environ.get("TICKER", "SPY")


def main() -> None:
    """Load regime-tagged data, run the strategy, and save the trade log."""
    current_ticker = os.environ.get("TICKER", ticker)
    project_root = Path(__file__).resolve().parents[1]
    data_clean_dir = resolve_data_clean_dir(project_root)
    input_path = data_clean_dir / f"{current_ticker}_regimes.csv"
    output_path = data_clean_dir / f"{current_ticker}_volatility_managed_tsmom_trades.csv"

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "ma_50",
        "ma_100",
        "ma_200",
        "atr_14",
        VOL_MANAGED_TSMOM_ENTRY_RETURN_COLUMN,
        VOL_MANAGED_TSMOM_EXIT_RETURN_COLUMN,
        VOL_MANAGED_TSMOM_VOL_COLUMN,
        "avg_volume_20",
        "regime",
    ]
    df = load_regime_data(input_path, required_columns)
    df.attrs["ticker"] = current_ticker

    trades_df = run_strategy("volatility_managed_tsmom", df, ticker=current_ticker)
    trades_df.to_csv(output_path, index=False)

    print(f"Total number of trades: {len(trades_df)}")
    print("\nFirst 10 trades:")
    print(trades_df.head(10))


if __name__ == "__main__":
    main()
