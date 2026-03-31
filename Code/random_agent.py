"""Run the random baseline strategy for the active ticker."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from single_ticker_agent_common import load_regime_data
    from strategy_simulator import (
        RANDOM_AGENT_REPRODUCIBLE,
        RANDOM_AGENT_SEED,
        resolve_data_clean_dir,
        run_strategy,
    )
except ModuleNotFoundError:
    from Code.single_ticker_agent_common import load_regime_data
    from Code.strategy_simulator import (
        RANDOM_AGENT_REPRODUCIBLE,
        RANDOM_AGENT_SEED,
        resolve_data_clean_dir,
        run_strategy,
    )


ticker = os.environ.get("TICKER", "SPY")


def main() -> None:
    """Load regime-tagged data, run the random baseline, and save the trade log."""
    project_root = Path(__file__).resolve().parents[1]
    data_clean_dir = resolve_data_clean_dir(project_root)
    input_path = data_clean_dir / f"{ticker}_regimes.csv"
    output_path = data_clean_dir / f"{ticker}_random_trades.csv"

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "atr_14",
        "avg_volume_20",
        "regime",
    ]
    df = load_regime_data(input_path, required_columns)
    df.attrs["ticker"] = ticker

    decision_seed = RANDOM_AGENT_SEED if RANDOM_AGENT_REPRODUCIBLE else None
    trades_df = run_strategy("random", df, random_decision_seed=decision_seed, ticker=ticker)
    trades_df.to_csv(output_path, index=False)

    print(f"Total number of trades: {len(trades_df)}")
    print(f"Random agent reproducible mode: {RANDOM_AGENT_REPRODUCIBLE}")
    print(f"Random agent seed used: {RANDOM_AGENT_SEED if RANDOM_AGENT_REPRODUCIBLE else 'None'}")
    print("\nFirst 10 trades:")
    print(trades_df.head(10))


if __name__ == "__main__":
    main()
