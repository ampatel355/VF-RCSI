"""Calculate trade-level metrics for the Breakout + Volume + Momentum strategy."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from breakout_volume_momentum_agent import main as create_trades
    from strategy_metrics_common import create_and_save_metrics, resolve_data_clean_dir
except ModuleNotFoundError:
    from Code.breakout_volume_momentum_agent import main as create_trades
    from Code.strategy_metrics_common import create_and_save_metrics, resolve_data_clean_dir


ticker = os.environ.get("TICKER", "SPY")


def main() -> None:
    """Create and save the one-row metrics table for this strategy."""
    project_root = Path(__file__).resolve().parents[1]
    data_clean_dir = resolve_data_clean_dir(project_root)
    input_path = data_clean_dir / f"{ticker}_breakout_volume_momentum_trades.csv"
    output_path = data_clean_dir / f"{ticker}_breakout_volume_momentum_metrics.csv"
    if not input_path.exists():
        create_trades()

    metrics_df = create_and_save_metrics(input_path=input_path, output_path=output_path)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
