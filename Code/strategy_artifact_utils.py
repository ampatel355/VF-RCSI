"""Helpers that rebuild missing per-strategy trade logs before analysis runs."""

from __future__ import annotations

import os
from pathlib import Path


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


def _strategy_trade_creators():
    """Import trade-generator entrypoints lazily to avoid circular imports."""
    try:
        from breakout_volume_momentum_agent import main as create_breakout_trades
        from mean_reversion_vol_filter_agent import main as create_mean_reversion_trades
        from momentum_relative_strength_agent import main as create_relative_strength_trades
        from random_agent import main as create_random_trades
        from trend_pullback_agent import main as create_trend_trades
        from volatility_managed_tsmom_agent import main as create_volatility_managed_tsmom_trades
    except ModuleNotFoundError:
        from Code.breakout_volume_momentum_agent import main as create_breakout_trades
        from Code.mean_reversion_vol_filter_agent import main as create_mean_reversion_trades
        from Code.momentum_relative_strength_agent import main as create_relative_strength_trades
        from Code.random_agent import main as create_random_trades
        from Code.trend_pullback_agent import main as create_trend_trades
        from Code.volatility_managed_tsmom_agent import main as create_volatility_managed_tsmom_trades

    return {
        "trend_pullback": create_trend_trades,
        "breakout_volume_momentum": create_breakout_trades,
        "mean_reversion_vol_filter": create_mean_reversion_trades,
        "volatility_managed_tsmom": create_volatility_managed_tsmom_trades,
        "momentum_relative_strength": create_relative_strength_trades,
        "random": create_random_trades,
    }


def ensure_trade_file_exists(current_ticker: str, agent_name: str) -> Path:
    """Rebuild one missing strategy trade file so downstream analysis stays consistent."""
    project_root = Path(__file__).resolve().parents[1]
    data_clean_dir = resolve_data_clean_dir(project_root)
    trade_path = data_clean_dir / f"{current_ticker.upper()}_{agent_name}_trades.csv"

    if trade_path.exists():
        return trade_path

    os.environ["TICKER"] = current_ticker.upper()
    creators = _strategy_trade_creators()
    if agent_name not in creators:
        raise ValueError(f"Unsupported strategy name for trade regeneration: {agent_name}")

    creators[agent_name]()
    if not trade_path.exists():
        raise FileNotFoundError(
            f"Trade file is still missing after regeneration attempt: {trade_path}"
        )

    return trade_path
