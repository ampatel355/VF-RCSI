"""Audit trade frequency and execution activity for each active strategy."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

try:
    from strategy_config import (
        AGENT_ORDER,
        MIN_ACCEPTABLE_TOTAL_TRADES,
        OVERACTIVE_MIN_AVG_HOLDING_BARS,
        OVERACTIVE_TRADES_PER_MONTH,
        UNDERACTIVE_TRADES_PER_MONTH,
    )
    from strategy_metrics_common import load_trade_data, safe_mean
    from timeframe_config import RESEARCH_INTERVAL, RESEARCH_TIMEFRAME_LABEL, infer_months_covered, timeframe_output_suffix
except ModuleNotFoundError:
    from Code.strategy_config import (
        AGENT_ORDER,
        MIN_ACCEPTABLE_TOTAL_TRADES,
        OVERACTIVE_MIN_AVG_HOLDING_BARS,
        OVERACTIVE_TRADES_PER_MONTH,
        UNDERACTIVE_TRADES_PER_MONTH,
    )
    from Code.strategy_metrics_common import load_trade_data, safe_mean
    from Code.timeframe_config import RESEARCH_INTERVAL, RESEARCH_TIMEFRAME_LABEL, infer_months_covered, timeframe_output_suffix


ticker = os.environ.get("TICKER", "SPY").strip().upper()


def resolve_data_clean_dir(project_root: Path) -> Path:
    """Return the clean-data folder, supporting either naming style."""
    suffix = timeframe_output_suffix()
    uppercase_dir = project_root / f"Data_Clean{suffix}"
    lowercase_dir = project_root / f"data_clean{suffix}"

    if uppercase_dir.exists():
        return uppercase_dir
    if lowercase_dir.exists():
        return lowercase_dir

    uppercase_dir.mkdir(parents=True, exist_ok=True)
    return uppercase_dir


def load_signal_summary(path: Path) -> pd.DataFrame:
    """Load one signal-summary CSV when it exists."""
    if not path.exists():
        return pd.DataFrame(
            [
                {
                    "signal_count": 0,
                    "executed_trade_count": 0,
                    "rejected_signal_count": 0,
                    "months_covered": 0.0,
                }
            ]
        )
    return pd.read_csv(path)


def classify_activity(
    *,
    total_trades: int,
    trades_per_month: float,
    average_holding_bars: float,
) -> tuple[str, str]:
    """Classify whether a strategy is too quiet, healthy, or too noisy."""
    if total_trades < MIN_ACCEPTABLE_TOTAL_TRADES or trades_per_month < UNDERACTIVE_TRADES_PER_MONTH:
        return (
            "underactive",
            "Too few trades for strong statistical power. Consider reviewing thresholds or sample length.",
        )
    if (
        trades_per_month > OVERACTIVE_TRADES_PER_MONTH
        or (average_holding_bars > 0 and average_holding_bars < OVERACTIVE_MIN_AVG_HOLDING_BARS)
    ):
        return (
            "overactive",
            "Trade frequency is high relative to holding period. Review whether the strategy is chasing noise.",
        )
    return (
        "acceptable",
        f"Trade activity is in a reasonable range for {RESEARCH_TIMEFRAME_LABEL.lower()} research and null-hypothesis testing.",
    )


def build_activity_row(
    agent_name: str,
    data_clean_dir: Path,
    *,
    current_ticker: str,
) -> dict[str, float | int | str]:
    """Build one trade-activity row for one strategy."""
    trade_path = data_clean_dir / f"{current_ticker}_{agent_name}_trades.csv"
    signal_summary_path = data_clean_dir / f"{current_ticker}_{agent_name}_signal_summary.csv"
    regime_path = data_clean_dir / f"{current_ticker}_regimes.csv"
    trades_df = load_trade_data(trade_path) if trade_path.exists() else pd.DataFrame(columns=["return"])
    signal_df = load_signal_summary(signal_summary_path)
    signal_row = signal_df.iloc[0]
    total_bars = 0
    if regime_path.exists():
        try:
            total_bars = int(len(pd.read_csv(regime_path, usecols=["Date"])))
        except Exception:
            total_bars = 0

    total_trades = int(len(trades_df))
    average_holding_bars = safe_mean(
        pd.to_numeric(trades_df.get("holding_bars", pd.Series(dtype=float)), errors="coerce").dropna()
    )
    average_holding_hours = safe_mean(
        pd.to_numeric(trades_df.get("holding_period_hours", pd.Series(dtype=float)), errors="coerce").dropna()
    )
    average_holding_days = safe_mean(
        pd.to_numeric(trades_df.get("holding_period_days", pd.Series(dtype=float)), errors="coerce").dropna()
    )
    months_covered = float(pd.to_numeric(pd.Series([signal_row.get("months_covered", 0.0)]), errors="coerce").iloc[0])
    if months_covered <= 0 and not trades_df.empty:
        months_covered = infer_months_covered(trades_df["entry_date"])

    trades_per_month = float(total_trades / months_covered) if months_covered > 0 else 0.0
    signal_count = int(pd.to_numeric(pd.Series([signal_row.get("signal_count", total_trades)]), errors="coerce").fillna(total_trades).iloc[0])
    rejected_signal_count = int(pd.to_numeric(pd.Series([signal_row.get("rejected_signal_count", 0)]), errors="coerce").fillna(0).iloc[0])
    signal_to_trade_conversion = float(total_trades / signal_count) if signal_count > 0 else 0.0
    signal_rate_pct = float((signal_count / total_bars) * 100.0) if total_bars > 0 else 0.0
    activity_status, activity_warning = classify_activity(
        total_trades=total_trades,
        trades_per_month=trades_per_month,
        average_holding_bars=average_holding_bars,
    )

    return {
        "ticker": current_ticker,
        "agent": agent_name,
        "timeframe_label": RESEARCH_TIMEFRAME_LABEL,
        "data_interval": RESEARCH_INTERVAL,
        "total_trades": total_trades,
        "signal_count": signal_count,
        "rejected_signal_count": rejected_signal_count,
        "signal_to_trade_conversion": signal_to_trade_conversion,
        "total_bars": total_bars,
        "signal_rate_pct": signal_rate_pct,
        "trades_per_month": trades_per_month,
        "average_holding_bars": average_holding_bars,
        "average_holding_days": average_holding_days,
        "average_holding_hours": average_holding_hours,
        "activity_status": activity_status,
        "activity_warning": activity_warning,
    }


def main() -> None:
    """Build and save the trade-activity validation table for the active ticker."""
    project_root = Path(__file__).resolve().parents[1]
    clean_dir = resolve_data_clean_dir(project_root)
    current_ticker = os.environ.get("TICKER", ticker).strip().upper()
    rows = [
        build_activity_row(agent_name, clean_dir, current_ticker=current_ticker)
        for agent_name in AGENT_ORDER
    ]
    output_df = pd.DataFrame(rows)
    output_path = clean_dir / f"{current_ticker}_trade_activity_validation.csv"
    output_df.to_csv(output_path, index=False)
    print(output_df.to_string(index=False))


if __name__ == "__main__":
    main()
