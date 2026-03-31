"""Shared trade-metric helpers used by the per-strategy metric scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from research_metrics import calculate_trade_level_return_ratio
except ModuleNotFoundError:
    from Code.research_metrics import calculate_trade_level_return_ratio


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


def load_trade_data(input_path: Path) -> pd.DataFrame:
    """Load one trade log and normalize its numeric columns."""
    df = pd.read_csv(input_path)
    for column in ["entry_date", "exit_date"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    df["return"] = pd.to_numeric(df["return"], errors="coerce")
    if "holding_period_days" in df.columns:
        df["holding_period_days"] = pd.to_numeric(df["holding_period_days"], errors="coerce")
    return df.dropna(subset=["return"]).reset_index(drop=True)


def safe_mean(series: pd.Series) -> float:
    """Return the mean of a series, or 0.0 when the series is empty."""
    if series.empty:
        return 0.0
    return float(series.mean())


def build_metrics_row(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize one trade log into the metric table used by the UI and pipeline."""
    winning_returns = trades_df.loc[trades_df["return"] > 0, "return"]
    losing_returns = trades_df.loc[trades_df["return"] < 0, "return"]
    total_trades = int(len(trades_df))
    average_return = safe_mean(trades_df["return"])
    median_return = float(trades_df["return"].median()) if total_trades > 0 else 0.0
    win_rate = float((trades_df["return"] > 0).mean()) if total_trades > 0 else 0.0
    std_return = float(trades_df["return"].std(ddof=0)) if total_trades > 1 else 0.0
    trade_level_return_ratio = calculate_trade_level_return_ratio(
        trades_df["return"].to_numpy(dtype=float)
    )
    average_win = safe_mean(winning_returns)
    average_loss = abs(safe_mean(losing_returns))
    expected_value = (win_rate * average_win) - ((1 - win_rate) * average_loss)

    average_holding_period_days = 0.0
    if "holding_period_days" in trades_df.columns and total_trades > 0:
        average_holding_period_days = safe_mean(
            pd.to_numeric(trades_df["holding_period_days"], errors="coerce").dropna()
        )

    stop_loss_exit_rate = 0.0
    take_profit_exit_rate = 0.0
    if "exit_reason" in trades_df.columns and total_trades > 0:
        exit_reason_series = trades_df["exit_reason"].astype(str).str.lower()
        stop_loss_exit_rate = float(exit_reason_series.str.contains("stop_loss").mean())
        take_profit_exit_rate = float(exit_reason_series.str.contains("take_profit").mean())

    return pd.DataFrame(
        [
            {
                "total_trades": total_trades,
                "average_return": average_return,
                "median_return": median_return,
                "win_rate": win_rate,
                "std_return": std_return,
                "trade_level_return_ratio": trade_level_return_ratio,
                "average_win": average_win,
                "average_loss": average_loss,
                "expected_value": expected_value,
                "average_holding_period_days": average_holding_period_days,
                "stop_loss_exit_rate": stop_loss_exit_rate,
                "take_profit_exit_rate": take_profit_exit_rate,
            }
        ]
    )


def create_and_save_metrics(
    *,
    input_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Load one trade log, build the metrics row, and save it."""
    trades_df = load_trade_data(input_path)
    metrics_df = build_metrics_row(trades_df)
    metrics_df.to_csv(output_path, index=False)
    return metrics_df
