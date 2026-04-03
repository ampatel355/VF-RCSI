"""Shared trade-metric helpers used by the per-strategy metric scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from artifact_provenance import write_dataframe_artifact
    from research_metrics import calculate_trade_level_return_ratio
    from timeframe_config import infer_months_covered, timeframe_output_suffix
except ModuleNotFoundError:
    from Code.artifact_provenance import write_dataframe_artifact
    from Code.research_metrics import calculate_trade_level_return_ratio
    from Code.timeframe_config import infer_months_covered, timeframe_output_suffix


def resolve_data_clean_dir(project_root: Path) -> Path:
    """Return the project's clean-data folder, preferring the uppercase path."""
    suffix = timeframe_output_suffix()
    lowercase_dir = project_root / f"data_clean{suffix}"
    uppercase_dir = project_root / f"Data_Clean{suffix}"

    if uppercase_dir.exists():
        return uppercase_dir
    if lowercase_dir.exists():
        return lowercase_dir

    uppercase_dir.mkdir(parents=True, exist_ok=True)
    return uppercase_dir


def load_trade_data(input_path: Path) -> pd.DataFrame:
    """Load one trade log and normalize its numeric columns."""
    df = pd.read_csv(input_path)
    for column in ["entry_date", "exit_date"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce", utc=True).dt.tz_convert(None)
    df["return"] = pd.to_numeric(df["return"], errors="coerce")
    if "holding_period_days" in df.columns:
        df["holding_period_days"] = pd.to_numeric(df["holding_period_days"], errors="coerce")
    if "holding_period_hours" in df.columns:
        df["holding_period_hours"] = pd.to_numeric(df["holding_period_hours"], errors="coerce")
    if "holding_bars" in df.columns:
        df["holding_bars"] = pd.to_numeric(df["holding_bars"], errors="coerce")
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

    average_holding_period_hours = 0.0
    if "holding_period_hours" in trades_df.columns and total_trades > 0:
        average_holding_period_hours = safe_mean(
            pd.to_numeric(trades_df["holding_period_hours"], errors="coerce").dropna()
        )

    average_holding_bars = 0.0
    if "holding_bars" in trades_df.columns and total_trades > 0:
        average_holding_bars = safe_mean(
            pd.to_numeric(trades_df["holding_bars"], errors="coerce").dropna()
        )

    months_covered = infer_months_covered(trades_df.get("entry_date", pd.Series(dtype="datetime64[ns]")))
    trades_per_month = float(total_trades / months_covered) if months_covered > 0 else 0.0

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
                "average_holding_period_hours": average_holding_period_hours,
                "average_holding_bars": average_holding_bars,
                "trades_per_month": trades_per_month,
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
    ticker_prefix = output_path.name.split("_", 1)[0].strip().upper()
    write_dataframe_artifact(
        metrics_df,
        output_path,
        producer="create_and_save_metrics",
        current_ticker=ticker_prefix,
        dependencies=[input_path],
        research_grade=True,
        canonical_policy="always",
        parameters={
            "artifact_type": "strategy_metrics",
        },
    )
    return metrics_df
