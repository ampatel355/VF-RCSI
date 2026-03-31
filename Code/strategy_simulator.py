"""Shared in-memory strategy runners used by the single-ticker pipeline."""

from __future__ import annotations

import os
from pathlib import Path
import random

import pandas as pd

try:
    from execution_model import (
        STARTING_CAPITAL,
        TRADE_LOG_COLUMNS,
        build_execution_rng,
        close_position_from_signal,
        open_position_from_signal,
    )
    from strategy_config import (
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOW_COLUMN,
        MOMENTUM_RETURN_COLUMN,
    )
except ModuleNotFoundError:
    from Code.execution_model import (
        STARTING_CAPITAL,
        TRADE_LOG_COLUMNS,
        build_execution_rng,
        close_position_from_signal,
        open_position_from_signal,
    )
    from Code.strategy_config import (
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOW_COLUMN,
        MOMENTUM_RETURN_COLUMN,
    )


RANDOM_AGENT_REPRODUCIBLE = os.environ.get("RANDOM_AGENT_REPRODUCIBLE", "1") == "1"
RANDOM_AGENT_SEED = int(os.environ.get("RANDOM_AGENT_SEED", "42"))
ENTRY_PROBABILITY = 0.05
EXIT_PROBABILITY = 0.05


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


def _prepare_market_df(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    """Validate and normalize a regime-tagged market DataFrame."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(
            "Market data is missing the columns required for strategy execution: "
            + ", ".join(missing_columns)
        )

    prepared_df = df.copy()
    prepared_df["Date"] = pd.to_datetime(prepared_df["Date"], errors="coerce")
    numeric_columns = [column for column in required_columns if column not in {"Date", "regime"}]
    for column in numeric_columns:
        prepared_df[column] = pd.to_numeric(prepared_df[column], errors="coerce")
    prepared_df["regime"] = prepared_df["regime"].astype(str)

    prepared_df = prepared_df.dropna(subset=required_columns).sort_values("Date").reset_index(drop=True)
    return prepared_df


def _build_random_decision_generator(seed_override: int | None = None):
    """Create the random decision generator used by the random baseline."""
    if seed_override is not None:
        return random.Random(seed_override)
    if RANDOM_AGENT_REPRODUCIBLE:
        return random.Random(RANDOM_AGENT_SEED)
    return random.SystemRandom()


def _finalize_trades(trades: list[dict[str, object]]) -> pd.DataFrame:
    """Convert collected trades into a consistently ordered DataFrame."""
    return pd.DataFrame(trades, columns=TRADE_LOG_COLUMNS)


def run_trend_strategy(market_df: pd.DataFrame) -> pd.DataFrame:
    """Run the trend strategy on an in-memory market DataFrame."""
    df = _prepare_market_df(market_df, ["Date", "Open", "Close", "ma_50", "avg_volume_20", "regime"])
    trades: list[dict[str, object]] = []
    capital = STARTING_CAPITAL
    execution_rng = build_execution_rng("trend")
    in_position = False
    open_position = None

    for current_index in range(len(df) - 1):
        row = df.iloc[current_index]
        next_row = df.iloc[current_index + 1]

        if current_index == 0:
            entry_signal = row.Close > row.ma_50
            exit_signal = False
        else:
            previous_row = df.iloc[current_index - 1]
            entry_signal = previous_row.Close <= previous_row.ma_50 and row.Close > row.ma_50
            exit_signal = previous_row.Close >= previous_row.ma_50 and row.Close < row.ma_50

        if not in_position and entry_signal:
            candidate_position = open_position_from_signal(
                signal_row=row,
                next_row=next_row,
                capital_before=capital,
                regime_at_entry=str(row.regime),
                entry_index=current_index + 1,
                rng=execution_rng,
            )
            if candidate_position is not None:
                in_position = True
                open_position = candidate_position
        elif in_position and exit_signal:
            trade_record = close_position_from_signal(
                position=open_position,
                next_row=next_row,
                exit_index=current_index + 1,
                rng=execution_rng,
            )
            trades.append(trade_record)
            capital = float(trade_record["capital_after"])
            in_position = False
            open_position = None

    return _finalize_trades(trades)


def run_mean_reversion_strategy(market_df: pd.DataFrame) -> pd.DataFrame:
    """Run the mean-reversion strategy on an in-memory market DataFrame."""
    df = _prepare_market_df(
        market_df,
        [
            "Date",
            "Open",
            "Close",
            "ma_20",
            "price_std_20",
            "zscore_20",
            "avg_volume_20",
            "regime",
        ],
    )
    trades: list[dict[str, object]] = []
    capital = STARTING_CAPITAL
    execution_rng = build_execution_rng("mean_reversion")
    in_position = False
    open_position = None

    for current_index in range(len(df) - 1):
        row = df.iloc[current_index]
        next_row = df.iloc[current_index + 1]

        if current_index == 0:
            entry_signal = row.zscore_20 <= -2.0
            exit_signal = False
        else:
            previous_row = df.iloc[current_index - 1]
            entry_signal = previous_row.zscore_20 > -2.0 and row.zscore_20 <= -2.0
            exit_signal = previous_row.Close < previous_row.ma_20 and row.Close >= row.ma_20

        if not in_position and entry_signal:
            candidate_position = open_position_from_signal(
                signal_row=row,
                next_row=next_row,
                capital_before=capital,
                regime_at_entry=str(row.regime),
                entry_index=current_index + 1,
                rng=execution_rng,
            )
            if candidate_position is not None:
                in_position = True
                open_position = candidate_position
        elif in_position and exit_signal:
            trade_record = close_position_from_signal(
                position=open_position,
                next_row=next_row,
                exit_index=current_index + 1,
                rng=execution_rng,
            )
            trades.append(trade_record)
            capital = float(trade_record["capital_after"])
            in_position = False
            open_position = None

    return _finalize_trades(trades)


def run_random_strategy(market_df: pd.DataFrame, decision_seed: int | None = None) -> pd.DataFrame:
    """Run the random control strategy on an in-memory market DataFrame."""
    df = _prepare_market_df(market_df, ["Date", "Open", "Close", "avg_volume_20", "regime"])
    trades: list[dict[str, object]] = []
    capital = STARTING_CAPITAL
    in_position = False
    open_position = None
    execution_rng = build_execution_rng("random")
    decision_rng = _build_random_decision_generator(seed_override=decision_seed)

    for current_index in range(len(df) - 1):
        row = df.iloc[current_index]
        next_row = df.iloc[current_index + 1]

        if not in_position and decision_rng.random() < ENTRY_PROBABILITY:
            candidate_position = open_position_from_signal(
                signal_row=row,
                next_row=next_row,
                capital_before=capital,
                regime_at_entry=str(row.regime),
                entry_index=current_index + 1,
                rng=execution_rng,
            )
            if candidate_position is not None:
                in_position = True
                open_position = candidate_position
        elif in_position and decision_rng.random() < EXIT_PROBABILITY:
            trade_record = close_position_from_signal(
                position=open_position,
                next_row=next_row,
                exit_index=current_index + 1,
                rng=execution_rng,
            )
            trades.append(trade_record)
            capital = float(trade_record["capital_after"])
            in_position = False
            open_position = None

    return _finalize_trades(trades)


def run_momentum_strategy(market_df: pd.DataFrame) -> pd.DataFrame:
    """Run the time-series momentum strategy on an in-memory market DataFrame."""
    df = _prepare_market_df(
        market_df,
        ["Date", "Open", "Close", MOMENTUM_RETURN_COLUMN, "avg_volume_20", "regime"],
    )
    trades: list[dict[str, object]] = []
    capital = STARTING_CAPITAL
    execution_rng = build_execution_rng("momentum")
    in_position = False
    open_position = None

    for current_index in range(len(df) - 1):
        row = df.iloc[current_index]
        next_row = df.iloc[current_index + 1]

        if current_index == 0:
            entry_signal = row[MOMENTUM_RETURN_COLUMN] > 0
            exit_signal = False
        else:
            previous_row = df.iloc[current_index - 1]
            entry_signal = previous_row[MOMENTUM_RETURN_COLUMN] <= 0 and row[MOMENTUM_RETURN_COLUMN] > 0
            exit_signal = previous_row[MOMENTUM_RETURN_COLUMN] > 0 and row[MOMENTUM_RETURN_COLUMN] <= 0

        if not in_position and entry_signal:
            candidate_position = open_position_from_signal(
                signal_row=row,
                next_row=next_row,
                capital_before=capital,
                regime_at_entry=str(row.regime),
                entry_index=current_index + 1,
                rng=execution_rng,
            )
            if candidate_position is not None:
                in_position = True
                open_position = candidate_position
        elif in_position and exit_signal:
            trade_record = close_position_from_signal(
                position=open_position,
                next_row=next_row,
                exit_index=current_index + 1,
                rng=execution_rng,
            )
            trades.append(trade_record)
            capital = float(trade_record["capital_after"])
            in_position = False
            open_position = None

    return _finalize_trades(trades)


def run_breakout_strategy(market_df: pd.DataFrame) -> pd.DataFrame:
    """Run the volatility-breakout strategy on an in-memory market DataFrame."""
    df = _prepare_market_df(
        market_df,
        [
            "Date",
            "Open",
            "Close",
            BREAKOUT_HIGH_COLUMN,
            BREAKOUT_LOW_COLUMN,
            "avg_volume_20",
            "regime",
        ],
    )
    trades: list[dict[str, object]] = []
    capital = STARTING_CAPITAL
    execution_rng = build_execution_rng("breakout")
    in_position = False
    open_position = None

    for current_index in range(len(df) - 1):
        row = df.iloc[current_index]
        next_row = df.iloc[current_index + 1]

        if current_index == 0:
            entry_signal = row.Close > row[BREAKOUT_HIGH_COLUMN]
            exit_signal = False
        else:
            previous_row = df.iloc[current_index - 1]
            entry_signal = (
                previous_row.Close <= previous_row[BREAKOUT_HIGH_COLUMN]
                and row.Close > row[BREAKOUT_HIGH_COLUMN]
            )
            exit_signal = (
                previous_row.Close >= previous_row[BREAKOUT_LOW_COLUMN]
                and row.Close < row[BREAKOUT_LOW_COLUMN]
            )

        if not in_position and entry_signal:
            candidate_position = open_position_from_signal(
                signal_row=row,
                next_row=next_row,
                capital_before=capital,
                regime_at_entry=str(row.regime),
                entry_index=current_index + 1,
                rng=execution_rng,
            )
            if candidate_position is not None:
                in_position = True
                open_position = candidate_position
        elif in_position and exit_signal:
            trade_record = close_position_from_signal(
                position=open_position,
                next_row=next_row,
                exit_index=current_index + 1,
                rng=execution_rng,
            )
            trades.append(trade_record)
            capital = float(trade_record["capital_after"])
            in_position = False
            open_position = None

    return _finalize_trades(trades)


def run_strategy(
    agent_name: str,
    market_df: pd.DataFrame,
    random_decision_seed: int | None = None,
) -> pd.DataFrame:
    """Dispatch one shared in-memory runner by strategy name."""
    if agent_name == "trend":
        return run_trend_strategy(market_df)
    if agent_name == "mean_reversion":
        return run_mean_reversion_strategy(market_df)
    if agent_name == "random":
        return run_random_strategy(market_df, decision_seed=random_decision_seed)
    if agent_name == "momentum":
        return run_momentum_strategy(market_df)
    if agent_name == "breakout":
        return run_breakout_strategy(market_df)

    raise ValueError(f"Unsupported strategy name: {agent_name}")
