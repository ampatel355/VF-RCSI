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
        OpenPosition,
        build_execution_rng,
        close_position_from_signal,
        open_position_from_signal,
    )
    from strategy_config import (
        BREAKOUT_CLOSE_BUFFER,
        BREAKOUT_FAILED_BREAKOUT_BUFFER,
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOW_COLUMN,
        BREAKOUT_MOMENTUM_RETURN_COLUMN,
        BREAKOUT_MOMENTUM_THRESHOLD,
        BREAKOUT_NO_VOLUME_RANGE_EXPANSION_RATIO,
        BREAKOUT_REWARD_TO_RISK,
        BREAKOUT_STOP_ATR_BUFFER,
        BREAKOUT_TIME_STOP_BARS,
        BREAKOUT_VOLUME_MULTIPLIER,
        MEAN_REVERSION_ATR_RATIO_MAX,
        MEAN_REVERSION_BB_WIDTH_RATIO_MAX,
        MEAN_REVERSION_LOWER_BAND_BUFFER,
        MEAN_REVERSION_MA_ALIGNMENT_THRESHOLD,
        MEAN_REVERSION_RSI_ENTRY_MAX,
        MEAN_REVERSION_RSI_EXIT_MIN,
        MEAN_REVERSION_STOP_ATR_BUFFER,
        MEAN_REVERSION_STOP_LOW_COLUMN,
        MEAN_REVERSION_TREND_DISTANCE_THRESHOLD,
        RANDOM_ENTRY_PROBABILITY,
        RANDOM_HOLDING_PERIOD_MAX_BARS,
        RANDOM_HOLDING_PERIOD_MIN_BARS,
        TREND_PULLBACK_PULLBACK_TOLERANCE,
        TREND_PULLBACK_RSI_MAX,
        TREND_PULLBACK_RSI_MIN,
        TREND_PULLBACK_STOP_ATR_BUFFER,
        TREND_PULLBACK_STOP_LOW_COLUMN,
        TREND_PULLBACK_TARGET_HIGH_COLUMN,
        VOL_MANAGED_TSMOM_ENTRY_RETURN_COLUMN,
        VOL_MANAGED_TSMOM_ENTRY_THRESHOLD,
        VOL_MANAGED_TSMOM_EXIT_RETURN_COLUMN,
        VOL_MANAGED_TSMOM_EXIT_THRESHOLD,
        VOL_MANAGED_TSMOM_MAX_CAPITAL_FRACTION,
        VOL_MANAGED_TSMOM_MIN_CAPITAL_FRACTION,
        VOL_MANAGED_TSMOM_STOP_ATR_MULTIPLIER,
        VOL_MANAGED_TSMOM_TARGET_ANNUALIZED_VOL,
        VOL_MANAGED_TSMOM_VOL_COLUMN,
    )
except ModuleNotFoundError:
    from Code.execution_model import (
        STARTING_CAPITAL,
        TRADE_LOG_COLUMNS,
        OpenPosition,
        build_execution_rng,
        close_position_from_signal,
        open_position_from_signal,
    )
    from Code.strategy_config import (
        BREAKOUT_CLOSE_BUFFER,
        BREAKOUT_FAILED_BREAKOUT_BUFFER,
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOW_COLUMN,
        BREAKOUT_MOMENTUM_RETURN_COLUMN,
        BREAKOUT_MOMENTUM_THRESHOLD,
        BREAKOUT_NO_VOLUME_RANGE_EXPANSION_RATIO,
        BREAKOUT_REWARD_TO_RISK,
        BREAKOUT_STOP_ATR_BUFFER,
        BREAKOUT_TIME_STOP_BARS,
        BREAKOUT_VOLUME_MULTIPLIER,
        MEAN_REVERSION_ATR_RATIO_MAX,
        MEAN_REVERSION_BB_WIDTH_RATIO_MAX,
        MEAN_REVERSION_LOWER_BAND_BUFFER,
        MEAN_REVERSION_MA_ALIGNMENT_THRESHOLD,
        MEAN_REVERSION_RSI_ENTRY_MAX,
        MEAN_REVERSION_RSI_EXIT_MIN,
        MEAN_REVERSION_STOP_ATR_BUFFER,
        MEAN_REVERSION_STOP_LOW_COLUMN,
        MEAN_REVERSION_TREND_DISTANCE_THRESHOLD,
        RANDOM_ENTRY_PROBABILITY,
        RANDOM_HOLDING_PERIOD_MAX_BARS,
        RANDOM_HOLDING_PERIOD_MIN_BARS,
        TREND_PULLBACK_PULLBACK_TOLERANCE,
        TREND_PULLBACK_RSI_MAX,
        TREND_PULLBACK_RSI_MIN,
        TREND_PULLBACK_STOP_ATR_BUFFER,
        TREND_PULLBACK_STOP_LOW_COLUMN,
        TREND_PULLBACK_TARGET_HIGH_COLUMN,
        VOL_MANAGED_TSMOM_ENTRY_RETURN_COLUMN,
        VOL_MANAGED_TSMOM_ENTRY_THRESHOLD,
        VOL_MANAGED_TSMOM_EXIT_RETURN_COLUMN,
        VOL_MANAGED_TSMOM_EXIT_THRESHOLD,
        VOL_MANAGED_TSMOM_MAX_CAPITAL_FRACTION,
        VOL_MANAGED_TSMOM_MIN_CAPITAL_FRACTION,
        VOL_MANAGED_TSMOM_STOP_ATR_MULTIPLIER,
        VOL_MANAGED_TSMOM_TARGET_ANNUALIZED_VOL,
        VOL_MANAGED_TSMOM_VOL_COLUMN,
    )


RANDOM_AGENT_REPRODUCIBLE = os.environ.get("RANDOM_AGENT_REPRODUCIBLE", "1") == "1"
RANDOM_AGENT_SEED = int(os.environ.get("RANDOM_AGENT_SEED", "42"))


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

    return (
        prepared_df.dropna(subset=required_columns)
        .sort_values("Date")
        .reset_index(drop=True)
    )


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


def _resolve_strategy_ticker(market_df: pd.DataFrame, ticker: str | None = None) -> str | None:
    """Prefer an explicit ticker, then fall back to DataFrame metadata when available."""
    if ticker:
        return str(ticker).strip().upper()

    ticker_attr = market_df.attrs.get("ticker")
    if ticker_attr:
        return str(ticker_attr).strip().upper()

    return None


def _first_daily_exit_reason(
    row: pd.Series,
    stop_loss_used: float | None,
    take_profit_used: float | None,
) -> str | None:
    """Check stop/target breaches using only the completed current daily bar.

    Practical simplification for daily data:
    - Signals are generated after the close.
    - Entries and exits happen at the next session's open.
    - If both stop and target are touched inside the same daily bar, we do not
      know which came first intraday, so we choose the stop-loss path
      conservatively.
    """
    stop_hit = (
        stop_loss_used is not None
        and pd.notna(row["Low"])
        and float(row["Low"]) <= float(stop_loss_used)
    )
    target_hit = (
        take_profit_used is not None
        and pd.notna(row["High"])
        and float(row["High"]) >= float(take_profit_used)
    )

    if stop_hit and target_hit:
        return "stop_loss_same_bar_priority"
    if stop_hit:
        return "stop_loss"
    if target_hit:
        return "take_profit"
    return None


def _entry_is_still_valid(
    next_open: float,
    stop_loss_used: float | None,
    take_profit_used: float | None,
) -> bool:
    """Skip trades that gap through the planned stop or target before entry."""
    if not pd.notna(next_open) or float(next_open) <= 0:
        return False

    if stop_loss_used is not None and float(next_open) <= float(stop_loss_used):
        return False
    if take_profit_used is not None and float(next_open) >= float(take_profit_used):
        return False
    return True


def _safe_stop(value: float | int | None) -> float | None:
    """Convert optional numeric thresholds into clean floats."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def _is_month_end_rebalance(current_date: pd.Timestamp, next_date: pd.Timestamp) -> bool:
    """Return whether the signal day is the last trading day of its calendar month."""
    return (
        current_date.month != next_date.month
        or current_date.year != next_date.year
    )


def _volatility_managed_position_fraction(realized_vol_annualized: float | int | None) -> float | None:
    """Convert realized volatility into a capital-allocation fraction.

    Practical simplification for this project's discrete trade framework:
    instead of continuously rebalancing leverage every day, we size the trade
    at entry using the volatility estimate available on the signal day. That
    keeps the strategy compatible with the existing non-overlapping trade logs
    used by the metrics, Monte Carlo, and robustness pipeline.
    """
    if realized_vol_annualized is None or pd.isna(realized_vol_annualized):
        return None

    realized_vol_annualized = float(realized_vol_annualized)
    if realized_vol_annualized <= 0:
        return None

    raw_fraction = VOL_MANAGED_TSMOM_TARGET_ANNUALIZED_VOL / realized_vol_annualized
    return float(
        min(
            VOL_MANAGED_TSMOM_MAX_CAPITAL_FRACTION,
            max(VOL_MANAGED_TSMOM_MIN_CAPITAL_FRACTION, raw_fraction),
        )
    )


def run_trend_pullback_strategy(market_df: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    """Buy pullbacks inside an established uptrend and exit on target, stop, or trend failure."""
    df = _prepare_market_df(
        market_df,
        [
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
        ],
    )
    resolved_ticker = _resolve_strategy_ticker(market_df, ticker)
    trades: list[dict[str, object]] = []
    capital = STARTING_CAPITAL
    execution_rng = build_execution_rng("trend_pullback")
    open_position: OpenPosition | None = None

    for current_index in range(1, len(df) - 1):
        previous_row = df.iloc[current_index - 1]
        row = df.iloc[current_index]
        next_row = df.iloc[current_index + 1]

        if open_position is None:
            in_uptrend = row.ma_50 > row.ma_200 and row.Close > row.ma_200
            pullback_to_fast_average = row.Low <= row.ma_20 * (1.0 + TREND_PULLBACK_PULLBACK_TOLERANCE)
            pullback_to_intermediate_average = row.Low <= row.ma_50 * (
                1.0 + TREND_PULLBACK_PULLBACK_TOLERANCE
            )
            touched_pullback_zone = pullback_to_fast_average or pullback_to_intermediate_average
            rsi_filter = TREND_PULLBACK_RSI_MIN <= row.rsi_14 <= TREND_PULLBACK_RSI_MAX
            bounce_confirmation = (
                touched_pullback_zone
                and row.Close > previous_row.Close
                and (row.Close >= row.ma_20 or row.Close >= row.ma_50)
            )
            entry_signal = in_uptrend and rsi_filter and bounce_confirmation

            if not entry_signal:
                continue

            stop_loss_used = _safe_stop(
                min(row[TREND_PULLBACK_STOP_LOW_COLUMN], row.Low) - (row.atr_14 * TREND_PULLBACK_STOP_ATR_BUFFER)
            )
            take_profit_used = _safe_stop(row[TREND_PULLBACK_TARGET_HIGH_COLUMN])
            if (
                take_profit_used is None
                or take_profit_used <= row.Close
            ) and stop_loss_used is not None:
                take_profit_used = float(row.Close + (2.0 * max(row.Close - stop_loss_used, 0.0)))

            if not _entry_is_still_valid(float(next_row.Open), stop_loss_used, take_profit_used):
                continue

            candidate_position = open_position_from_signal(
                signal_row=row,
                next_row=next_row,
                capital_before=capital,
                regime_at_entry=str(row.regime),
                entry_index=current_index + 1,
                rng=execution_rng,
                strategy_name="trend_pullback",
                ticker=resolved_ticker,
                stop_loss_used=stop_loss_used,
                take_profit_used=take_profit_used,
            )
            if candidate_position is not None:
                open_position = candidate_position
        else:
            exit_reason = _first_daily_exit_reason(
                row,
                open_position.stop_loss_used,
                open_position.take_profit_used,
            )
            if exit_reason is None and (row.Close < row.ma_50 or row.ma_50 <= row.ma_200):
                exit_reason = "trend_filter_failed"

            if exit_reason is None:
                continue

            trade_record = close_position_from_signal(
                position=open_position,
                next_row=next_row,
                exit_index=current_index + 1,
                rng=execution_rng,
                ticker=resolved_ticker,
                exit_reason=exit_reason,
            )
            trades.append(trade_record)
            capital = float(trade_record["capital_after"])
            open_position = None

    return _finalize_trades(trades)


def run_breakout_volume_momentum_strategy(
    market_df: pd.DataFrame,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Trade strong breakouts confirmed by broad participation and positive momentum."""
    df = _prepare_market_df(
        market_df,
        [
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
        ],
    )
    resolved_ticker = _resolve_strategy_ticker(market_df, ticker)
    trades: list[dict[str, object]] = []
    capital = STARTING_CAPITAL
    execution_rng = build_execution_rng("breakout_volume_momentum")
    open_position: OpenPosition | None = None

    for current_index in range(1, len(df) - 1):
        previous_row = df.iloc[current_index - 1]
        row = df.iloc[current_index]
        next_row = df.iloc[current_index + 1]

        if open_position is None:
            breakout_level = float(row[BREAKOUT_HIGH_COLUMN])
            breakout_signal = (
                row.Close > breakout_level
                and previous_row.Close <= previous_row[BREAKOUT_HIGH_COLUMN]
            )
            # Daily-data simplification:
            # A strong breakout sometimes pokes above resistance intraday and
            # closes only slightly below it. Allowing a close within a small
            # buffer keeps the strategy realistic without requiring intraday data.
            near_breakout_close = (
                row.High >= breakout_level
                and row.Close >= breakout_level * (1.0 - BREAKOUT_CLOSE_BUFFER)
                and row.Close >= previous_row.Close
            )
            has_reliable_volume = pd.notna(row.avg_volume_20) and float(row.avg_volume_20) > 0
            if has_reliable_volume:
                participation_filter = row.volume_ratio_20 >= BREAKOUT_VOLUME_MULTIPLIER
            else:
                # Some markets in Yahoo data, especially forex, have no real
                # exchange volume field. In those cases use daily range
                # expansion relative to ATR as the participation proxy.
                participation_filter = (
                    float(row.High - row.Low) >= (float(row.atr_14) * BREAKOUT_NO_VOLUME_RANGE_EXPANSION_RATIO)
                )
            momentum_filter = (
                (row.macd_line > row.macd_signal or (row.macd_hist > 0 and row[BREAKOUT_MOMENTUM_RETURN_COLUMN] > 0))
                and row[BREAKOUT_MOMENTUM_RETURN_COLUMN] >= BREAKOUT_MOMENTUM_THRESHOLD
            )
            entry_signal = (breakout_signal or near_breakout_close) and participation_filter and momentum_filter

            if not entry_signal:
                continue

            stop_anchor = min(row.Low, row[BREAKOUT_LOW_COLUMN])
            stop_loss_used = _safe_stop(stop_anchor - (BREAKOUT_STOP_ATR_BUFFER * row.atr_14))
            if stop_loss_used is None:
                continue

            signal_risk = max(row.Close - stop_loss_used, 0.0)
            if signal_risk <= 0:
                continue

            take_profit_used = float(row.Close + (BREAKOUT_REWARD_TO_RISK * signal_risk))
            if not _entry_is_still_valid(float(next_row.Open), stop_loss_used, take_profit_used):
                continue

            candidate_position = open_position_from_signal(
                signal_row=row,
                next_row=next_row,
                capital_before=capital,
                regime_at_entry=str(row.regime),
                entry_index=current_index + 1,
                rng=execution_rng,
                strategy_name="breakout_volume_momentum",
                ticker=resolved_ticker,
                stop_loss_used=stop_loss_used,
                take_profit_used=take_profit_used,
            )
            if candidate_position is not None:
                open_position = candidate_position
        else:
            exit_reason = _first_daily_exit_reason(
                row,
                open_position.stop_loss_used,
                open_position.take_profit_used,
            )
            holding_bars = current_index - open_position.entry_index
            failed_breakout = (
                row.Close < row[BREAKOUT_HIGH_COLUMN] * (1.0 - BREAKOUT_FAILED_BREAKOUT_BUFFER)
                and row.macd_hist < 0
            )
            timed_exit = holding_bars >= BREAKOUT_TIME_STOP_BARS

            if exit_reason is None and failed_breakout:
                exit_reason = "failed_breakout"
            if exit_reason is None and timed_exit:
                exit_reason = "time_stop"

            if exit_reason is None:
                continue

            trade_record = close_position_from_signal(
                position=open_position,
                next_row=next_row,
                exit_index=current_index + 1,
                rng=execution_rng,
                ticker=resolved_ticker,
                exit_reason=exit_reason,
            )
            trades.append(trade_record)
            capital = float(trade_record["capital_after"])
            open_position = None

    return _finalize_trades(trades)


def run_mean_reversion_vol_filter_strategy(
    market_df: pd.DataFrame,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Fade oversold moves only when volatility and trend conditions stay tame."""
    df = _prepare_market_df(
        market_df,
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "bollinger_mid",
            "bollinger_lower",
            "rsi_14",
            "atr_14",
            "atr_percent_ratio_60",
            "bollinger_width_ratio_60",
            "distance_from_ma_50",
            "ma_50_over_ma_200",
            MEAN_REVERSION_STOP_LOW_COLUMN,
            "avg_volume_20",
            "regime",
        ],
    )
    resolved_ticker = _resolve_strategy_ticker(market_df, ticker)
    trades: list[dict[str, object]] = []
    capital = STARTING_CAPITAL
    execution_rng = build_execution_rng("mean_reversion_vol_filter")
    open_position: OpenPosition | None = None

    for current_index in range(1, len(df) - 1):
        row = df.iloc[current_index]
        next_row = df.iloc[current_index + 1]

        if open_position is None:
            low_volatility_environment = (
                row.atr_percent_ratio_60 <= MEAN_REVERSION_ATR_RATIO_MAX
                and row.bollinger_width_ratio_60 <= MEAN_REVERSION_BB_WIDTH_RATIO_MAX
            )
            weak_trend_environment = (
                abs(row.distance_from_ma_50) <= MEAN_REVERSION_TREND_DISTANCE_THRESHOLD
                and abs(row.ma_50_over_ma_200) <= MEAN_REVERSION_MA_ALIGNMENT_THRESHOLD
            )
            # Daily-data simplification:
            # Requiring the closing price to finish below the lower Bollinger
            # Band was too strict. Using the intraday low or a close very near
            # the band better approximates a washed-out move on end-of-day data.
            oversold_entry = (
                (row.Low <= row.bollinger_lower * (1.0 + MEAN_REVERSION_LOWER_BAND_BUFFER))
                or (row.Close <= row.bollinger_lower * (1.0 + MEAN_REVERSION_LOWER_BAND_BUFFER))
            ) and row.rsi_14 < MEAN_REVERSION_RSI_ENTRY_MAX
            entry_signal = low_volatility_environment and weak_trend_environment and oversold_entry

            if not entry_signal:
                continue

            stop_loss_used = _safe_stop(
                min(row.Low, row[MEAN_REVERSION_STOP_LOW_COLUMN]) - (row.atr_14 * MEAN_REVERSION_STOP_ATR_BUFFER)
            )
            take_profit_used = _safe_stop(row.bollinger_mid)
            if not _entry_is_still_valid(float(next_row.Open), stop_loss_used, take_profit_used):
                continue

            candidate_position = open_position_from_signal(
                signal_row=row,
                next_row=next_row,
                capital_before=capital,
                regime_at_entry=str(row.regime),
                entry_index=current_index + 1,
                rng=execution_rng,
                strategy_name="mean_reversion_vol_filter",
                ticker=resolved_ticker,
                stop_loss_used=stop_loss_used,
                take_profit_used=take_profit_used,
            )
            if candidate_position is not None:
                open_position = candidate_position
        else:
            exit_reason = _first_daily_exit_reason(
                row,
                open_position.stop_loss_used,
                open_position.take_profit_used,
            )
            if exit_reason is None and row.Close >= row.bollinger_mid:
                exit_reason = "mean_reversion_complete"
            if exit_reason is None and row.rsi_14 >= MEAN_REVERSION_RSI_EXIT_MIN:
                exit_reason = "rsi_recovery"

            if exit_reason is None:
                continue

            trade_record = close_position_from_signal(
                position=open_position,
                next_row=next_row,
                exit_index=current_index + 1,
                rng=execution_rng,
                ticker=resolved_ticker,
                exit_reason=exit_reason,
            )
            trades.append(trade_record)
            capital = float(trade_record["capital_after"])
            open_position = None

    return _finalize_trades(trades)


def run_volatility_managed_tsmom_strategy(
    market_df: pd.DataFrame,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Trade one asset's own trend while scaling entry size by realized volatility."""
    df = _prepare_market_df(
        market_df,
        [
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
        ],
    )
    resolved_ticker = _resolve_strategy_ticker(market_df, ticker)
    trades: list[dict[str, object]] = []
    capital = STARTING_CAPITAL
    execution_rng = build_execution_rng("volatility_managed_tsmom")
    open_position: OpenPosition | None = None
    highest_close_since_entry: float | None = None

    for current_index in range(1, len(df) - 1):
        previous_row = df.iloc[current_index - 1]
        row = df.iloc[current_index]
        next_row = df.iloc[current_index + 1]
        rebalance_today = _is_month_end_rebalance(
            pd.Timestamp(row.Date),
            pd.Timestamp(next_row.Date),
        )

        if open_position is None:
            positive_trend_signal = (
                row[VOL_MANAGED_TSMOM_ENTRY_RETURN_COLUMN] > VOL_MANAGED_TSMOM_ENTRY_THRESHOLD
                and row.Close > row.ma_200
                and row.ma_50 >= row.ma_200
            )
            trend_confirmation = (
                row.Close >= previous_row.Close
                or row[VOL_MANAGED_TSMOM_ENTRY_RETURN_COLUMN]
                >= previous_row[VOL_MANAGED_TSMOM_ENTRY_RETURN_COLUMN]
            )
            position_fraction = _volatility_managed_position_fraction(
                row[VOL_MANAGED_TSMOM_VOL_COLUMN]
            )
            entry_signal = (
                rebalance_today
                and positive_trend_signal
                and trend_confirmation
                and position_fraction is not None
            )

            if not entry_signal:
                continue

            stop_loss_used = _safe_stop(
                row.Close - (VOL_MANAGED_TSMOM_STOP_ATR_MULTIPLIER * row.atr_14)
            )
            if not _entry_is_still_valid(float(next_row.Open), stop_loss_used, None):
                continue

            candidate_position = open_position_from_signal(
                signal_row=row,
                next_row=next_row,
                capital_before=capital,
                regime_at_entry=str(row.regime),
                entry_index=current_index + 1,
                rng=execution_rng,
                strategy_name="volatility_managed_tsmom",
                ticker=resolved_ticker,
                stop_loss_used=stop_loss_used,
                take_profit_used=None,
                capital_fraction_override=position_fraction,
            )
            if candidate_position is not None:
                open_position = candidate_position
                highest_close_since_entry = float(row.Close)
        else:
            highest_close_since_entry = (
                float(row.Close)
                if highest_close_since_entry is None
                else max(highest_close_since_entry, float(row.Close))
            )
            trailing_stop = highest_close_since_entry - (
                VOL_MANAGED_TSMOM_STOP_ATR_MULTIPLIER * float(row.atr_14)
            )
            if open_position.stop_loss_used is None:
                open_position.stop_loss_used = float(trailing_stop)
            else:
                open_position.stop_loss_used = max(
                    float(open_position.stop_loss_used),
                    float(trailing_stop),
                )

            exit_reason = _first_daily_exit_reason(
                row,
                open_position.stop_loss_used,
                open_position.take_profit_used,
            )
            negative_trend_signal = (
                (
                    row[VOL_MANAGED_TSMOM_EXIT_RETURN_COLUMN] <= VOL_MANAGED_TSMOM_EXIT_THRESHOLD
                    and row.Close < row.ma_100
                )
                or row.ma_50 < row.ma_200
            )
            if exit_reason is None and negative_trend_signal:
                exit_reason = "trend_reversal"

            if exit_reason is None:
                continue

            trade_record = close_position_from_signal(
                position=open_position,
                next_row=next_row,
                exit_index=current_index + 1,
                rng=execution_rng,
                ticker=resolved_ticker,
                exit_reason=exit_reason,
            )
            trades.append(trade_record)
            capital = float(trade_record["capital_after"])
            open_position = None
            highest_close_since_entry = None

    return _finalize_trades(trades)


def run_random_strategy(
    market_df: pd.DataFrame,
    decision_seed: int | None = None,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Run a clean random-timing control with fixed risk controls and random holding periods."""
    df = _prepare_market_df(
        market_df,
        ["Date", "Open", "High", "Low", "Close", "atr_14", "avg_volume_20", "regime"],
    )
    resolved_ticker = _resolve_strategy_ticker(market_df, ticker)
    trades: list[dict[str, object]] = []
    capital = STARTING_CAPITAL
    execution_rng = build_execution_rng("random")
    decision_rng = _build_random_decision_generator(seed_override=decision_seed)
    open_position: OpenPosition | None = None
    planned_exit_index: int | None = None

    for current_index in range(1, len(df) - 1):
        row = df.iloc[current_index]
        next_row = df.iloc[current_index + 1]

        if open_position is None:
            if decision_rng.random() >= RANDOM_ENTRY_PROBABILITY:
                continue

            atr_based_risk = max(float(row.atr_14), 0.0)
            if atr_based_risk <= 0:
                continue

            stop_loss_used = float(row.Close - (1.5 * atr_based_risk))
            take_profit_used = float(row.Close + (1.5 * atr_based_risk))
            if not _entry_is_still_valid(float(next_row.Open), stop_loss_used, take_profit_used):
                continue

            candidate_position = open_position_from_signal(
                signal_row=row,
                next_row=next_row,
                capital_before=capital,
                regime_at_entry=str(row.regime),
                entry_index=current_index + 1,
                rng=execution_rng,
                strategy_name="random",
                ticker=resolved_ticker,
                stop_loss_used=stop_loss_used,
                take_profit_used=take_profit_used,
            )
            if candidate_position is not None:
                open_position = candidate_position
                planned_exit_index = open_position.entry_index + decision_rng.randint(
                    RANDOM_HOLDING_PERIOD_MIN_BARS,
                    RANDOM_HOLDING_PERIOD_MAX_BARS,
                )
        else:
            exit_reason = _first_daily_exit_reason(
                row,
                open_position.stop_loss_used,
                open_position.take_profit_used,
            )
            if exit_reason is None and planned_exit_index is not None and current_index >= planned_exit_index:
                exit_reason = "random_time_exit"

            if exit_reason is None:
                continue

            trade_record = close_position_from_signal(
                position=open_position,
                next_row=next_row,
                exit_index=current_index + 1,
                rng=execution_rng,
                ticker=resolved_ticker,
                exit_reason=exit_reason,
            )
            trades.append(trade_record)
            capital = float(trade_record["capital_after"])
            open_position = None
            planned_exit_index = None

    return _finalize_trades(trades)


def run_strategy(
    agent_name: str,
    market_df: pd.DataFrame,
    random_decision_seed: int | None = None,
    *,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Dispatch one shared in-memory runner by strategy name."""
    if agent_name == "trend_pullback":
        return run_trend_pullback_strategy(market_df, ticker=ticker)
    if agent_name == "breakout_volume_momentum":
        return run_breakout_volume_momentum_strategy(market_df, ticker=ticker)
    if agent_name == "mean_reversion_vol_filter":
        return run_mean_reversion_vol_filter_strategy(market_df, ticker=ticker)
    if agent_name == "volatility_managed_tsmom":
        return run_volatility_managed_tsmom_strategy(market_df, ticker=ticker)
    if agent_name == "random":
        return run_random_strategy(
            market_df,
            decision_seed=random_decision_seed,
            ticker=ticker,
        )

    raise ValueError(f"Unsupported strategy name: {agent_name}")
