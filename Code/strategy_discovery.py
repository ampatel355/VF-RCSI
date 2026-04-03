"""Research-backed strategy discovery with honest train/validation/holdout testing.

This script is intentionally separate from the default single-ticker pipeline.
It allows broad candidate exploration without mutating the production strategy
set or the production classifier thresholds.

Design goals:
1. Use the same execution model and structure-preserving Monte Carlo null model.
2. Preserve scientific honesty:
   - keep full candidate logs (including failures)
   - apply multiple-testing correction
   - use explicit train/validation/holdout splits
3. Avoid brute-force micro-tuning and avoid lookahead.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:
    from artifact_provenance import write_dataframe_artifact
    from execution_model import (
        STARTING_CAPITAL,
        OpenPosition,
        TRADE_LOG_COLUMNS,
        build_execution_rng,
        close_position_from_signal,
        open_position_from_signal,
    )
    from monte_carlo import (
        MIN_RESEARCH_GRADE_SIMULATIONS,
        TRANSACTION_COST,
        adjust_trade_returns,
        benjamini_hochberg_adjusted_p_values,
        build_random_generator,
        calculate_actual_percentile,
        calculate_p_value,
        simulate_agent_null_cumulative_returns,
    )
    from momentum_relative_strength_agent import (
        load_aligned_universe_data,
        resolve_universe,
    )
    from regimes import build_regime_dataframe_for_ticker
    from strategy_simulator import run_strategy as run_existing_strategy
    from strategy_verdicts import (
        apply_multiple_testing_guard,
        classify_metrics,
        evidence_label,
    )
except ModuleNotFoundError:
    from Code.artifact_provenance import write_dataframe_artifact
    from Code.execution_model import (
        STARTING_CAPITAL,
        OpenPosition,
        TRADE_LOG_COLUMNS,
        build_execution_rng,
        close_position_from_signal,
        open_position_from_signal,
    )
    from Code.monte_carlo import (
        MIN_RESEARCH_GRADE_SIMULATIONS,
        TRANSACTION_COST,
        adjust_trade_returns,
        benjamini_hochberg_adjusted_p_values,
        build_random_generator,
        calculate_actual_percentile,
        calculate_p_value,
        simulate_agent_null_cumulative_returns,
    )
    from Code.momentum_relative_strength_agent import (
        load_aligned_universe_data,
        resolve_universe,
    )
    from Code.regimes import build_regime_dataframe_for_ticker
    from Code.strategy_simulator import run_strategy as run_existing_strategy
    from Code.strategy_verdicts import (
        apply_multiple_testing_guard,
        classify_metrics,
        evidence_label,
    )


DISCOVERY_TICKERS = [
    symbol.strip().upper()
    for symbol in os.environ.get(
        "DISCOVERY_TICKERS",
        "SPY,QQQ,BTC-USD,EURUSD=X,ES=F",
    ).split(",")
    if symbol.strip()
]
DISCOVERY_MONTE_CARLO_SIMULATIONS = int(
    os.environ.get("DISCOVERY_MONTE_CARLO_SIMULATIONS", "1200")
)
HOLDOUT_MONTE_CARLO_SIMULATIONS = int(
    os.environ.get("HOLDOUT_MONTE_CARLO_SIMULATIONS", "5000")
)
ROBUSTNESS_SIMULATIONS = int(os.environ.get("DISCOVERY_ROBUSTNESS_SIMULATIONS", "1200"))
ROBUSTNESS_SEEDS = int(os.environ.get("DISCOVERY_ROBUSTNESS_SEEDS", "5"))
DISCOVERY_SEED = int(os.environ.get("DISCOVERY_SEED", "20260402"))
MIN_TRADES_FOR_INFERENCE = int(os.environ.get("DISCOVERY_MIN_TRADES_FOR_INFERENCE", "20"))

TRAIN_FRACTION = float(os.environ.get("DISCOVERY_TRAIN_FRACTION", "0.60"))
VALIDATION_FRACTION = float(os.environ.get("DISCOVERY_VALIDATION_FRACTION", "0.20"))

DATA_CLEAN_DIR = Path(__file__).resolve().parents[1] / "Data_Clean"
OUTPUT_CANDIDATE_RESULTS_PATH = DATA_CLEAN_DIR / "strategy_discovery_candidate_results.csv"
OUTPUT_SELECTION_PATH = DATA_CLEAN_DIR / "strategy_discovery_selection_summary.csv"
OUTPUT_HOLDOUT_PATH = DATA_CLEAN_DIR / "strategy_discovery_holdout_deep_results.csv"
OUTPUT_ROBUSTNESS_PATH = DATA_CLEAN_DIR / "strategy_discovery_seed_robustness.csv"
OUTPUT_SOURCES_PATH = DATA_CLEAN_DIR / "strategy_discovery_sources.csv"


SOURCE_LIBRARY: dict[str, dict[str, str]] = {
    "tsmom_moskowitz_2012": {
        "citation": "Moskowitz, Ooi, Pedersen (2012), Time Series Momentum, Journal of Financial Economics.",
        "url": "https://doi.org/10.1016/j.jfineco.2011.11.003",
    },
    "trend_century_hurst_2017": {
        "citation": "Hurst, Ooi, Pedersen (2017), A Century of Evidence on Trend-Following Investing, Journal of Portfolio Management.",
        "url": "https://www.aqr.com/insights/research/journal-article/a-century-of-evidence-on-trend-following-investing",
    },
    "momentum_jegadeesh_titman_1993": {
        "citation": "Jegadeesh, Titman (1993), Returns to Buying Winners and Selling Losers, Journal of Finance.",
        "url": "https://www.bauer.uh.edu/rsusmel/phd/jegadeesh-titman93.pdf",
    },
    "vol_managed_moreira_muir_2016": {
        "citation": "Moreira, Muir (2016/2017), Volatility Managed Portfolios.",
        "url": "https://www.nber.org/papers/w22208",
    },
    "technical_rules_brock_1992": {
        "citation": "Brock, Lakonishok, LeBaron (1992), Simple Technical Trading Rules and the Stochastic Properties of Stock Returns, Journal of Finance.",
        "url": "https://www.jstor.org/stable/i340162",
    },
    "data_snooping_sullivan_1999": {
        "citation": "Sullivan, Timmermann, White (1999), Data-Snooping, Technical Trading Rule Performance, and the Bootstrap, Journal of Finance.",
        "url": "https://ideas.repec.org/a/bla/jfinan/v54y1999i5p1647-1691.html",
    },
    "faber_tactical_2007": {
        "citation": "Faber (2007), A Quantitative Approach to Tactical Asset Allocation.",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=962461",
    },
    "connors_short_term_reversion": {
        "citation": "Connors, Alvarez (2009), Short-Term Trading Strategies That Work.",
        "url": "https://www.wiley.com/en-us/Short+Term+Trading+Strategies+That+Work-p-9780470444581",
    },
    "turn_of_month_lakonishok_1988": {
        "citation": "Lakonishok, Smidt (1988), Are Seasonal Anomalies Real?, Review of Financial Studies.",
        "url": "https://doi.org/10.1093/rfs/1.4.403",
    },
    "donchian_channel_breakout": {
        "citation": "Donchian channel breakout trend-following framework (industry standard managed-futures rule family).",
        "url": "https://www.investopedia.com/terms/d/donchianchannels.asp",
    },
}


def _prepare_market_df(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    """Validate and normalize a candidate market DataFrame."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(
            "Candidate market data is missing required columns: "
            + ", ".join(missing_columns)
        )
    prepared = df.copy()
    prepared["Date"] = pd.to_datetime(prepared["Date"], errors="coerce")
    numeric_columns = [column for column in required_columns if column not in {"Date", "regime"}]
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared["regime"] = prepared["regime"].astype(str)
    prepared = (
        prepared.dropna(subset=required_columns)
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )
    return prepared


def _safe_stop(value: float | int | None) -> float | None:
    """Convert an optional numeric threshold into a clean float."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def _entry_is_still_valid(
    next_open: float,
    stop_loss_used: float | None,
    take_profit_used: float | None,
) -> bool:
    """Reject entries that gap through stop/target before the trade can open."""
    if not pd.notna(next_open) or float(next_open) <= 0:
        return False
    if stop_loss_used is not None and float(next_open) <= float(stop_loss_used):
        return False
    if take_profit_used is not None and float(next_open) >= float(take_profit_used):
        return False
    return True


def _has_room_for_entry_and_exit(current_index: int, market_length: int) -> bool:
    """Ensure one next-open entry still leaves one later open available for exit."""
    return (current_index + 1) < (market_length - 1)


def _first_daily_exit_reason(
    row: pd.Series,
    stop_loss_used: float | None,
    take_profit_used: float | None,
) -> str | None:
    """Check stop/target breaches on completed-bar data."""
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


@dataclass(frozen=True)
class EntryPlan:
    """Entry decision returned by one candidate entry rule."""

    entry_reason: str
    stop_loss: float | None = None
    take_profit: float | None = None
    capital_fraction: float | None = None
    state: dict[str, float | int | str] | None = None


EntryRule = Callable[[pd.Series, pd.Series, int, pd.DataFrame], EntryPlan | None]
ExitRule = Callable[[pd.Series, pd.Series, int, pd.DataFrame, OpenPosition, dict[str, float | int | str]], str | None]
StateUpdateRule = Callable[[pd.Series, pd.Series, int, pd.DataFrame, dict[str, float | int | str]], None]


def run_custom_strategy(
    *,
    market_df: pd.DataFrame,
    ticker: str,
    strategy_name: str,
    required_columns: list[str],
    entry_rule: EntryRule,
    exit_rule: ExitRule,
    state_update_rule: StateUpdateRule | None = None,
) -> pd.DataFrame:
    """Run one custom long-only strategy with shared execution realism."""
    df = _prepare_market_df(market_df, required_columns)
    execution_rng = build_execution_rng(strategy_name, ticker)
    trades: list[dict[str, object]] = []
    capital = STARTING_CAPITAL
    open_position: OpenPosition | None = None
    position_state: dict[str, float | int | str] = {}
    signal_count = 0
    rejected_signal_count = 0

    for current_index in range(1, len(df) - 1):
        previous_row = df.iloc[current_index - 1]
        row = df.iloc[current_index]
        next_row = df.iloc[current_index + 1]

        if open_position is None:
            plan = entry_rule(row, previous_row, current_index, df)
            if plan is None:
                continue
            signal_count += 1
            if not _has_room_for_entry_and_exit(current_index, len(df)):
                rejected_signal_count += 1
                continue
            if not _entry_is_still_valid(float(next_row.Open), plan.stop_loss, plan.take_profit):
                rejected_signal_count += 1
                continue
            candidate_position = open_position_from_signal(
                signal_row=row,
                next_row=next_row,
                capital_before=capital,
                regime_at_entry=str(row.regime),
                entry_index=current_index + 1,
                rng=execution_rng,
                strategy_name=strategy_name,
                ticker=ticker,
                stop_loss_used=plan.stop_loss,
                take_profit_used=plan.take_profit,
                capital_fraction_override=plan.capital_fraction,
                entry_reason=plan.entry_reason,
            )
            if candidate_position is None:
                rejected_signal_count += 1
                continue
            open_position = candidate_position
            position_state = dict(plan.state or {})
            position_state.setdefault("entry_signal_index", int(current_index))
            continue

        if state_update_rule is not None:
            state_update_rule(row, previous_row, current_index, df, position_state)
        exit_reason = _first_daily_exit_reason(
            row=row,
            stop_loss_used=open_position.stop_loss_used,
            take_profit_used=open_position.take_profit_used,
        )
        if exit_reason is None:
            exit_reason = exit_rule(
                row,
                previous_row,
                current_index,
                df,
                open_position,
                position_state,
            )
        if exit_reason is None:
            continue
        trade_record = close_position_from_signal(
            position=open_position,
            next_row=next_row,
            exit_index=current_index + 1,
            rng=execution_rng,
            ticker=ticker,
            exit_reason=exit_reason,
        )
        trades.append(trade_record)
        capital = float(trade_record["capital_after"])
        open_position = None
        position_state = {}

    if open_position is not None and len(df) >= 2:
        final_row = df.iloc[-1]
        trade_record = close_position_from_signal(
            position=open_position,
            next_row=final_row,
            exit_index=len(df) - 1,
            rng=execution_rng,
            ticker=ticker,
            exit_reason="end_of_sample",
        )
        trades.append(trade_record)
        open_position = None

    trades_df = pd.DataFrame(trades, columns=TRADE_LOG_COLUMNS)
    trades_df.attrs["signal_count"] = int(signal_count)
    trades_df.attrs["rejected_signal_count"] = int(rejected_signal_count)
    return trades_df


def augment_candidate_features(market_df: pd.DataFrame) -> pd.DataFrame:
    """Add extra rolling features used by discovery candidates."""
    df = market_df.copy()
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    df["trailing_return_21"] = (df["Close"] / df["Close"].shift(21)) - 1.0
    df["trailing_return_63"] = (df["Close"] / df["Close"].shift(63)) - 1.0
    df["trailing_return_126"] = (df["Close"] / df["Close"].shift(126)) - 1.0
    df["trailing_return_252"] = (df["Close"] / df["Close"].shift(252)) - 1.0
    df["rolling_high_20_prev"] = df["High"].rolling(window=20).max().shift(1)
    df["rolling_low_20_prev"] = df["Low"].rolling(window=20).min().shift(1)
    df["rolling_high_55_prev"] = df["High"].rolling(window=55).max().shift(1)
    df["rolling_low_10_prev"] = df["Low"].rolling(window=10).min().shift(1)
    df["rolling_low_5_prev"] = df["Low"].rolling(window=5).min().shift(1)
    return df


@dataclass(frozen=True)
class CandidateSpec:
    """One strategy candidate definition."""

    candidate_id: str
    display_name: str
    family: str
    hypothesis: str
    source_keys: tuple[str, ...]
    required_columns: tuple[str, ...]
    runner: Callable[[pd.DataFrame, str], pd.DataFrame]
    null_model_agent_name: str | None = None
    is_control: bool = False


def _build_candidates() -> list[CandidateSpec]:
    """Create the fixed discovery candidate library."""
    base_required = (
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "avg_volume_20",
        "ema_20",
        "sma_20",
        "sma_50",
        "sma_100",
        "sma_200",
        "rsi_14",
        "adx_14",
        "atr_14",
        "atr_percent_ratio_60",
        "bollinger_lower",
        "bollinger_mid",
        "bollinger_width_ratio_60",
        "macd_line",
        "macd_signal",
        "zscore_20",
        "ma_50_over_ma_200",
        "volume_ratio_20",
        "trailing_return_20",
        "trailing_return_21",
        "trailing_return_63",
        "trailing_return_126",
        "trailing_return_252",
        "rolling_high_20_prev",
        "rolling_high_55_prev",
        "rolling_low_20_prev",
        "rolling_low_10_prev",
        "rolling_low_5_prev",
        "regime",
    )

    def tsmom_12m_filter(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        def entry_rule(row: pd.Series, _: pd.Series, _i: int, _df: pd.DataFrame) -> EntryPlan | None:
            if (
                row.trailing_return_252 > 0
                and row.trailing_return_21 > 0
                and row.Close > row.sma_200
                and row.macd_line > row.macd_signal
            ):
                return EntryPlan(
                    entry_reason="tsmom_12m_positive_with_trend_filter",
                    stop_loss=_safe_stop(row.Close - (2.5 * row.atr_14)),
                    take_profit=_safe_stop(row.Close + (5.0 * row.atr_14)),
                    state={"highest_close": float(row.Close)},
                )
            return None

        def update_rule(
            row: pd.Series,
            _: pd.Series,
            _i: int,
            _df: pd.DataFrame,
            state: dict[str, float | int | str],
        ) -> None:
            highest_close = float(state.get("highest_close", row.Close))
            if float(row.Close) > highest_close:
                state["highest_close"] = float(row.Close)

        def exit_rule(
            row: pd.Series,
            _previous_row: pd.Series,
            current_index: int,
            _df: pd.DataFrame,
            position: OpenPosition,
            state: dict[str, float | int | str],
        ) -> str | None:
            trailing_stop = float(state.get("highest_close", row.Close)) - (3.0 * float(row.atr_14))
            if float(row.Low) <= trailing_stop:
                return "trailing_stop"
            if row.trailing_return_21 < 0 or row.Close < row.sma_200:
                return "momentum_or_trend_lost"
            if (current_index - int(position.entry_index)) >= 126:
                return "time_stop_126"
            return None

        return run_custom_strategy(
            market_df=market_df,
            ticker=ticker,
            strategy_name="cand_tsmom_12m_filter",
            required_columns=list(base_required),
            entry_rule=entry_rule,
            exit_rule=exit_rule,
            state_update_rule=update_rule,
        )

    def ma_cross_50_200(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        def entry_rule(row: pd.Series, previous_row: pd.Series, _i: int, _df: pd.DataFrame) -> EntryPlan | None:
            crossed_up = row.sma_50 > row.sma_200 and previous_row.sma_50 <= previous_row.sma_200
            if crossed_up and row.Close > row.sma_50:
                return EntryPlan(
                    entry_reason="golden_cross",
                    stop_loss=_safe_stop(row.Close - (2.0 * row.atr_14)),
                    take_profit=_safe_stop(row.Close + (4.0 * row.atr_14)),
                )
            return None

        def exit_rule(
            row: pd.Series,
            previous_row: pd.Series,
            current_index: int,
            _df: pd.DataFrame,
            position: OpenPosition,
            _state: dict[str, float | int | str],
        ) -> str | None:
            crossed_down = row.sma_50 < row.sma_200 and previous_row.sma_50 >= previous_row.sma_200
            if crossed_down or row.Close < row.sma_200:
                return "death_cross_or_trend_break"
            if (current_index - int(position.entry_index)) >= 126:
                return "time_stop_126"
            return None

        return run_custom_strategy(
            market_df=market_df,
            ticker=ticker,
            strategy_name="cand_ma_cross_50_200",
            required_columns=list(base_required),
            entry_rule=entry_rule,
            exit_rule=exit_rule,
        )

    def donchian_55_breakout(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        def entry_rule(row: pd.Series, _prev: pd.Series, _i: int, _df: pd.DataFrame) -> EntryPlan | None:
            if (
                row.Close > row.rolling_high_55_prev
                and row.trailing_return_63 > 0
                and row.volume_ratio_20 >= 0.9
            ):
                stop_loss = _safe_stop(min(row.rolling_low_20_prev, row.Close - (2.0 * row.atr_14)))
                return EntryPlan(
                    entry_reason="donchian_55_breakout",
                    stop_loss=stop_loss,
                    take_profit=_safe_stop(row.Close + (4.0 * row.atr_14)),
                )
            return None

        def exit_rule(
            row: pd.Series,
            _previous_row: pd.Series,
            current_index: int,
            _df: pd.DataFrame,
            position: OpenPosition,
            _state: dict[str, float | int | str],
        ) -> str | None:
            if row.Close < row.ema_20:
                return "ema20_break"
            if row.macd_line < row.macd_signal and row.Close < row.sma_50:
                return "momentum_fade"
            if (current_index - int(position.entry_index)) >= 80:
                return "time_stop_80"
            return None

        return run_custom_strategy(
            market_df=market_df,
            ticker=ticker,
            strategy_name="cand_donchian_55_breakout",
            required_columns=list(base_required),
            entry_rule=entry_rule,
            exit_rule=exit_rule,
        )

    def breakout_20_volume(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        def entry_rule(row: pd.Series, _prev: pd.Series, _i: int, _df: pd.DataFrame) -> EntryPlan | None:
            if (
                row.Close > row.rolling_high_20_prev
                and row.volume_ratio_20 >= 1.2
                and row.macd_line > row.macd_signal
            ):
                stop_loss = _safe_stop(min(row.rolling_low_10_prev, row.Close - (1.8 * row.atr_14)))
                return EntryPlan(
                    entry_reason="breakout_20_volume",
                    stop_loss=stop_loss,
                    take_profit=_safe_stop(row.Close + (3.5 * row.atr_14)),
                )
            return None

        def exit_rule(
            row: pd.Series,
            _previous_row: pd.Series,
            current_index: int,
            _df: pd.DataFrame,
            position: OpenPosition,
            _state: dict[str, float | int | str],
        ) -> str | None:
            if row.Close < row.ema_20:
                return "ema20_break"
            if (current_index - int(position.entry_index)) >= 40:
                return "time_stop_40"
            return None

        return run_custom_strategy(
            market_df=market_df,
            ticker=ticker,
            strategy_name="cand_breakout_20_volume",
            required_columns=list(base_required),
            entry_rule=entry_rule,
            exit_rule=exit_rule,
        )

    def trend_pullback_ema_rsi(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        def entry_rule(row: pd.Series, previous_row: pd.Series, _i: int, _df: pd.DataFrame) -> EntryPlan | None:
            if (
                row.sma_50 > row.sma_200
                and row.Close > row.sma_200
                and row.Low <= row.ema_20
                and 40.0 <= row.rsi_14 <= 65.0
                and row.macd_line > row.macd_signal
                and row.Close >= previous_row.Close
            ):
                return EntryPlan(
                    entry_reason="trend_pullback_ema_rsi",
                    stop_loss=_safe_stop(min(row.rolling_low_10_prev, row.Close - (2.0 * row.atr_14))),
                    take_profit=_safe_stop(row.Close + (3.0 * row.atr_14)),
                )
            return None

        def exit_rule(
            row: pd.Series,
            _previous_row: pd.Series,
            current_index: int,
            _df: pd.DataFrame,
            position: OpenPosition,
            _state: dict[str, float | int | str],
        ) -> str | None:
            if row.Close < row.sma_50:
                return "trend_filter_fail"
            if row.rsi_14 >= 75:
                return "rsi_exhaustion"
            if (current_index - int(position.entry_index)) >= 45:
                return "time_stop_45"
            return None

        return run_custom_strategy(
            market_df=market_df,
            ticker=ticker,
            strategy_name="cand_trend_pullback_ema_rsi",
            required_columns=list(base_required),
            entry_rule=entry_rule,
            exit_rule=exit_rule,
        )

    def trend_pullback_adx(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        def entry_rule(row: pd.Series, previous_row: pd.Series, _i: int, _df: pd.DataFrame) -> EntryPlan | None:
            if (
                row.sma_50 > row.sma_200
                and row.adx_14 >= 18
                and row.trailing_return_20 > 0
                and row.Low <= row.ema_20 * 1.01
                and row.Close >= row.ema_20
                and row.Close > previous_row.Close
            ):
                return EntryPlan(
                    entry_reason="trend_pullback_adx",
                    stop_loss=_safe_stop(min(row.rolling_low_10_prev, row.Close - (2.0 * row.atr_14))),
                    take_profit=_safe_stop(row.Close + (3.5 * row.atr_14)),
                )
            return None

        def exit_rule(
            row: pd.Series,
            _previous_row: pd.Series,
            current_index: int,
            _df: pd.DataFrame,
            position: OpenPosition,
            _state: dict[str, float | int | str],
        ) -> str | None:
            if row.Close < row.sma_50 or row.sma_50 <= row.sma_200:
                return "trend_break"
            if row.macd_line < row.macd_signal and row.rsi_14 < 45:
                return "momentum_loss"
            if (current_index - int(position.entry_index)) >= 60:
                return "time_stop_60"
            return None

        return run_custom_strategy(
            market_df=market_df,
            ticker=ticker,
            strategy_name="cand_trend_pullback_adx",
            required_columns=list(base_required),
            entry_rule=entry_rule,
            exit_rule=exit_rule,
        )

    def mean_reversion_bbands(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        def entry_rule(row: pd.Series, _prev: pd.Series, _i: int, _df: pd.DataFrame) -> EntryPlan | None:
            if (
                row.adx_14 < 18
                and row.Close <= row.bollinger_lower
                and row.rsi_14 < 35
                and row.atr_percent_ratio_60 <= 1.8
            ):
                stop_anchor = min(row.rolling_low_5_prev, row.Close - (1.5 * row.atr_14))
                return EntryPlan(
                    entry_reason="bbands_mean_reversion",
                    stop_loss=_safe_stop(stop_anchor),
                    take_profit=_safe_stop(row.bollinger_mid),
                )
            return None

        def exit_rule(
            row: pd.Series,
            _previous_row: pd.Series,
            current_index: int,
            _df: pd.DataFrame,
            position: OpenPosition,
            _state: dict[str, float | int | str],
        ) -> str | None:
            if row.Close >= row.bollinger_mid or row.rsi_14 >= 55:
                return "mean_reversion_complete"
            if (current_index - int(position.entry_index)) >= 15:
                return "time_stop_15"
            return None

        return run_custom_strategy(
            market_df=market_df,
            ticker=ticker,
            strategy_name="cand_mean_reversion_bbands",
            required_columns=list(base_required),
            entry_rule=entry_rule,
            exit_rule=exit_rule,
        )

    def mean_reversion_zscore(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        def entry_rule(row: pd.Series, _prev: pd.Series, _i: int, _df: pd.DataFrame) -> EntryPlan | None:
            if (
                row.zscore_20 <= -1.75
                and row.atr_percent_ratio_60 <= 1.6
                and abs(float(row.ma_50_over_ma_200)) <= 0.02
                and row.regime.lower() != "stressed"
            ):
                return EntryPlan(
                    entry_reason="zscore_mean_reversion",
                    stop_loss=_safe_stop(row.Close - (1.5 * row.atr_14)),
                    take_profit=_safe_stop(row.ema_20),
                )
            return None

        def exit_rule(
            row: pd.Series,
            _previous_row: pd.Series,
            current_index: int,
            _df: pd.DataFrame,
            position: OpenPosition,
            _state: dict[str, float | int | str],
        ) -> str | None:
            if row.zscore_20 >= -0.10 or row.Close >= row.ema_20:
                return "zscore_reverted"
            if (current_index - int(position.entry_index)) >= 12:
                return "time_stop_12"
            return None

        return run_custom_strategy(
            market_df=market_df,
            ticker=ticker,
            strategy_name="cand_mean_reversion_zscore",
            required_columns=list(base_required),
            entry_rule=entry_rule,
            exit_rule=exit_rule,
        )

    def dual_momentum_trend_filter(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        def entry_rule(row: pd.Series, _prev: pd.Series, _i: int, _df: pd.DataFrame) -> EntryPlan | None:
            if (
                row.trailing_return_126 > 0
                and row.trailing_return_252 > 0
                and row.Close > row.sma_200
                and 45.0 <= row.rsi_14 <= 75.0
            ):
                return EntryPlan(
                    entry_reason="dual_momentum_absolute_filter",
                    stop_loss=_safe_stop(row.Close - (2.2 * row.atr_14)),
                    take_profit=_safe_stop(row.Close + (4.5 * row.atr_14)),
                )
            return None

        def exit_rule(
            row: pd.Series,
            _previous_row: pd.Series,
            current_index: int,
            _df: pd.DataFrame,
            position: OpenPosition,
            _state: dict[str, float | int | str],
        ) -> str | None:
            if row.trailing_return_21 < 0 or row.Close < row.sma_200:
                return "absolute_momentum_lost"
            if (current_index - int(position.entry_index)) >= 90:
                return "time_stop_90"
            return None

        return run_custom_strategy(
            market_df=market_df,
            ticker=ticker,
            strategy_name="cand_dual_momentum_trend_filter",
            required_columns=list(base_required),
            entry_rule=entry_rule,
            exit_rule=exit_rule,
        )

    def vol_managed_momentum(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        def entry_rule(row: pd.Series, _prev: pd.Series, _i: int, _df: pd.DataFrame) -> EntryPlan | None:
            if (
                row.trailing_return_126 > 0
                and row.trailing_return_63 > 0
                and row.macd_line > row.macd_signal
                and row.Close > row.sma_200
            ):
                volatility_scale = float(np.clip(1.25 / max(float(row.atr_percent_ratio_60), 0.25), 0.25, 1.0))
                return EntryPlan(
                    entry_reason="vol_managed_momentum",
                    stop_loss=_safe_stop(row.Close - (2.5 * row.atr_14)),
                    take_profit=_safe_stop(row.Close + (5.0 * row.atr_14)),
                    capital_fraction=volatility_scale,
                    state={"highest_close": float(row.Close)},
                )
            return None

        def update_rule(
            row: pd.Series,
            _: pd.Series,
            _i: int,
            _df: pd.DataFrame,
            state: dict[str, float | int | str],
        ) -> None:
            state["highest_close"] = max(float(state.get("highest_close", row.Close)), float(row.Close))

        def exit_rule(
            row: pd.Series,
            _previous_row: pd.Series,
            current_index: int,
            _df: pd.DataFrame,
            position: OpenPosition,
            state: dict[str, float | int | str],
        ) -> str | None:
            trailing_stop = float(state.get("highest_close", row.Close)) - (3.0 * float(row.atr_14))
            if float(row.Low) <= trailing_stop:
                return "trailing_stop"
            if row.trailing_return_21 < 0:
                return "short_momentum_flip"
            if (current_index - int(position.entry_index)) >= 90:
                return "time_stop_90"
            return None

        return run_custom_strategy(
            market_df=market_df,
            ticker=ticker,
            strategy_name="cand_vol_managed_momentum",
            required_columns=list(base_required),
            entry_rule=entry_rule,
            exit_rule=exit_rule,
            state_update_rule=update_rule,
        )

    def hybrid_trend_breakout_pullback(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        def entry_rule(row: pd.Series, previous_row: pd.Series, _i: int, _df: pd.DataFrame) -> EntryPlan | None:
            trend_filter = row.sma_50 > row.sma_200 and row.Close > row.sma_200
            breakout_leg = row.Close > row.rolling_high_55_prev
            pullback_leg = row.Low <= row.ema_20 and row.Close >= row.ema_20 and row.Close > previous_row.Close
            quality_filter = row.trailing_return_63 > 0 and row.volume_ratio_20 >= 0.9
            if trend_filter and quality_filter and (breakout_leg or pullback_leg):
                stop_loss = _safe_stop(min(row.rolling_low_20_prev, row.Close - (2.0 * row.atr_14)))
                return EntryPlan(
                    entry_reason="hybrid_trend_breakout_pullback",
                    stop_loss=stop_loss,
                    take_profit=_safe_stop(row.Close + (4.0 * row.atr_14)),
                )
            return None

        def exit_rule(
            row: pd.Series,
            _previous_row: pd.Series,
            current_index: int,
            _df: pd.DataFrame,
            position: OpenPosition,
            _state: dict[str, float | int | str],
        ) -> str | None:
            if row.Close < row.sma_50:
                return "trend_support_lost"
            if row.macd_line < row.macd_signal and row.rsi_14 < 45:
                return "momentum_filter_fail"
            if (current_index - int(position.entry_index)) >= 70:
                return "time_stop_70"
            return None

        return run_custom_strategy(
            market_df=market_df,
            ticker=ticker,
            strategy_name="cand_hybrid_trend_breakout_pullback",
            required_columns=list(base_required),
            entry_rule=entry_rule,
            exit_rule=exit_rule,
        )

    def regime_filtered_momentum(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        def entry_rule(row: pd.Series, _prev: pd.Series, _i: int, _df: pd.DataFrame) -> EntryPlan | None:
            if (
                str(row.regime).lower() != "stressed"
                and row.trailing_return_63 > 0
                and row.Close > row.sma_100
                and row.macd_line > row.macd_signal
            ):
                return EntryPlan(
                    entry_reason="regime_filtered_momentum",
                    stop_loss=_safe_stop(row.Close - (2.0 * row.atr_14)),
                    take_profit=_safe_stop(row.Close + (4.0 * row.atr_14)),
                )
            return None

        def exit_rule(
            row: pd.Series,
            _previous_row: pd.Series,
            current_index: int,
            _df: pd.DataFrame,
            position: OpenPosition,
            _state: dict[str, float | int | str],
        ) -> str | None:
            if str(row.regime).lower() == "stressed":
                return "regime_stress_exit"
            if row.Close < row.sma_100 or row.trailing_return_21 < 0:
                return "momentum_lost"
            if (current_index - int(position.entry_index)) >= 60:
                return "time_stop_60"
            return None

        return run_custom_strategy(
            market_df=market_df,
            ticker=ticker,
            strategy_name="cand_regime_filtered_momentum",
            required_columns=list(base_required),
            entry_rule=entry_rule,
            exit_rule=exit_rule,
        )

    def control_existing_validation(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        return run_existing_strategy(
            "trend_momentum_verification",
            market_df,
            ticker=ticker,
        )

    def control_random(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        seed_component = sum((index + 1) * ord(ch) for index, ch in enumerate(ticker))
        return run_existing_strategy(
            "random",
            market_df,
            random_decision_seed=DISCOVERY_SEED + seed_component,
            ticker=ticker,
        )

    def existing_trend_pullback(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        return run_existing_strategy("trend_pullback", market_df, ticker=ticker)

    def existing_breakout_volume_momentum(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        return run_existing_strategy("breakout_volume_momentum", market_df, ticker=ticker)

    def existing_mean_reversion_vol_filter(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        return run_existing_strategy("mean_reversion_vol_filter", market_df, ticker=ticker)

    def existing_momentum_relative_strength(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        return run_existing_strategy("momentum_relative_strength", market_df, ticker=ticker)

    def existing_connors_rsi2_pullback(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        return run_existing_strategy("connors_rsi2_pullback", market_df, ticker=ticker)

    def existing_turn_of_month_seasonality(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        return run_existing_strategy("turn_of_month_seasonality", market_df, ticker=ticker)

    def existing_donchian_trend_reentry(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        return run_existing_strategy("donchian_trend_reentry", market_df, ticker=ticker)

    def build_cross_asset_monthly_momentum_runner(
        *,
        strategy_name: str,
        primary_lookback: int,
        secondary_lookback: int | None,
        use_trailing_stop: bool,
    ) -> Callable[[pd.DataFrame, str], pd.DataFrame]:
        """Create one monthly cross-asset momentum runner."""
        if primary_lookback < 20:
            raise ValueError("primary_lookback must be at least 20 bars.")

        max_lookback = max(
            primary_lookback,
            secondary_lookback if secondary_lookback is not None else primary_lookback,
        )

        def runner(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
            """Run one literature-style monthly cross-asset momentum rotation."""
            allowed_dates = set(pd.to_datetime(market_df["Date"], errors="coerce").dropna().tolist())
            universe = resolve_universe(ticker)
            common_dates, asset_frames = load_aligned_universe_data(universe)
            filtered_common_dates = [date for date in common_dates if date in allowed_dates]
            if len(filtered_common_dates) < (max_lookback + 8):
                return pd.DataFrame(columns=TRADE_LOG_COLUMNS)

            filtered_frames: dict[str, pd.DataFrame] = {}
            for asset_ticker, asset_df in asset_frames.items():
                filtered_df = asset_df.loc[asset_df["Date"].isin(filtered_common_dates)].copy()
                filtered_df = filtered_df.sort_values("Date").reset_index(drop=True)
                if len(filtered_df) != len(filtered_common_dates):
                    continue
                filtered_df["mom_primary"] = (
                    filtered_df["Close"] / filtered_df["Close"].shift(primary_lookback)
                ) - 1.0
                if secondary_lookback is not None:
                    filtered_df["mom_secondary"] = (
                        filtered_df["Close"] / filtered_df["Close"].shift(secondary_lookback)
                    ) - 1.0
                else:
                    filtered_df["mom_secondary"] = np.nan
                filtered_frames[asset_ticker] = filtered_df

            if len(filtered_frames) < 2:
                return pd.DataFrame(columns=TRADE_LOG_COLUMNS)

            execution_rng = build_execution_rng(strategy_name, ticker)
            trades: list[dict[str, object]] = []
            capital = STARTING_CAPITAL
            open_position: OpenPosition | None = None
            current_asset_ticker: str | None = None
            highest_close_since_entry: float | None = None
            signal_count = 0
            rejected_signal_count = 0
            date_series = pd.Series(pd.to_datetime(filtered_common_dates))

            def select_target_asset(index: int) -> str | None:
                ranked: list[tuple[str, float]] = []
                for asset_ticker, asset_df in filtered_frames.items():
                    row = asset_df.iloc[index]
                    momentum_primary = float(row["mom_primary"]) if pd.notna(row["mom_primary"]) else np.nan
                    momentum_secondary = float(row["mom_secondary"]) if pd.notna(row["mom_secondary"]) else np.nan
                    if not np.isfinite(momentum_primary):
                        continue
                    if momentum_primary <= 0:
                        continue
                    # Optional secondary filter keeps only broader uptrends.
                    if np.isfinite(momentum_secondary) and momentum_secondary <= 0:
                        continue
                    if float(row["Close"]) <= float(row["sma_50"]):
                        continue
                    ranked.append((asset_ticker, momentum_primary))
                if not ranked:
                    return None
                ranked.sort(key=lambda item: item[1], reverse=True)
                return ranked[0][0]

            for current_index in range(max_lookback, len(date_series) - 1):
                current_date = date_series.iloc[current_index]
                next_date = date_series.iloc[current_index + 1]
                rebalance_today = (
                    current_date.month != next_date.month
                    or current_date.year != next_date.year
                )
                target_asset = select_target_asset(current_index) if rebalance_today else None

                if open_position is not None and current_asset_ticker is not None:
                    active_df = filtered_frames[current_asset_ticker]
                    row = active_df.iloc[current_index]
                    next_row = active_df.iloc[current_index + 1]
                    if use_trailing_stop:
                        highest_close_since_entry = (
                            float(row.Close)
                            if highest_close_since_entry is None
                            else max(highest_close_since_entry, float(row.Close))
                        )
                        trailing_stop = highest_close_since_entry - (3.0 * float(row.atr_14))
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
                    if exit_reason is None and rebalance_today and target_asset != current_asset_ticker:
                        exit_reason = (
                            "rebalance_rotation"
                            if target_asset is not None
                            else "relative_strength_to_cash"
                        )
                    if exit_reason is not None:
                        trade_record = close_position_from_signal(
                            position=open_position,
                            next_row=next_row,
                            exit_index=current_index + 1,
                            rng=execution_rng,
                            ticker=current_asset_ticker,
                            exit_reason=exit_reason,
                        )
                        trades.append(trade_record)
                        capital = float(trade_record["capital_after"])
                        open_position = None
                        current_asset_ticker = None
                        highest_close_since_entry = None

                if open_position is None and rebalance_today and target_asset is not None:
                    if (current_index + 1) >= (len(date_series) - 1):
                        rejected_signal_count += 1
                        continue
                    target_df = filtered_frames[target_asset]
                    row = target_df.iloc[current_index]
                    next_row = target_df.iloc[current_index + 1]
                    initial_stop = (
                        float(row.Close - (3.0 * float(row.atr_14)))
                        if use_trailing_stop
                        else None
                    )
                    signal_count += 1
                    if not _entry_is_still_valid(float(next_row.Open), initial_stop, None):
                        rejected_signal_count += 1
                        continue
                    candidate_position = open_position_from_signal(
                        signal_row=row,
                        next_row=next_row,
                        capital_before=capital,
                        regime_at_entry=str(row.regime),
                        entry_index=current_index + 1,
                        rng=execution_rng,
                        strategy_name=strategy_name,
                        ticker=target_asset,
                        stop_loss_used=initial_stop,
                        take_profit_used=None,
                        entry_reason=strategy_name,
                    )
                    if candidate_position is None:
                        rejected_signal_count += 1
                        continue
                    open_position = candidate_position
                    current_asset_ticker = target_asset
                    highest_close_since_entry = float(row.Close)

            if open_position is not None and current_asset_ticker is not None:
                final_df = filtered_frames[current_asset_ticker]
                trade_record = close_position_from_signal(
                    position=open_position,
                    next_row=final_df.iloc[-1],
                    exit_index=len(date_series) - 1,
                    rng=execution_rng,
                    ticker=current_asset_ticker,
                    exit_reason="end_of_sample",
                )
                trades.append(trade_record)

            trades_df = pd.DataFrame(trades, columns=TRADE_LOG_COLUMNS)
            trades_df.attrs["signal_count"] = int(signal_count)
            trades_df.attrs["rejected_signal_count"] = int(rejected_signal_count)
            return trades_df

        return runner

    cross_asset_monthly_momentum = build_cross_asset_monthly_momentum_runner(
        strategy_name="cand_cross_asset_monthly_momentum",
        primary_lookback=126,
        secondary_lookback=252,
        use_trailing_stop=True,
    )
    cross_asset_monthly_momentum_nostop = build_cross_asset_monthly_momentum_runner(
        strategy_name="cand_cross_asset_monthly_momentum_nostop",
        primary_lookback=126,
        secondary_lookback=252,
        use_trailing_stop=False,
    )
    cross_asset_monthly_momentum_63 = build_cross_asset_monthly_momentum_runner(
        strategy_name="cand_cross_asset_monthly_momentum_63",
        primary_lookback=63,
        secondary_lookback=126,
        use_trailing_stop=False,
    )

    def faber_10m_timing(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        def entry_rule(row: pd.Series, _prev: pd.Series, _i: int, _df: pd.DataFrame) -> EntryPlan | None:
            if row.Close > row.sma_200:
                return EntryPlan(
                    entry_reason="faber_10m_timing",
                    stop_loss=_safe_stop(row.Close - (2.5 * row.atr_14)),
                    take_profit=None,
                )
            return None

        def exit_rule(
            row: pd.Series,
            _previous_row: pd.Series,
            _current_index: int,
            _df: pd.DataFrame,
            _position: OpenPosition,
            _state: dict[str, float | int | str],
        ) -> str | None:
            if row.Close < row.sma_200:
                return "below_200ma"
            return None

        return run_custom_strategy(
            market_df=market_df,
            ticker=ticker,
            strategy_name="cand_faber_10m_timing",
            required_columns=list(base_required),
            entry_rule=entry_rule,
            exit_rule=exit_rule,
        )

    def medium_term_momentum_63(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        def entry_rule(row: pd.Series, _prev: pd.Series, _i: int, _df: pd.DataFrame) -> EntryPlan | None:
            if (
                row.trailing_return_63 > 0
                and row.Close > row.sma_100
                and row.macd_line >= row.macd_signal
            ):
                return EntryPlan(
                    entry_reason="medium_term_momentum_63",
                    stop_loss=_safe_stop(row.Close - (2.0 * row.atr_14)),
                    take_profit=_safe_stop(row.Close + (4.0 * row.atr_14)),
                )
            return None

        def exit_rule(
            row: pd.Series,
            _previous_row: pd.Series,
            current_index: int,
            _df: pd.DataFrame,
            position: OpenPosition,
            _state: dict[str, float | int | str],
        ) -> str | None:
            if row.trailing_return_21 < 0 or row.Close < row.sma_50:
                return "momentum_or_trend_lost"
            if (current_index - int(position.entry_index)) >= 40:
                return "time_stop_40"
            return None

        return run_custom_strategy(
            market_df=market_df,
            ticker=ticker,
            strategy_name="cand_medium_term_momentum_63",
            required_columns=list(base_required),
            entry_rule=entry_rule,
            exit_rule=exit_rule,
        )

    def bb_squeeze_breakout(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        def entry_rule(row: pd.Series, _prev: pd.Series, _i: int, _df: pd.DataFrame) -> EntryPlan | None:
            squeeze = row.bollinger_width_ratio_60 <= 0.85
            breakout = row.Close > row.rolling_high_20_prev
            if squeeze and breakout and row.macd_line > row.macd_signal:
                return EntryPlan(
                    entry_reason="bb_squeeze_breakout",
                    stop_loss=_safe_stop(min(row.rolling_low_10_prev, row.Close - (1.8 * row.atr_14))),
                    take_profit=_safe_stop(row.Close + (3.2 * row.atr_14)),
                )
            return None

        def exit_rule(
            row: pd.Series,
            _previous_row: pd.Series,
            current_index: int,
            _df: pd.DataFrame,
            position: OpenPosition,
            _state: dict[str, float | int | str],
        ) -> str | None:
            if row.Close < row.ema_20:
                return "ema20_break"
            if (current_index - int(position.entry_index)) >= 35:
                return "time_stop_35"
            return None

        return run_custom_strategy(
            market_df=market_df,
            ticker=ticker,
            strategy_name="cand_bb_squeeze_breakout",
            required_columns=list(base_required),
            entry_rule=entry_rule,
            exit_rule=exit_rule,
        )

    def dual_horizon_trend(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        def entry_rule(row: pd.Series, _prev: pd.Series, _i: int, _df: pd.DataFrame) -> EntryPlan | None:
            if (
                row.sma_20 > row.sma_100
                and row.sma_100 > row.sma_200
                and row.trailing_return_63 > 0
                and row.rsi_14 > 50
            ):
                return EntryPlan(
                    entry_reason="dual_horizon_trend",
                    stop_loss=_safe_stop(row.Close - (2.2 * row.atr_14)),
                    take_profit=_safe_stop(row.Close + (4.2 * row.atr_14)),
                )
            return None

        def exit_rule(
            row: pd.Series,
            _previous_row: pd.Series,
            current_index: int,
            _df: pd.DataFrame,
            position: OpenPosition,
            _state: dict[str, float | int | str],
        ) -> str | None:
            if row.sma_20 < row.sma_100 or row.rsi_14 < 45:
                return "trend_horizon_break"
            if (current_index - int(position.entry_index)) >= 55:
                return "time_stop_55"
            return None

        return run_custom_strategy(
            market_df=market_df,
            ticker=ticker,
            strategy_name="cand_dual_horizon_trend",
            required_columns=list(base_required),
            entry_rule=entry_rule,
            exit_rule=exit_rule,
        )

    candidates: list[CandidateSpec] = [
        CandidateSpec(
            candidate_id="existing_trend_pullback",
            display_name="Existing Strategy: Trend + Pullback",
            family="existing_production",
            hypothesis="Current production trend-pullback implementation under split protocol.",
            source_keys=("trend_century_hurst_2017",),
            required_columns=base_required,
            runner=existing_trend_pullback,
            null_model_agent_name="trend_pullback",
        ),
        CandidateSpec(
            candidate_id="existing_breakout_volume_momentum",
            display_name="Existing Strategy: Breakout + Volume + Momentum",
            family="existing_production",
            hypothesis="Current production breakout implementation under split protocol.",
            source_keys=("technical_rules_brock_1992",),
            required_columns=base_required,
            runner=existing_breakout_volume_momentum,
            null_model_agent_name="breakout_volume_momentum",
        ),
        CandidateSpec(
            candidate_id="existing_mean_reversion_vol_filter",
            display_name="Existing Strategy: Mean Reversion + Volatility Filter",
            family="existing_production",
            hypothesis="Current production mean-reversion implementation under split protocol.",
            source_keys=("data_snooping_sullivan_1999",),
            required_columns=base_required,
            runner=existing_mean_reversion_vol_filter,
            null_model_agent_name="mean_reversion_vol_filter",
        ),
        CandidateSpec(
            candidate_id="existing_momentum_relative_strength",
            display_name="Existing Strategy: Momentum + Relative Strength",
            family="existing_production",
            hypothesis="Current production cross-asset momentum strategy under split protocol.",
            source_keys=("momentum_jegadeesh_titman_1993", "tsmom_moskowitz_2012"),
            required_columns=base_required,
            runner=existing_momentum_relative_strength,
            null_model_agent_name="momentum_relative_strength",
        ),
        CandidateSpec(
            candidate_id="existing_connors_rsi2_pullback",
            display_name="Existing Strategy: Connors RSI(2) Pullback",
            family="existing_production",
            hypothesis="Short-horizon RSI(2) washout entries inside structural uptrends.",
            source_keys=("connors_short_term_reversion",),
            required_columns=base_required,
            runner=existing_connors_rsi2_pullback,
            null_model_agent_name="connors_rsi2_pullback",
        ),
        CandidateSpec(
            candidate_id="existing_turn_of_month_seasonality",
            display_name="Existing Strategy: Turn-of-Month Seasonality",
            family="existing_production",
            hypothesis="Calendar-window returns around month transitions remain statistically distinct in some assets.",
            source_keys=("turn_of_month_lakonishok_1988",),
            required_columns=base_required,
            runner=existing_turn_of_month_seasonality,
            null_model_agent_name="turn_of_month_seasonality",
        ),
        CandidateSpec(
            candidate_id="existing_donchian_trend_reentry",
            display_name="Existing Strategy: Donchian Trend Reentry",
            family="existing_production",
            hypothesis="Donchian breakout re-entry captures continuation bursts after consolidation.",
            source_keys=("donchian_channel_breakout", "trend_century_hurst_2017"),
            required_columns=base_required,
            runner=existing_donchian_trend_reentry,
            null_model_agent_name="donchian_trend_reentry",
        ),
        CandidateSpec(
            candidate_id="cand_cross_asset_monthly_momentum",
            display_name="Cross-Asset Monthly Momentum Rotation",
            family="relative_strength_momentum",
            hypothesis="Monthly cross-asset momentum rotation can reduce noise versus daily switching.",
            source_keys=("momentum_jegadeesh_titman_1993", "tsmom_moskowitz_2012"),
            required_columns=base_required,
            runner=cross_asset_monthly_momentum,
            null_model_agent_name="momentum_relative_strength",
        ),
        CandidateSpec(
            candidate_id="cand_cross_asset_monthly_momentum_nostop",
            display_name="Cross-Asset Monthly Momentum (No Trailing Stop)",
            family="relative_strength_momentum",
            hypothesis="Cross-asset momentum may benefit from lower stop-induced turnover.",
            source_keys=("momentum_jegadeesh_titman_1993", "tsmom_moskowitz_2012"),
            required_columns=base_required,
            runner=cross_asset_monthly_momentum_nostop,
            null_model_agent_name="momentum_relative_strength",
        ),
        CandidateSpec(
            candidate_id="cand_cross_asset_monthly_momentum_63",
            display_name="Cross-Asset Monthly Momentum (63/126)",
            family="relative_strength_momentum",
            hypothesis="Shorter-horizon cross-asset momentum may adapt faster across mixed asset classes.",
            source_keys=("momentum_jegadeesh_titman_1993", "tsmom_moskowitz_2012"),
            required_columns=base_required,
            runner=cross_asset_monthly_momentum_63,
            null_model_agent_name="momentum_relative_strength",
        ),
        CandidateSpec(
            candidate_id="cand_tsmom_12m_filter",
            display_name="TSMOM 12M + Trend Filter",
            family="time_series_momentum",
            hypothesis="12-month absolute momentum with trend filter captures persistent drift.",
            source_keys=("tsmom_moskowitz_2012", "trend_century_hurst_2017"),
            required_columns=base_required,
            runner=tsmom_12m_filter,
        ),
        CandidateSpec(
            candidate_id="cand_ma_cross_50_200",
            display_name="50/200 Moving-Average Crossover",
            family="trend_following",
            hypothesis="Classic moving-average crossover captures medium-term trend persistence.",
            source_keys=("technical_rules_brock_1992", "data_snooping_sullivan_1999"),
            required_columns=base_required,
            runner=ma_cross_50_200,
        ),
        CandidateSpec(
            candidate_id="cand_faber_10m_timing",
            display_name="Faber 10-Month Timing Rule",
            family="trend_following",
            hypothesis="Long-only tactical timing above long moving average captures broad trend risk premium.",
            source_keys=("faber_tactical_2007", "trend_century_hurst_2017"),
            required_columns=base_required,
            runner=faber_10m_timing,
        ),
        CandidateSpec(
            candidate_id="cand_medium_term_momentum_63",
            display_name="Medium-Term Momentum (63-Day)",
            family="momentum",
            hypothesis="Intermediate momentum with trend filter can improve trade frequency without pure noise.",
            source_keys=("momentum_jegadeesh_titman_1993", "tsmom_moskowitz_2012"),
            required_columns=base_required,
            runner=medium_term_momentum_63,
        ),
        CandidateSpec(
            candidate_id="cand_bb_squeeze_breakout",
            display_name="Bollinger Squeeze Breakout",
            family="breakout",
            hypothesis="Low-volatility squeeze followed by breakout may identify expansion regimes.",
            source_keys=("technical_rules_brock_1992",),
            required_columns=base_required,
            runner=bb_squeeze_breakout,
        ),
        CandidateSpec(
            candidate_id="cand_dual_horizon_trend",
            display_name="Dual-Horizon Trend Stack",
            family="trend_following",
            hypothesis="Stacked trend filters across short/medium/long windows can reduce whipsaw.",
            source_keys=("trend_century_hurst_2017", "technical_rules_brock_1992"),
            required_columns=base_required,
            runner=dual_horizon_trend,
        ),
        CandidateSpec(
            candidate_id="cand_donchian_55_breakout",
            display_name="Donchian 55-Day Breakout",
            family="breakout",
            hypothesis="Intermediate breakouts with non-overlapping risk exits exploit trend continuation.",
            source_keys=("technical_rules_brock_1992", "trend_century_hurst_2017"),
            required_columns=base_required,
            runner=donchian_55_breakout,
        ),
        CandidateSpec(
            candidate_id="cand_breakout_20_volume",
            display_name="20-Day Breakout + Volume Confirmation",
            family="breakout",
            hypothesis="Price breakouts with participation filters reduce false positives.",
            source_keys=("technical_rules_brock_1992", "data_snooping_sullivan_1999"),
            required_columns=base_required,
            runner=breakout_20_volume,
        ),
        CandidateSpec(
            candidate_id="cand_trend_pullback_ema_rsi",
            display_name="Trend Pullback (EMA + RSI)",
            family="pullback",
            hypothesis="Pullbacks in strong uptrends can improve entry efficiency versus chasing highs.",
            source_keys=("trend_century_hurst_2017", "technical_rules_brock_1992"),
            required_columns=base_required,
            runner=trend_pullback_ema_rsi,
        ),
        CandidateSpec(
            candidate_id="cand_trend_pullback_adx",
            display_name="Trend Pullback (ADX Confirmed)",
            family="pullback",
            hypothesis="Trend-strength filtering can remove low-quality pullbacks in weak trends.",
            source_keys=("technical_rules_brock_1992",),
            required_columns=base_required,
            runner=trend_pullback_adx,
        ),
        CandidateSpec(
            candidate_id="cand_mean_reversion_bbands",
            display_name="Bollinger Mean Reversion",
            family="mean_reversion",
            hypothesis="Short-horizon reversion is more likely in low-trend, low-stress conditions.",
            source_keys=("data_snooping_sullivan_1999",),
            required_columns=base_required,
            runner=mean_reversion_bbands,
        ),
        CandidateSpec(
            candidate_id="cand_mean_reversion_zscore",
            display_name="Z-Score Mean Reversion (Calm Regime)",
            family="mean_reversion",
            hypothesis="Extremes in z-score with calm-volatility constraints can revert to mean levels.",
            source_keys=("data_snooping_sullivan_1999",),
            required_columns=base_required,
            runner=mean_reversion_zscore,
        ),
        CandidateSpec(
            candidate_id="cand_dual_momentum_trend_filter",
            display_name="Dual Momentum + Trend Filter",
            family="momentum",
            hypothesis="Combining medium and long absolute momentum with trend filter improves robustness.",
            source_keys=("momentum_jegadeesh_titman_1993", "tsmom_moskowitz_2012"),
            required_columns=base_required,
            runner=dual_momentum_trend_filter,
        ),
        CandidateSpec(
            candidate_id="cand_vol_managed_momentum",
            display_name="Volatility-Managed Momentum",
            family="volatility_scaled_momentum",
            hypothesis="Scaling position size by volatility can improve risk-adjusted momentum performance.",
            source_keys=("vol_managed_moreira_muir_2016", "tsmom_moskowitz_2012"),
            required_columns=base_required,
            runner=vol_managed_momentum,
        ),
        CandidateSpec(
            candidate_id="cand_hybrid_trend_breakout_pullback",
            display_name="Hybrid Trend + Breakout/Pullback",
            family="hybrid",
            hypothesis="Trend filter with dual entry archetypes can capture continuation in different trend phases.",
            source_keys=("trend_century_hurst_2017", "technical_rules_brock_1992"),
            required_columns=base_required,
            runner=hybrid_trend_breakout_pullback,
        ),
        CandidateSpec(
            candidate_id="cand_regime_filtered_momentum",
            display_name="Regime-Filtered Momentum",
            family="hybrid",
            hypothesis="Momentum signals may degrade in stressed regimes; regime filters can improve quality.",
            source_keys=("tsmom_moskowitz_2012",),
            required_columns=base_required,
            runner=regime_filtered_momentum,
        ),
        CandidateSpec(
            candidate_id="control_existing_validation",
            display_name="Control: Existing Validation Strategy",
            family="control",
            hypothesis="Reference current project validation strategy under identical split protocol.",
            source_keys=tuple(),
            required_columns=base_required,
            runner=control_existing_validation,
            is_control=True,
        ),
        CandidateSpec(
            candidate_id="control_random_baseline",
            display_name="Control: Random Baseline",
            family="control",
            hypothesis="Sanity-check that random timing mostly remains null-like.",
            source_keys=tuple(),
            required_columns=base_required,
            runner=control_random,
            is_control=True,
        ),
    ]
    return candidates


def split_market_dataframe(market_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build non-overlapping train/validation/holdout slices."""
    ordered = market_df.sort_values("Date").reset_index(drop=True)
    row_count = len(ordered)
    if row_count < 800:
        raise ValueError(
            f"Need at least 800 rows for honest three-way splitting; only found {row_count}."
        )
    train_end = int(math.floor(row_count * TRAIN_FRACTION))
    validation_end = int(math.floor(row_count * (TRAIN_FRACTION + VALIDATION_FRACTION)))
    train_df = ordered.iloc[:train_end].reset_index(drop=True)
    validation_df = ordered.iloc[train_end:validation_end].reset_index(drop=True)
    holdout_df = ordered.iloc[validation_end:].reset_index(drop=True)
    if min(len(train_df), len(validation_df), len(holdout_df)) < 200:
        raise ValueError(
            "One or more split windows are too small for stable inference. "
            f"Rows: train={len(train_df)}, validation={len(validation_df)}, holdout={len(holdout_df)}."
        )
    return {
        "train": train_df,
        "validation": validation_df,
        "holdout": holdout_df,
    }


def ensure_relative_strength_universe_metadata(current_ticker: str) -> None:
    """Create the universe metadata file required by relative-strength Monte Carlo."""
    metadata_path = DATA_CLEAN_DIR / f"{current_ticker.upper()}_momentum_relative_strength_universe.csv"
    required_columns = {
        "universe_position",
        "universe_ticker",
        "included_in_aligned_universe",
    }
    if metadata_path.exists():
        try:
            existing_df = pd.read_csv(metadata_path)
            if required_columns.issubset(existing_df.columns):
                included = (
                    existing_df["included_in_aligned_universe"]
                    .astype(str)
                    .str.lower()
                    .map({"true": True, "false": False})
                    .fillna(False)
                )
                if int(included.sum()) >= 2:
                    return
        except Exception:
            pass

    universe = resolve_universe(current_ticker)
    _, aligned_asset_frames = load_aligned_universe_data(universe)
    aligned_symbols = set(aligned_asset_frames.keys())
    rows: list[dict[str, object]] = []
    for position, symbol in enumerate(universe, start=1):
        rows.append(
            {
                "anchor_ticker": current_ticker.upper(),
                "universe_position": int(position),
                "universe_ticker": str(symbol).strip().upper(),
                "included_in_aligned_universe": bool(str(symbol).strip().upper() in aligned_symbols),
                "aligned_rows": int(len(aligned_asset_frames.get(symbol, pd.DataFrame()))),
            }
        )
    metadata_df = pd.DataFrame(rows)
    DATA_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(metadata_path, index=False)


def classify_row_from_metrics(
    trade_count: int,
    rcsi_z: float,
    p_value: float,
    percentile: float,
) -> str:
    """Return the raw classification bucket for one evaluation row."""
    if trade_count <= 0:
        return "no_trades"
    return classify_metrics(
        rcsi_z=rcsi_z,
        p_value=p_value,
        percentile=percentile,
    )


def evaluate_candidate_on_split(
    candidate: CandidateSpec,
    ticker: str,
    split_name: str,
    split_df: pd.DataFrame,
    *,
    simulation_count: int,
    seed: int,
) -> dict[str, object]:
    """Run one candidate on one ticker split and compute full inferential metrics."""
    trades_df = candidate.runner(split_df.copy(), ticker)
    trade_count = int(len(trades_df))
    signal_count = int(trades_df.attrs.get("signal_count", 0))
    rejected_signal_count = int(trades_df.attrs.get("rejected_signal_count", 0))

    if trade_count == 0:
        return {
            "candidate_id": candidate.candidate_id,
            "candidate_name": candidate.display_name,
            "candidate_family": candidate.family,
            "is_control": bool(candidate.is_control),
            "ticker": ticker,
            "split": split_name,
            "split_start": pd.Timestamp(split_df["Date"].iloc[0]),
            "split_end": pd.Timestamp(split_df["Date"].iloc[-1]),
            "split_rows": int(len(split_df)),
            "signal_count": signal_count,
            "rejected_signal_count": rejected_signal_count,
            "trade_count": 0,
            "average_holding_bars": np.nan,
            "actual_cumulative_return": 0.0,
            "mean_simulated_return": 0.0,
            "median_simulated_return": 0.0,
            "std_simulated_return": 0.0,
            "RCSI": 0.0,
            "RCSI_z": 0.0,
            "p_value": 1.0,
            "percentile": 100.0,
            "classification_raw": "no_trades",
            "classification_raw_label": evidence_label("no_trades"),
            "simulation_count": int(simulation_count),
            "research_grade": bool(simulation_count >= MIN_RESEARCH_GRADE_SIMULATIONS),
            "null_model": "structure_preserving_random_timing",
            "seed": int(seed),
        }

    raw_returns = trades_df["return"].to_numpy(dtype=float)
    adjusted_returns = adjust_trade_returns(
        raw_returns=raw_returns,
        transaction_cost=TRANSACTION_COST,
        input_path=Path(f"{ticker}_{candidate.candidate_id}_{split_name}_trades.csv"),
    )
    actual_cumulative_return = float(np.expm1(np.log1p(adjusted_returns).sum()))

    rng = build_random_generator(reproducible=True, seed=seed)
    null_agent_name = candidate.null_model_agent_name or candidate.candidate_id
    trade_df_for_null = trades_df.copy()
    if null_agent_name == "momentum_relative_strength":
        ensure_relative_strength_universe_metadata(ticker)
        # Split-window alignment can shorten the effective peer calendar and
        # make previously recorded holding_bars differ by one bar. The null
        # model derives durations directly from entry/exit timestamps, so we
        # avoid false failures by disabling the redundant strict equality check.
        if "holding_bars" in trade_df_for_null.columns:
            trade_df_for_null["holding_bars"] = np.nan
    simulated_returns, null_model_name = simulate_agent_null_cumulative_returns(
        agent_name=null_agent_name,
        current_ticker=ticker,
        trade_df=trade_df_for_null,
        market_df=split_df,
        input_path=Path(f"{ticker}_{candidate.candidate_id}_{split_name}_trades.csv"),
        simulation_count=simulation_count,
        rng=rng,
    )
    simulated_array = simulated_returns.to_numpy(dtype=float)
    mean_simulated_return = float(np.mean(simulated_array))
    median_simulated_return = float(np.median(simulated_array))
    std_simulated_return = float(np.std(simulated_array, ddof=0))
    rcsi_value = float(actual_cumulative_return - mean_simulated_return)
    if std_simulated_return > 0:
        rcsi_z_value = float(rcsi_value / std_simulated_return)
    else:
        rcsi_z_value = 0.0 if np.isclose(rcsi_value, 0.0) else np.nan
    p_value = float(calculate_p_value(simulated_array, actual_cumulative_return))
    percentile = float(calculate_actual_percentile(simulated_array, actual_cumulative_return))
    classification_raw = classify_row_from_metrics(
        trade_count=trade_count,
        rcsi_z=rcsi_z_value,
        p_value=p_value,
        percentile=percentile,
    )
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_name": candidate.display_name,
        "candidate_family": candidate.family,
        "is_control": bool(candidate.is_control),
        "ticker": ticker,
        "split": split_name,
        "split_start": pd.Timestamp(split_df["Date"].iloc[0]),
        "split_end": pd.Timestamp(split_df["Date"].iloc[-1]),
        "split_rows": int(len(split_df)),
        "signal_count": signal_count,
        "rejected_signal_count": rejected_signal_count,
        "trade_count": trade_count,
        "average_holding_bars": float(pd.to_numeric(trades_df["holding_bars"], errors="coerce").mean()),
        "actual_cumulative_return": actual_cumulative_return,
        "mean_simulated_return": mean_simulated_return,
        "median_simulated_return": median_simulated_return,
        "std_simulated_return": std_simulated_return,
        "RCSI": rcsi_value,
        "RCSI_z": rcsi_z_value,
        "p_value": p_value,
        "percentile": percentile,
        "classification_raw": classification_raw,
        "classification_raw_label": evidence_label(classification_raw),
        "simulation_count": int(simulation_count),
        "research_grade": bool(simulation_count >= MIN_RESEARCH_GRADE_SIMULATIONS),
        "null_model": str(null_model_name),
        "seed": int(seed),
    }


def apply_bh_by_group(
    input_df: pd.DataFrame,
    *,
    group_columns: list[str],
    adjusted_column_name: str,
) -> pd.DataFrame:
    """Apply Benjamini-Hochberg correction inside each requested group."""
    output_df = input_df.copy()
    output_df[adjusted_column_name] = np.nan
    for _, group_df in output_df.groupby(group_columns):
        adjusted = benjamini_hochberg_adjusted_p_values(group_df["p_value"])
        output_df.loc[group_df.index, adjusted_column_name] = adjusted
    return output_df


def add_adjusted_classifications(input_df: pd.DataFrame, *, adjusted_p_column: str, class_column: str) -> pd.DataFrame:
    """Apply multiple-testing guard to each raw classification."""
    output_df = input_df.copy()
    adjusted_classes: list[str] = []
    adjusted_labels: list[str] = []
    for _, row in output_df.iterrows():
        guarded_bucket = apply_multiple_testing_guard(
            str(row["classification_raw"]),
            float(row[adjusted_p_column]) if pd.notna(row[adjusted_p_column]) else np.nan,
        )
        adjusted_classes.append(guarded_bucket)
        adjusted_labels.append(evidence_label(guarded_bucket))
    output_df[class_column] = adjusted_classes
    output_df[f"{class_column}_label"] = adjusted_labels
    return output_df


def select_finalists(candidate_results: pd.DataFrame) -> pd.DataFrame:
    """Select holdout finalists using fixed train+validation gates."""
    discovery_df = candidate_results[
        (~candidate_results["is_control"])
        & (candidate_results["split"].isin(["train", "validation"]))
    ].copy()
    summary_rows: list[dict[str, object]] = []
    for candidate_id, group_df in discovery_df.groupby("candidate_id"):
        train_df = group_df[group_df["split"] == "train"]
        validation_df = group_df[group_df["split"] == "validation"]
        if train_df.empty or validation_df.empty:
            continue

        def positive_share(df_slice: pd.DataFrame) -> float:
            inference_df = df_slice[df_slice["trade_count"] >= MIN_TRADES_FOR_INFERENCE]
            if inference_df.empty:
                return 0.0
            positives = (
                (pd.to_numeric(inference_df["RCSI_z"], errors="coerce") >= 0.5)
                & (pd.to_numeric(inference_df["p_value"], errors="coerce") <= 0.20)
                & (pd.to_numeric(inference_df["percentile"], errors="coerce") >= 70.0)
            )
            return float(positives.mean())

        train_median_trade_count = float(pd.to_numeric(train_df["trade_count"], errors="coerce").median())
        validation_median_trade_count = float(pd.to_numeric(validation_df["trade_count"], errors="coerce").median())
        train_positive_share = positive_share(train_df)
        validation_positive_share = positive_share(validation_df)
        train_median_z = float(pd.to_numeric(train_df["RCSI_z"], errors="coerce").median())
        validation_median_z = float(pd.to_numeric(validation_df["RCSI_z"], errors="coerce").median())
        train_median_p = float(pd.to_numeric(train_df["p_value"], errors="coerce").median())
        validation_median_p = float(pd.to_numeric(validation_df["p_value"], errors="coerce").median())

        train_gate = (
            train_median_trade_count >= float(MIN_TRADES_FOR_INFERENCE)
            and train_positive_share >= 0.40
            and train_median_z >= 0.20
            and train_median_p <= 0.35
        )
        validation_gate = (
            validation_median_trade_count >= float(MIN_TRADES_FOR_INFERENCE)
            and validation_positive_share >= 0.40
            and validation_median_z >= 0.20
            and validation_median_p <= 0.35
        )
        summary_rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_name": str(group_df["candidate_name"].iloc[0]),
                "candidate_family": str(group_df["candidate_family"].iloc[0]),
                "train_median_trade_count": train_median_trade_count,
                "validation_median_trade_count": validation_median_trade_count,
                "train_positive_share": train_positive_share,
                "validation_positive_share": validation_positive_share,
                "train_median_RCSI_z": train_median_z,
                "validation_median_RCSI_z": validation_median_z,
                "train_median_p_value": train_median_p,
                "validation_median_p_value": validation_median_p,
                "passed_train_gate": bool(train_gate),
                "passed_validation_gate": bool(validation_gate),
                "selected_for_holdout": bool(train_gate and validation_gate),
            }
        )

    selection_df = pd.DataFrame(summary_rows).sort_values(
        ["selected_for_holdout", "validation_median_RCSI_z", "validation_positive_share"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return selection_df


def run_seed_robustness(
    finalists_df: pd.DataFrame,
    splits_by_ticker: dict[str, dict[str, pd.DataFrame]],
    candidates_by_id: dict[str, CandidateSpec],
) -> pd.DataFrame:
    """Re-evaluate holdout finalists across multiple seeds for stability diagnostics."""
    robustness_rows: list[dict[str, object]] = []
    selected_ids = (
        finalists_df["candidate_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    for candidate_id in selected_ids:
        candidate = candidates_by_id[candidate_id]
        for ticker in DISCOVERY_TICKERS:
            holdout_df = splits_by_ticker[ticker]["holdout"]
            seed_rows: list[dict[str, object]] = []
            for seed_offset in range(ROBUSTNESS_SEEDS):
                current_seed = DISCOVERY_SEED + 1000 + seed_offset
                evaluated = evaluate_candidate_on_split(
                    candidate=candidate,
                    ticker=ticker,
                    split_name="holdout",
                    split_df=holdout_df,
                    simulation_count=ROBUSTNESS_SIMULATIONS,
                    seed=current_seed,
                )
                seed_rows.append(evaluated)
            seed_df = pd.DataFrame(seed_rows)
            robustness_rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_name": candidate.display_name,
                    "candidate_family": candidate.family,
                    "ticker": ticker,
                    "seed_runs": int(len(seed_df)),
                    "trade_count_median": float(pd.to_numeric(seed_df["trade_count"], errors="coerce").median()),
                    "RCSI_z_mean": float(pd.to_numeric(seed_df["RCSI_z"], errors="coerce").mean()),
                    "RCSI_z_std": float(pd.to_numeric(seed_df["RCSI_z"], errors="coerce").std(ddof=0)),
                    "p_value_mean": float(pd.to_numeric(seed_df["p_value"], errors="coerce").mean()),
                    "p_value_std": float(pd.to_numeric(seed_df["p_value"], errors="coerce").std(ddof=0)),
                    "percentile_mean": float(pd.to_numeric(seed_df["percentile"], errors="coerce").mean()),
                    "share_p_le_0_20": float(
                        (pd.to_numeric(seed_df["p_value"], errors="coerce") <= 0.20).mean()
                    ),
                    "share_rcsi_z_ge_0_5": float(
                        (pd.to_numeric(seed_df["RCSI_z"], errors="coerce") >= 0.50).mean()
                    ),
                }
            )
    if not robustness_rows:
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "candidate_name",
                "candidate_family",
                "ticker",
                "seed_runs",
                "trade_count_median",
                "RCSI_z_mean",
                "RCSI_z_std",
                "p_value_mean",
                "p_value_std",
                "percentile_mean",
                "share_p_le_0_20",
                "share_rcsi_z_ge_0_5",
            ]
        )
    return pd.DataFrame(robustness_rows).sort_values(
        ["candidate_id", "ticker"]
    ).reset_index(drop=True)


def save_sources_table(candidates: list[CandidateSpec]) -> pd.DataFrame:
    """Save the source mapping used to design the candidate families."""
    rows: list[dict[str, str]] = []
    for candidate in candidates:
        if not candidate.source_keys:
            continue
        for source_key in candidate.source_keys:
            source_info = SOURCE_LIBRARY.get(source_key, {})
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "candidate_name": candidate.display_name,
                    "source_key": source_key,
                    "citation": source_info.get("citation", ""),
                    "url": source_info.get("url", ""),
                }
            )
    sources_df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    write_dataframe_artifact(
        sources_df,
        OUTPUT_SOURCES_PATH,
        producer="strategy_discovery.save_sources_table",
        current_ticker="MULTI",
        dependencies=[],
        research_grade=True,
        canonical_policy="always",
        parameters={
            "artifact_type": "strategy_discovery_sources",
        },
    )
    return sources_df


def main() -> None:
    """Run discovery, selection, holdout confirmation, and robustness diagnostics."""
    print("Starting research-backed strategy discovery...")
    print(f"Tickers: {', '.join(DISCOVERY_TICKERS)}")
    print(
        "Simulation depth: "
        f"discovery={DISCOVERY_MONTE_CARLO_SIMULATIONS}, holdout={HOLDOUT_MONTE_CARLO_SIMULATIONS}, "
        f"robustness={ROBUSTNESS_SIMULATIONS} x {ROBUSTNESS_SEEDS} seeds"
    )

    candidates = _build_candidates()
    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    save_sources_table(candidates)

    splits_by_ticker: dict[str, dict[str, pd.DataFrame]] = {}
    for ticker in DISCOVERY_TICKERS:
        market_df = build_regime_dataframe_for_ticker(ticker, save_output=True)
        market_df = augment_candidate_features(market_df)
        market_df.attrs["ticker"] = ticker
        splits_by_ticker[ticker] = split_market_dataframe(market_df)
        train_rows = len(splits_by_ticker[ticker]["train"])
        validation_rows = len(splits_by_ticker[ticker]["validation"])
        holdout_rows = len(splits_by_ticker[ticker]["holdout"])
        print(
            f"[{ticker}] split sizes -> train={train_rows}, validation={validation_rows}, holdout={holdout_rows}"
        )

    candidate_rows: list[dict[str, object]] = []
    total_jobs = len(candidates) * len(DISCOVERY_TICKERS) * 3
    completed_jobs = 0
    for candidate in candidates:
        for ticker in DISCOVERY_TICKERS:
            for split_name in ("train", "validation", "holdout"):
                split_df = splits_by_ticker[ticker][split_name]
                simulation_count = (
                    HOLDOUT_MONTE_CARLO_SIMULATIONS
                    if split_name == "holdout"
                    else DISCOVERY_MONTE_CARLO_SIMULATIONS
                )
                seed_component = sum((index + 1) * ord(ch) for index, ch in enumerate(f"{candidate.candidate_id}|{ticker}|{split_name}"))
                job_seed = DISCOVERY_SEED + seed_component
                row = evaluate_candidate_on_split(
                    candidate=candidate,
                    ticker=ticker,
                    split_name=split_name,
                    split_df=split_df,
                    simulation_count=simulation_count,
                    seed=job_seed,
                )
                candidate_rows.append(row)
                completed_jobs += 1
                if completed_jobs % 10 == 0 or completed_jobs == total_jobs:
                    print(f"Completed {completed_jobs}/{total_jobs} candidate evaluations...")

    candidate_results_df = pd.DataFrame(candidate_rows)
    candidate_results_df = apply_bh_by_group(
        candidate_results_df,
        group_columns=["ticker", "split"],
        adjusted_column_name="bh_adjusted_p_value_by_ticker_split",
    )
    candidate_results_df = apply_bh_by_group(
        candidate_results_df,
        group_columns=["split"],
        adjusted_column_name="bh_adjusted_p_value_by_split_global",
    )
    candidate_results_df = add_adjusted_classifications(
        candidate_results_df,
        adjusted_p_column="bh_adjusted_p_value_by_ticker_split",
        class_column="classification_bh_by_ticker_split",
    )
    candidate_results_df = add_adjusted_classifications(
        candidate_results_df,
        adjusted_p_column="bh_adjusted_p_value_by_split_global",
        class_column="classification_bh_by_split_global",
    )
    candidate_results_df = candidate_results_df.sort_values(
        ["split", "ticker", "candidate_family", "candidate_id"]
    ).reset_index(drop=True)

    write_dataframe_artifact(
        candidate_results_df,
        OUTPUT_CANDIDATE_RESULTS_PATH,
        producer="strategy_discovery.main",
        current_ticker="MULTI",
        dependencies=[],
        research_grade=True,
        canonical_policy="always",
        parameters={
            "artifact_type": "strategy_discovery_candidate_results",
            "tickers": DISCOVERY_TICKERS,
            "discovery_simulations": DISCOVERY_MONTE_CARLO_SIMULATIONS,
            "holdout_simulations": HOLDOUT_MONTE_CARLO_SIMULATIONS,
            "min_trades_for_inference": MIN_TRADES_FOR_INFERENCE,
        },
    )

    selection_df = select_finalists(candidate_results_df)
    write_dataframe_artifact(
        selection_df,
        OUTPUT_SELECTION_PATH,
        producer="strategy_discovery.main",
        current_ticker="MULTI",
        dependencies=[OUTPUT_CANDIDATE_RESULTS_PATH],
        research_grade=True,
        canonical_policy="always",
        parameters={
            "artifact_type": "strategy_discovery_selection_summary",
            "selection_rule": (
                "train+validation gates: median trades >= min_trades, positive share >= 0.40, "
                "median RCSI_z >= 0.20, median p <= 0.35"
            ),
        },
    )

    finalists = selection_df[selection_df["selected_for_holdout"]].copy()
    if finalists.empty:
        print("No candidate passed strict train+validation gates. Selecting top 3 by validation median RCSI_z for diagnostic holdout.")
        finalists = selection_df.head(3).copy()
        finalists["selected_for_holdout"] = False
        finalists["fallback_diagnostic_holdout"] = True
    else:
        finalists["fallback_diagnostic_holdout"] = False

    holdout_rows: list[dict[str, object]] = []
    for _, finalist_row in finalists.iterrows():
        candidate_id = str(finalist_row["candidate_id"])
        candidate = candidates_by_id[candidate_id]
        for ticker in DISCOVERY_TICKERS:
            holdout_df = splits_by_ticker[ticker]["holdout"]
            seed_component = sum((index + 1) * ord(ch) for index, ch in enumerate(f"holdout_deep|{candidate_id}|{ticker}"))
            holdout_seed = DISCOVERY_SEED + 500000 + seed_component
            evaluated = evaluate_candidate_on_split(
                candidate=candidate,
                ticker=ticker,
                split_name="holdout",
                split_df=holdout_df,
                simulation_count=HOLDOUT_MONTE_CARLO_SIMULATIONS,
                seed=holdout_seed,
            )
            evaluated["selected_for_holdout"] = bool(finalist_row.get("selected_for_holdout", False))
            evaluated["fallback_diagnostic_holdout"] = bool(finalist_row.get("fallback_diagnostic_holdout", False))
            holdout_rows.append(evaluated)

    holdout_df = pd.DataFrame(holdout_rows)
    if not holdout_df.empty:
        holdout_df = apply_bh_by_group(
            holdout_df,
            group_columns=["ticker"],
            adjusted_column_name="holdout_bh_adjusted_p_by_ticker",
        )
        holdout_df = apply_bh_by_group(
            holdout_df,
            group_columns=["split"],
            adjusted_column_name="holdout_bh_adjusted_p_global",
        )
        holdout_df = add_adjusted_classifications(
            holdout_df,
            adjusted_p_column="holdout_bh_adjusted_p_by_ticker",
            class_column="holdout_classification_bh_by_ticker",
        )
        holdout_df = add_adjusted_classifications(
            holdout_df,
            adjusted_p_column="holdout_bh_adjusted_p_global",
            class_column="holdout_classification_bh_global",
        )
        holdout_df = holdout_df.sort_values(["candidate_id", "ticker"]).reset_index(drop=True)

    write_dataframe_artifact(
        holdout_df,
        OUTPUT_HOLDOUT_PATH,
        producer="strategy_discovery.main",
        current_ticker="MULTI",
        dependencies=[OUTPUT_CANDIDATE_RESULTS_PATH, OUTPUT_SELECTION_PATH],
        research_grade=True,
        canonical_policy="always",
        parameters={
            "artifact_type": "strategy_discovery_holdout_deep_results",
            "holdout_simulations": HOLDOUT_MONTE_CARLO_SIMULATIONS,
        },
    )

    robustness_df = run_seed_robustness(
        finalists_df=finalists,
        splits_by_ticker=splits_by_ticker,
        candidates_by_id=candidates_by_id,
    )
    write_dataframe_artifact(
        robustness_df,
        OUTPUT_ROBUSTNESS_PATH,
        producer="strategy_discovery.main",
        current_ticker="MULTI",
        dependencies=[OUTPUT_SELECTION_PATH, OUTPUT_HOLDOUT_PATH],
        research_grade=True,
        canonical_policy="always",
        parameters={
            "artifact_type": "strategy_discovery_seed_robustness",
            "robustness_simulations": ROBUSTNESS_SIMULATIONS,
            "robustness_seeds": ROBUSTNESS_SEEDS,
        },
    )

    print("\nStrategy discovery completed.")
    print(f"Candidate results: {OUTPUT_CANDIDATE_RESULTS_PATH}")
    print(f"Selection summary: {OUTPUT_SELECTION_PATH}")
    print(f"Holdout deep results: {OUTPUT_HOLDOUT_PATH}")
    print(f"Seed robustness: {OUTPUT_ROBUSTNESS_PATH}")
    print(f"Sources table: {OUTPUT_SOURCES_PATH}")


if __name__ == "__main__":
    main()
