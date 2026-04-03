"""Run repeated multi-asset walk-forward evaluations across rolling test windows."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from plot_config import data_clean_dir, format_agent_name

try:
    from artifact_provenance import write_dataframe_artifact
    from data_loader import main as refresh_raw_data_for_ticker
    from features import build_features_for_ticker
    from momentum_relative_strength_agent import (
        RELATIVE_STRENGTH_LOOKBACK_DAYS,
        load_aligned_universe_data,
        resolve_universe,
        run_relative_strength_on_aligned_universe,
    )
    from monte_carlo import (
        CONTEXT_MATCHING_ENABLED,
        NULL_MODEL_NAME,
        RELATIVE_STRENGTH_NULL_MODEL_NAME,
        SIMULATE_EXECUTION_COSTS,
        TRANSACTION_COST,
        NullModelInputs,
        adjust_trade_returns,
        build_context_entry_candidate_pools,
        build_trade_structure,
        calculate_actual_percentile,
        calculate_p_value,
        calculate_trade_durations,
        extract_position_value_fractions,
        simulate_structure_preserving_cumulative_returns,
    )
    from research_metrics import (
        build_daily_strategy_curve,
        calculate_p_value_prominence,
        calculate_trade_level_return_ratio,
        summarize_daily_curve,
    )
    from regimes import build_regime_dataframe_for_ticker
    from strategy_config import (
        AGENT_ORDER,
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOW_COLUMN,
        MOMENTUM_RETURN_COLUMN,
    )
    from strategy_simulator import run_strategy
    from strategy_verdicts import (
        adjust_for_evaluation_power,
        classify_robustness_evidence,
        confidence_bucket_from_score,
        confidence_label,
        evaluation_power_label,
        evaluation_power_score,
        evidence_label,
        format_verdict_label,
        verdict_from_evidence_bucket,
    )
    from timeframe_config import infer_bars_per_year, normalize_timestamp_series
except ModuleNotFoundError:
    from Code.artifact_provenance import write_dataframe_artifact
    from Code.data_loader import main as refresh_raw_data_for_ticker
    from Code.features import build_features_for_ticker
    from Code.momentum_relative_strength_agent import (
        RELATIVE_STRENGTH_LOOKBACK_DAYS,
        load_aligned_universe_data,
        resolve_universe,
        run_relative_strength_on_aligned_universe,
    )
    from Code.monte_carlo import (
        CONTEXT_MATCHING_ENABLED,
        NULL_MODEL_NAME,
        RELATIVE_STRENGTH_NULL_MODEL_NAME,
        SIMULATE_EXECUTION_COSTS,
        TRANSACTION_COST,
        NullModelInputs,
        adjust_trade_returns,
        build_context_entry_candidate_pools,
        build_trade_structure,
        calculate_actual_percentile,
        calculate_p_value,
        calculate_trade_durations,
        extract_position_value_fractions,
        simulate_structure_preserving_cumulative_returns,
    )
    from Code.research_metrics import (
        build_daily_strategy_curve,
        calculate_p_value_prominence,
        calculate_trade_level_return_ratio,
        summarize_daily_curve,
    )
    from Code.regimes import build_regime_dataframe_for_ticker
    from Code.strategy_config import (
        AGENT_ORDER,
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOW_COLUMN,
        MOMENTUM_RETURN_COLUMN,
    )
    from Code.strategy_simulator import run_strategy
    from Code.strategy_verdicts import (
        adjust_for_evaluation_power,
        classify_robustness_evidence,
        confidence_bucket_from_score,
        confidence_label,
        evaluation_power_label,
        evaluation_power_score,
        evidence_label,
        format_verdict_label,
        verdict_from_evidence_bucket,
    )
    from Code.timeframe_config import infer_bars_per_year, normalize_timestamp_series


DEFAULT_TICKERS = [
    ticker.strip().upper()
    for ticker in os.environ.get(
        "WALK_FORWARD_TICKERS",
        "SPY,QQQ,AAPL,VOO,NVDA,TSM,MRNA,NVAX,BTC-USD,EURUSD=X,JPYUSD=X,NQ=F,ES=F",
    ).split(",")
    if ticker.strip()
]
TICKERS = DEFAULT_TICKERS


def _env_int(*names: str, default: int) -> int:
    """Read the first configured integer environment variable."""
    for name in names:
        raw_value = os.environ.get(name, "").strip()
        if raw_value:
            return int(raw_value)
    return int(default)


def _env_flag(*names: str, default: bool) -> bool:
    """Read the first configured boolean-like environment variable."""
    for name in names:
        raw_value = os.environ.get(name, "").strip().lower()
        if raw_value:
            return raw_value in {"1", "true", "yes", "y", "on"}
    return bool(default)


def configured_min_trades_per_panel() -> int:
    """Return the panel-level minimum trade count, with a forex-friendly fallback."""
    explicit_value = os.environ.get("WALK_FORWARD_MIN_TRADES_PER_PANEL", "").strip()
    if explicit_value:
        return max(int(explicit_value), 1)

    legacy_explicit_value = os.environ.get("WALK_FORWARD_MIN_TRADES", "").strip()
    if legacy_explicit_value:
        return max(int(legacy_explicit_value), 1)

    if any(str(ticker).strip().upper().endswith("=X") for ticker in TICKERS):
        return 1

    return 5


TEST_BARS = _env_int("WALK_FORWARD_TEST_BARS", "TEST_BARS", default=504)
STEP_BARS = _env_int("WALK_FORWARD_STEP_BARS", "STEP_BARS", default=252)
MIN_TRADES_PER_PANEL = configured_min_trades_per_panel()
SIMULATIONS_PER_RUN = _env_int(
    "WALK_FORWARD_SIMULATIONS_PER_RUN",
    "SIMULATIONS_PER_RUN",
    default=1000,
)
OUTER_RUNS = _env_int("WALK_FORWARD_OUTER_RUNS", "OUTER_RUNS", default=50)
BASE_SEED = _env_int("WALK_FORWARD_BASE_SEED", "BASE_SEED", default=5000)
PROGRESS_EVERY = max(
    1,
    _env_int("WALK_FORWARD_PROGRESS_EVERY", "PROGRESS_EVERY", default=1),
)
REFRESH_DATA = _env_flag("WALK_FORWARD_REFRESH_DATA", "REFRESH_DATA", default=True)
RCSI_Z_STABILITY_FLOOR = 0.10


class FoldStrategyResult(dict):
    """Typed dict-like holder for one fold strategy evaluation."""


def required_market_columns() -> list[str]:
    """Return the minimum columns the walk-forward strategies require."""
    return [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "avg_volume_20",
        "regime",
        "sma_50",
        MOMENTUM_RETURN_COLUMN,
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOW_COLUMN,
    ]


def ensure_ticker_inputs(current_ticker: str) -> None:
    """Refresh or build the required saved inputs for one walk-forward ticker."""
    regime_path = data_clean_dir() / f"{current_ticker}_regimes.csv"
    if REFRESH_DATA or not regime_path.exists():
        refresh_raw_data_for_ticker(current_ticker)
        build_features_for_ticker(current_ticker, save_output=True)
        build_regime_dataframe_for_ticker(current_ticker, save_output=True)
        return

    build_regime_dataframe_for_ticker(current_ticker, save_output=True)


def load_fold_market_data(current_ticker: str) -> pd.DataFrame:
    """Load the saved market dataframe used to create walk-forward folds."""
    market_df = build_regime_dataframe_for_ticker(current_ticker, save_output=True).copy()
    missing_columns = [
        column for column in required_market_columns() if column not in market_df.columns
    ]
    if missing_columns:
        raise KeyError(
            f"{current_ticker} walk-forward market data is missing required columns: "
            f"{', '.join(missing_columns)}"
        )

    market_df["Date"] = normalize_timestamp_series(market_df["Date"])
    numeric_columns = [column for column in required_market_columns() if column not in {"Date", "regime"}]
    for column in numeric_columns:
        market_df[column] = pd.to_numeric(market_df[column], errors="coerce")
    market_df = (
        market_df.dropna(subset=required_market_columns())
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )
    market_df.attrs["ticker"] = current_ticker
    return market_df


def build_folds(market_df: pd.DataFrame) -> list[dict[str, object]]:
    """Create rolling fixed-width test folds."""
    folds: list[dict[str, object]] = []
    if len(market_df) < TEST_BARS:
        return folds

    fold_number = 0
    for start_index in range(0, len(market_df) - TEST_BARS + 1, STEP_BARS):
        end_index = start_index + TEST_BARS - 1
        fold_number += 1
        fold_market_df = market_df.iloc[start_index : end_index + 1].reset_index(drop=True)
        fold_market_df.attrs["ticker"] = str(market_df.attrs.get("ticker", "")).strip().upper()
        folds.append(
            {
                "fold_id": f"fold_{fold_number:02d}",
                "start_date": pd.Timestamp(fold_market_df["Date"].iloc[0]),
                "end_date": pd.Timestamp(fold_market_df["Date"].iloc[-1]),
                "market_df": fold_market_df,
                "test_bar_count": len(fold_market_df) - 1,
            }
        )

    return folds


def calculate_fold_annualized_return(
    cumulative_return: float,
    test_bar_count: int,
    fold_dates: pd.Series,
) -> float:
    """Annualize a fold return across the full out-of-sample window length."""
    if test_bar_count <= 0:
        return 0.0
    if cumulative_return <= -1.0:
        return -1.0

    bars_per_year = infer_bars_per_year(fold_dates)
    return float((1.0 + cumulative_return) ** (bars_per_year / float(test_bar_count)) - 1.0)


def series_std_or_nan(series: pd.Series) -> float:
    """Return a population standard deviation for one numeric series."""
    numeric_values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(numeric_values) == 0:
        return np.nan
    return float(np.std(numeric_values, ddof=0))


def series_percentile_or_nan(series: pd.Series, percentile: float) -> float:
    """Return one percentile or NaN for an empty numeric slice."""
    numeric_values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(numeric_values) == 0:
        return np.nan
    return float(np.percentile(numeric_values, percentile))


def actual_random_decision_seed(ticker: str, fold_id: str) -> int:
    """Create a deterministic random-strategy seed per ticker/fold panel."""
    ticker_component = sum((index + 1) * ord(character) for index, character in enumerate(ticker))
    fold_component = sum((index + 1) * ord(character) for index, character in enumerate(fold_id))
    return BASE_SEED + (97 * ticker_component) + (13 * fold_component)


def build_single_asset_curve(
    trade_df: pd.DataFrame,
    fold_market_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the marked-to-market curve for a single-asset strategy fold."""
    return build_daily_strategy_curve(
        trade_df=trade_df,
        market_df=fold_market_df[["Date", "Close"]].copy(),
    )


def build_relative_strength_fold_result(
    ticker: str,
    fold_market_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.Timestamp], dict[str, pd.DataFrame]]:
    """Run the relative-strength strategy on one fold and return its aligned context."""
    universe = resolve_universe(ticker)
    common_dates, asset_frames = load_aligned_universe_data(universe)
    allowed_dates = set(pd.to_datetime(fold_market_df["Date"], errors="coerce").dropna().tolist())
    filtered_common_dates = [date for date in common_dates if date in allowed_dates]
    if len(filtered_common_dates) < RELATIVE_STRENGTH_LOOKBACK_DAYS + 5:
        return pd.DataFrame(), pd.DataFrame(), [], {}

    filtered_frames: dict[str, pd.DataFrame] = {}
    for asset_ticker, asset_df in asset_frames.items():
        filtered_df = asset_df.loc[asset_df["Date"].isin(filtered_common_dates)].copy()
        filtered_df = filtered_df.sort_values("Date").reset_index(drop=True)
        if len(filtered_df) != len(filtered_common_dates):
            continue
        filtered_df.attrs["ticker"] = asset_ticker
        filtered_frames[asset_ticker] = filtered_df

    if len(filtered_frames) < 2:
        return pd.DataFrame(), pd.DataFrame(), [], {}

    trades_df, curve_df = run_relative_strength_on_aligned_universe(
        filtered_common_dates,
        filtered_frames,
        anchor_ticker=ticker,
    )
    return trades_df, curve_df, filtered_common_dates, filtered_frames


def build_null_model_inputs_for_fold(
    *,
    agent_name: str,
    ticker: str,
    trade_df: pd.DataFrame,
    fold_market_df: pd.DataFrame,
    relative_strength_dates: list[pd.Timestamp] | None = None,
    relative_strength_frames: dict[str, pd.DataFrame] | None = None,
    input_path: Path | str,
) -> NullModelInputs:
    """Assemble fold-local null-model inputs so timing randomization stays in-fold."""
    if trade_df.empty:
        raise ValueError("Walk-forward null model cannot be prepared from an empty trade log.")

    if agent_name == "momentum_relative_strength":
        if not relative_strength_dates or not relative_strength_frames:
            raise ValueError(
                "Relative-strength walk-forward null requires the aligned fold universe."
            )
        aligned_universe = tuple(relative_strength_frames.keys())
        symbol_to_index = {symbol: index for index, symbol in enumerate(aligned_universe)}
        asset_indices = (
            trade_df["asset_ticker"]
            .astype(str)
            .str.strip()
            .str.upper()
            .map(symbol_to_index)
        )
        if asset_indices.isna().any():
            missing_assets = (
                trade_df.loc[asset_indices.isna(), "asset_ticker"]
                .astype(str)
                .str.strip()
                .str.upper()
                .unique()
                .tolist()
            )
            raise ValueError(
                "Relative-strength fold trades reference assets that are not in the "
                f"aligned fold universe: {missing_assets}"
            )

        open_price_matrix = np.vstack(
            [
                relative_strength_frames[symbol]["Open"].to_numpy(dtype=float)
                for symbol in aligned_universe
            ]
        )
        trade_structure = build_trade_structure(
            trade_df=trade_df,
            calendar_dates=relative_strength_dates,
            input_path=input_path,
            max_open_index=open_price_matrix.shape[1] - 1,
            asset_indices=asset_indices.to_numpy(dtype=np.int64),
        )
        candidate_pools = (
            build_context_entry_candidate_pools(
                trade_structure=trade_structure,
                open_price_matrix=open_price_matrix,
                calendar_dates=tuple(pd.to_datetime(relative_strength_dates)),
            )
            if CONTEXT_MATCHING_ENABLED
            else tuple()
        )
        return NullModelInputs(
            open_price_matrix=open_price_matrix,
            trade_structure=trade_structure,
            null_model_name=RELATIVE_STRENGTH_NULL_MODEL_NAME,
            calendar_dates=tuple(pd.to_datetime(relative_strength_dates)),
            context_entry_candidate_pools=candidate_pools,
        )

    open_price_matrix = fold_market_df["Open"].to_numpy(dtype=float).reshape(1, -1)
    trade_structure = build_trade_structure(
        trade_df=trade_df,
        calendar_dates=fold_market_df["Date"],
        input_path=input_path,
        max_open_index=open_price_matrix.shape[1] - 1,
        asset_indices=np.zeros(len(trade_df), dtype=np.int64),
    )
    candidate_pools = (
        build_context_entry_candidate_pools(
            trade_structure=trade_structure,
            open_price_matrix=open_price_matrix,
            calendar_dates=tuple(pd.to_datetime(fold_market_df["Date"])),
        )
        if CONTEXT_MATCHING_ENABLED
        else tuple()
    )
    return NullModelInputs(
        open_price_matrix=open_price_matrix,
        trade_structure=trade_structure,
        null_model_name=NULL_MODEL_NAME,
        calendar_dates=tuple(pd.to_datetime(fold_market_df["Date"])),
        context_entry_candidate_pools=candidate_pools,
    )


def build_fold_actual_metrics(
    *,
    ticker: str,
    fold_id: str,
    agent_name: str,
    fold_market_df: pd.DataFrame,
    test_bar_count: int,
) -> FoldStrategyResult | None:
    """Run one strategy on one fold and calculate the realized metrics."""
    resolved_ticker = str(ticker).strip().upper()

    if agent_name == "momentum_relative_strength":
        trade_df, curve_df, relative_strength_dates, relative_strength_frames = (
            build_relative_strength_fold_result(resolved_ticker, fold_market_df)
        )
    else:
        random_seed = (
            actual_random_decision_seed(resolved_ticker, fold_id)
            if agent_name == "random"
            else None
        )
        trade_df = run_strategy(
            agent_name,
            fold_market_df.copy(),
            random_decision_seed=random_seed,
            ticker=resolved_ticker,
        )
        curve_df = build_single_asset_curve(trade_df, fold_market_df)
        relative_strength_dates = None
        relative_strength_frames = None

    if len(trade_df) < MIN_TRADES_PER_PANEL:
        return None

    input_path = Path(f"{resolved_ticker}_{agent_name}_{fold_id}")
    raw_returns = trade_df["return"].to_numpy(dtype=float)
    adjusted_returns = adjust_trade_returns(
        raw_returns=raw_returns,
        transaction_cost=TRANSACTION_COST,
        input_path=input_path,
    )
    durations = calculate_trade_durations(
        trade_df=trade_df,
        market_df=fold_market_df[["Date", "Open"]].copy()
        if agent_name != "momentum_relative_strength"
        else pd.DataFrame({"Date": relative_strength_dates, "Open": relative_strength_frames[next(iter(relative_strength_frames))]["Open"]}),
        input_path=input_path,
    )
    position_value_fractions = extract_position_value_fractions(trade_df)
    curve_summary = summarize_daily_curve(curve_df)
    cumulative_return = float(curve_summary["cumulative_return"])
    null_model_inputs = build_null_model_inputs_for_fold(
        agent_name=agent_name,
        ticker=resolved_ticker,
        trade_df=trade_df,
        fold_market_df=fold_market_df,
        relative_strength_dates=relative_strength_dates,
        relative_strength_frames=relative_strength_frames,
        input_path=input_path,
    )

    return FoldStrategyResult(
        trade_df=trade_df,
        curve_df=curve_df,
        actual_cumulative_return=cumulative_return,
        annualized_return=calculate_fold_annualized_return(
            cumulative_return,
            test_bar_count,
            curve_df["Date"],
        ),
        annualized_sharpe=float(curve_summary["annualized_sharpe"]),
        trade_level_return_ratio=calculate_trade_level_return_ratio(adjusted_returns),
        max_drawdown=float(curve_summary["max_drawdown"]),
        number_of_trades=int(len(trade_df)),
        durations=durations,
        position_value_fractions=position_value_fractions,
        null_model_inputs=null_model_inputs,
        null_model_name=str(null_model_inputs.null_model_name),
        execution_cost_model=(
            "stochastic_execution_model" if SIMULATE_EXECUTION_COSTS else "flat_transaction_cost"
        ),
    )


def build_panel_run_row(
    *,
    ticker: str,
    fold_id: str,
    fold_start: pd.Timestamp,
    fold_end: pd.Timestamp,
    agent_name: str,
    outer_run: int,
    seed_used: int,
    actual_metrics: FoldStrategyResult,
    test_bar_count: int,
    simulated_returns: pd.Series,
    null_model_name: str,
) -> dict[str, float | int | str]:
    """Create one repeated-null run row for a walk-forward panel."""
    actual_cumulative_return = float(actual_metrics["actual_cumulative_return"])
    simulated_array = simulated_returns.to_numpy(dtype=float)
    mean_simulated_return = float(np.mean(simulated_array))
    std_simulated_return = float(np.std(simulated_array, ddof=0))
    median_simulated_return = float(np.median(simulated_array))
    actual_percentile = calculate_actual_percentile(simulated_array, actual_cumulative_return)
    p_value = calculate_p_value(simulated_array, actual_cumulative_return)
    p_value_prominence = calculate_p_value_prominence(p_value)
    rcsi = actual_cumulative_return - mean_simulated_return

    rcsi_z = np.nan
    if std_simulated_return > 0:
        rcsi_z = rcsi / std_simulated_return

    return {
        "ticker": ticker,
        "fold_id": fold_id,
        "fold_start": fold_start.strftime("%Y-%m-%d"),
        "fold_end": fold_end.strftime("%Y-%m-%d"),
        "agent": agent_name,
        "outer_run": outer_run,
        "seed_used": seed_used,
        "actual_cumulative_return": actual_cumulative_return,
        "annualized_return": float(actual_metrics["annualized_return"]),
        "annualized_sharpe": float(actual_metrics["annualized_sharpe"]),
        "trade_level_return_ratio": float(actual_metrics["trade_level_return_ratio"]),
        "max_drawdown": float(actual_metrics["max_drawdown"]),
        "median_simulated_return": median_simulated_return,
        "mean_simulated_return": mean_simulated_return,
        "std_simulated_return": std_simulated_return,
        "actual_percentile": float(actual_percentile),
        "p_value": float(p_value),
        "p_value_prominence": float(p_value_prominence),
        "RCSI": float(rcsi),
        "RCSI_z": float(rcsi_z) if pd.notna(rcsi_z) else np.nan,
        "significant_run": int(p_value <= 0.05),
        "outperform_null_median": int(actual_cumulative_return > median_simulated_return),
        "number_of_trades": int(actual_metrics["number_of_trades"]),
        "test_bar_count": int(test_bar_count),
        "transaction_cost": float(TRANSACTION_COST),
        "simulations_per_run": int(SIMULATIONS_PER_RUN),
        "null_model": str(null_model_name),
        "execution_cost_model": str(actual_metrics["execution_cost_model"]),
    }


PANEL_SUMMARY_METRICS = [
    "actual_cumulative_return",
    "annualized_return",
    "annualized_sharpe",
    "trade_level_return_ratio",
    "max_drawdown",
    "p_value",
    "p_value_prominence",
    "actual_percentile",
    "RCSI",
    "RCSI_z",
]


def aggregate_panel_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated runs into one panel summary row per ticker/fold/agent."""
    group_keys = ["ticker", "fold_id", "fold_start", "fold_end", "agent"]
    grouped = runs_df.groupby(group_keys, as_index=False)
    panel_df = grouped[group_keys].first()

    for metric_name in PANEL_SUMMARY_METRICS:
        metric_summary = grouped[metric_name].agg(
            mean="mean",
            median="median",
            std=series_std_or_nan,
            min="min",
            max="max",
            p25=lambda series: series_percentile_or_nan(series, 25),
            p75=lambda series: series_percentile_or_nan(series, 75),
        )
        metric_summary = metric_summary.rename(
            columns={
                "mean": f"mean_{metric_name}",
                "median": f"median_{metric_name}",
                "std": f"std_{metric_name}",
                "min": f"min_{metric_name}",
                "max": f"max_{metric_name}",
                "p25": f"p25_{metric_name}",
                "p75": f"p75_{metric_name}",
            }
        )
        panel_df = panel_df.merge(metric_summary, on=group_keys, how="left")

    panel_df["mean_number_of_trades"] = (
        grouped["number_of_trades"].mean()["number_of_trades"].to_numpy(dtype=float)
    )
    panel_df["proportion_significant_runs"] = (
        grouped["significant_run"].mean()["significant_run"].to_numpy(dtype=float)
    )
    panel_df["proportion_outperforming_null_runs"] = (
        grouped["outperform_null_median"].mean()["outperform_null_median"].to_numpy(dtype=float)
    )
    panel_df["number_of_outer_runs"] = int(OUTER_RUNS)
    panel_df["simulations_per_run"] = int(SIMULATIONS_PER_RUN)
    panel_df["test_bar_count"] = grouped["test_bar_count"].first()["test_bar_count"].to_numpy(dtype=int)
    panel_df["transaction_cost"] = float(TRANSACTION_COST)
    panel_df["null_model"] = grouped["null_model"].first()["null_model"].to_numpy(dtype=str)
    panel_df["execution_cost_model"] = (
        grouped["execution_cost_model"].first()["execution_cost_model"].to_numpy(dtype=str)
    )

    panel_df["agent"] = pd.Categorical(
        panel_df["agent"],
        categories=AGENT_ORDER,
        ordered=True,
    )
    panel_df = panel_df.sort_values(["ticker", "fold_id", "agent"]).reset_index(drop=True)
    panel_df["agent"] = panel_df["agent"].astype(str)
    return panel_df


def classify_stability(summary_row: pd.Series) -> str:
    """Classify cross-panel stability from percentile and RCSI_z variation."""
    percentile_range = float(
        summary_row["max_mean_actual_percentile"] - summary_row["min_mean_actual_percentile"]
    )
    rcsi_z_scale = max(abs(float(summary_row["mean_mean_RCSI_z"])), RCSI_Z_STABILITY_FLOOR)
    rcsi_z_relative_std = float(summary_row["std_mean_RCSI_z"]) / rcsi_z_scale

    if percentile_range <= 10.0 and rcsi_z_relative_std <= 0.35:
        return "stable"
    if percentile_range <= 25.0 and rcsi_z_relative_std <= 0.70:
        return "moderately variable"
    return "unstable"


AGENT_SUMMARY_SOURCE_METRICS = [
    "mean_actual_cumulative_return",
    "mean_annualized_return",
    "mean_annualized_sharpe",
    "mean_trade_level_return_ratio",
    "mean_max_drawdown",
    "mean_p_value",
    "mean_p_value_prominence",
    "mean_actual_percentile",
    "mean_RCSI",
    "mean_RCSI_z",
]


def aggregate_agent_summary(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate panel summaries into one overall repeated-evaluation table per agent."""
    grouped = panel_df.groupby("agent", as_index=False)
    summary_df = grouped[["agent"]].first()

    for metric_name in AGENT_SUMMARY_SOURCE_METRICS:
        metric_summary = grouped[metric_name].agg(
            mean="mean",
            median="median",
            std=series_std_or_nan,
            min="min",
            max="max",
            p25=lambda series: series_percentile_or_nan(series, 25),
            p75=lambda series: series_percentile_or_nan(series, 75),
        )
        metric_summary = metric_summary.rename(
            columns={
                "mean": f"mean_{metric_name}",
                "median": f"median_{metric_name}",
                "std": f"std_{metric_name}",
                "min": f"min_{metric_name}",
                "max": f"max_{metric_name}",
                "p25": f"p25_{metric_name}",
                "p75": f"p75_{metric_name}",
            }
        )
        summary_df = summary_df.merge(metric_summary, on="agent", how="left")

    summary_df["panel_count"] = grouped.size()["size"].to_numpy(dtype=int)
    summary_df["mean_proportion_significant_runs"] = (
        grouped["proportion_significant_runs"]
        .mean()["proportion_significant_runs"]
        .to_numpy(dtype=float)
    )
    summary_df["mean_proportion_outperforming_null_runs"] = (
        grouped["proportion_outperforming_null_runs"]
        .mean()["proportion_outperforming_null_runs"]
        .to_numpy(dtype=float)
    )
    summary_df["proportion_significant_panels"] = grouped["mean_p_value"].agg(
        lambda series: float((pd.to_numeric(series, errors="coerce") <= 0.05).mean())
    )["mean_p_value"].to_numpy(dtype=float)
    summary_df["proportion_positive_rcsi_z_panels"] = grouped["mean_RCSI_z"].agg(
        lambda series: float((pd.to_numeric(series, errors="coerce") > 0).mean())
    )["mean_RCSI_z"].to_numpy(dtype=float)

    summary_df["stability_classification"] = summary_df.apply(classify_stability, axis=1)
    summary_df["evidence_tuple"] = summary_df.apply(
        lambda row: classify_robustness_evidence(
            p_value=row["mean_mean_p_value"],
            rcsi=row["mean_mean_RCSI"],
            rcsi_z=row["mean_mean_RCSI_z"],
            percentile=row["mean_mean_actual_percentile"],
            proportion_significant=row["mean_proportion_significant_runs"],
            proportion_outperforming_null_median=row["mean_proportion_outperforming_null_runs"],
            stability_classification=str(row["stability_classification"]),
        ),
        axis=1,
    )
    summary_df["evidence_bucket"] = summary_df["evidence_tuple"].apply(lambda value: value[0])
    summary_df["confidence_score"] = summary_df["evidence_tuple"].apply(lambda value: value[1])
    summary_df["evaluation_power_score"] = summary_df.apply(
        lambda _row: evaluation_power_score(
            number_of_outer_runs=OUTER_RUNS,
            simulations_per_run=SIMULATIONS_PER_RUN,
        ),
        axis=1,
    )
    summary_df["power_adjustment"] = summary_df.apply(
        lambda row: adjust_for_evaluation_power(
            evidence_bucket=row["evidence_bucket"],
            confidence_score=row["confidence_score"],
            power_score=row["evaluation_power_score"],
        ),
        axis=1,
    )
    summary_df["evidence_bucket"] = summary_df["power_adjustment"].apply(lambda value: value[0])
    summary_df["confidence_score"] = summary_df["power_adjustment"].apply(lambda value: value[1])
    summary_df["confidence_bucket"] = summary_df["confidence_score"].apply(confidence_bucket_from_score)
    summary_df["evaluation_power_label"] = summary_df["evaluation_power_score"].apply(evaluation_power_label)
    summary_df["evidence_label"] = summary_df["evidence_bucket"].apply(evidence_label)
    summary_df["confidence_label"] = summary_df["confidence_bucket"].apply(confidence_label)
    summary_df["skill_luck_verdict"] = summary_df["evidence_bucket"].apply(verdict_from_evidence_bucket)
    summary_df["verdict_label"] = summary_df["skill_luck_verdict"].apply(format_verdict_label)
    summary_df["final_classification"] = summary_df["evidence_label"].str.lower()
    summary_df["number_of_outer_runs"] = int(OUTER_RUNS)
    summary_df["simulations_per_run"] = int(SIMULATIONS_PER_RUN)
    summary_df["test_bars"] = int(TEST_BARS)
    summary_df["step_bars"] = int(STEP_BARS)
    summary_df["min_trades_per_panel"] = int(MIN_TRADES_PER_PANEL)
    summary_df["transaction_cost"] = float(TRANSACTION_COST)
    summary_df["null_model"] = grouped["null_model"].first()["null_model"].to_numpy(dtype=str)
    summary_df["execution_cost_model"] = (
        grouped["execution_cost_model"].first()["execution_cost_model"].to_numpy(dtype=str)
    )

    summary_df["agent"] = pd.Categorical(
        summary_df["agent"],
        categories=AGENT_ORDER,
        ordered=True,
    )
    summary_df = summary_df.sort_values("agent").reset_index(drop=True)
    summary_df["agent"] = summary_df["agent"].astype(str)
    summary_df = summary_df.drop(columns=["evidence_tuple", "power_adjustment"])
    return summary_df


def print_agent_summary(summary_row: pd.Series) -> None:
    """Print a compact interpretation for one walk-forward agent summary."""
    print(f"\nAgent: {format_agent_name(summary_row['agent'])}")
    print(f"Panels evaluated: {int(summary_row['panel_count'])}")
    print(f"Mean panel p-value: {summary_row['mean_mean_p_value']:.6f}")
    print(f"Mean panel percentile: {summary_row['mean_mean_actual_percentile']:.2f}")
    print(f"Mean panel RCSI_z: {summary_row['mean_mean_RCSI_z']:.6f}")
    print(
        "Mean proportion of significant runs: "
        f"{float(summary_row['mean_proportion_significant_runs']):.2%}"
    )
    print(
        "Mean proportion outperforming null: "
        f"{float(summary_row['mean_proportion_outperforming_null_runs']):.2%}"
    )
    print(f"Stability: {summary_row['stability_classification']}")
    print(
        "Evaluation power: "
        f"{summary_row['evaluation_power_label']} "
        f"(score={float(summary_row['evaluation_power_score']):.2f})"
    )
    print(f"Evidence: {summary_row['evidence_label']}")
    print(f"Confidence: {summary_row['confidence_label']}")
    print(f"Final classification: {summary_row['final_classification']}")
    print(f"Skill vs luck verdict: {summary_row['verdict_label']}")
    print(f"Null model: {summary_row['null_model']}")


def main() -> None:
    """Run the multi-asset walk-forward evaluation on the configured ticker list."""
    runs_output_path = data_clean_dir() / "multi_asset_walk_forward_runs.csv"
    panel_output_path = data_clean_dir() / "multi_asset_walk_forward_panel_summary.csv"
    summary_output_path = data_clean_dir() / "multi_asset_walk_forward_agent_summary.csv"

    start_time = time.perf_counter()
    run_rows: list[dict[str, Any]] = []
    panel_counter = 0

    for ticker in TICKERS:
        ensure_ticker_inputs(ticker)
        market_df = load_fold_market_data(ticker)
        folds = build_folds(market_df)

        if not folds:
            print(
                f"[walk-forward] skipping {ticker}: not enough history for one test fold.",
                flush=True,
            )
            continue

        for fold in folds:
            fold_market_df = fold["market_df"]
            fold_start = fold["start_date"]
            fold_end = fold["end_date"]
            test_bar_count = int(fold["test_bar_count"])
            fold_id = str(fold["fold_id"])

            for agent_name in AGENT_ORDER:
                actual_metrics = build_fold_actual_metrics(
                    ticker=ticker,
                    fold_id=fold_id,
                    agent_name=agent_name,
                    fold_market_df=fold_market_df,
                    test_bar_count=test_bar_count,
                )
                if actual_metrics is None:
                    continue

                panel_counter += 1
                for outer_run in range(1, OUTER_RUNS + 1):
                    seed_used = BASE_SEED + outer_run
                    if panel_counter == 1 and (
                        outer_run == 1 or outer_run % PROGRESS_EVERY == 0
                    ):
                        elapsed_seconds = time.perf_counter() - start_time
                        print(
                            f"[walk-forward] warm-up run {outer_run}/{OUTER_RUNS} "
                            f"elapsed={elapsed_seconds:.1f}s",
                            flush=True,
                        )

                    seed_sequence = np.random.SeedSequence(
                        [
                            seed_used,
                            panel_counter,
                            sum(ord(char) for char in ticker),
                            AGENT_ORDER.index(agent_name),
                        ]
                    )
                    rng = np.random.default_rng(seed_sequence)
                    simulated_returns = simulate_structure_preserving_cumulative_returns(
                        null_model_inputs=actual_metrics["null_model_inputs"],
                        simulation_count=SIMULATIONS_PER_RUN,
                        rng=rng,
                    )
                    run_rows.append(
                        build_panel_run_row(
                            ticker=ticker,
                            fold_id=fold_id,
                            fold_start=fold_start,
                            fold_end=fold_end,
                            agent_name=agent_name,
                            outer_run=outer_run,
                            seed_used=seed_used,
                            actual_metrics=actual_metrics,
                            test_bar_count=test_bar_count,
                            simulated_returns=simulated_returns,
                            null_model_name=str(actual_metrics["null_model_name"]),
                        )
                    )

            elapsed_seconds = time.perf_counter() - start_time
            print(
                f"[walk-forward] ticker={ticker} fold={fold_id} "
                f"window={fold_start.date()}->{fold_end.date()} elapsed={elapsed_seconds:.1f}s",
                flush=True,
            )

    if not run_rows:
        raise ValueError("No walk-forward panels met the minimum trade requirement.")

    runs_df = pd.DataFrame(run_rows)
    write_dataframe_artifact(
        runs_df,
        runs_output_path,
        producer="multi_asset_walk_forward.main",
        current_ticker="MULTI_ASSET",
        research_grade=bool(OUTER_RUNS >= 30 and SIMULATIONS_PER_RUN >= 1000),
        canonical_policy="auto",
        parameters={
            "tickers": TICKERS,
            "test_bars": TEST_BARS,
            "step_bars": STEP_BARS,
            "outer_runs": OUTER_RUNS,
            "simulations_per_run": SIMULATIONS_PER_RUN,
            "min_trades_per_panel": MIN_TRADES_PER_PANEL,
            "refresh_data": REFRESH_DATA,
        },
    )

    panel_df = aggregate_panel_runs(runs_df)
    write_dataframe_artifact(
        panel_df,
        panel_output_path,
        producer="multi_asset_walk_forward.main",
        current_ticker="MULTI_ASSET",
        dependencies=[runs_output_path],
        research_grade=bool(OUTER_RUNS >= 30 and SIMULATIONS_PER_RUN >= 1000),
        canonical_policy="auto",
        parameters={
            "tickers": TICKERS,
            "test_bars": TEST_BARS,
            "step_bars": STEP_BARS,
            "outer_runs": OUTER_RUNS,
            "simulations_per_run": SIMULATIONS_PER_RUN,
            "min_trades_per_panel": MIN_TRADES_PER_PANEL,
        },
    )

    summary_df = aggregate_agent_summary(panel_df)
    write_dataframe_artifact(
        summary_df,
        summary_output_path,
        producer="multi_asset_walk_forward.main",
        current_ticker="MULTI_ASSET",
        dependencies=[panel_output_path],
        research_grade=bool(OUTER_RUNS >= 30 and SIMULATIONS_PER_RUN >= 1000),
        canonical_policy="auto",
        parameters={
            "tickers": TICKERS,
            "test_bars": TEST_BARS,
            "step_bars": STEP_BARS,
            "outer_runs": OUTER_RUNS,
            "simulations_per_run": SIMULATIONS_PER_RUN,
            "min_trades_per_panel": MIN_TRADES_PER_PANEL,
        },
    )

    total_elapsed_seconds = time.perf_counter() - start_time
    print(f"[walk-forward] completed in {total_elapsed_seconds:.1f}s", flush=True)
    print(f"[walk-forward] tickers: {', '.join(TICKERS)}", flush=True)
    print(f"[walk-forward] panels evaluated: {len(panel_df)}", flush=True)

    for _, summary_row in summary_df.iterrows():
        print_agent_summary(summary_row)


if __name__ == "__main__":
    main()
