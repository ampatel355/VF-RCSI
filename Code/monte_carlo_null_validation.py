"""Validate the structure-preserving timing null against the legacy baseline."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from artifact_provenance import write_dataframe_artifact
    from monte_carlo import (
        ENTRY_CONTEXT_LOOKBACK_BARS,
        CONTEXT_MATCHING_ENABLED,
        LEGACY_NULL_MODEL_NAME,
        LEGACY_RELATIVE_STRENGTH_NULL_MODEL_NAME,
        TRANSACTION_COST,
        build_context_preserving_schedule_batch,
        build_entry_context_matrices,
        build_legacy_trade_schedule_batch,
        build_structure_preserving_schedule_batch,
        calculate_actual_percentile,
        calculate_cumulative_return_from_log_returns,
        calculate_p_value,
        calculate_same_bar_turnover_share,
        calculate_weighted_exposure_share,
        convert_to_log_returns,
        load_market_data,
        load_trade_data,
        prepare_agent_null_model_inputs,
        resolve_data_clean_dir,
        simulate_legacy_agent_null_cumulative_returns,
        simulate_structure_preserving_cumulative_returns,
        adjust_trade_returns,
    )
    from single_ticker_agent_common import load_regime_data
    from strategy_artifact_utils import ensure_trade_file_exists
    from strategy_config import AGENT_ORDER
    from strategy_simulator import run_strategy
    from strategy_verdicts import classify_metrics
except ModuleNotFoundError:
    from Code.artifact_provenance import write_dataframe_artifact
    from Code.monte_carlo import (
        ENTRY_CONTEXT_LOOKBACK_BARS,
        CONTEXT_MATCHING_ENABLED,
        LEGACY_NULL_MODEL_NAME,
        LEGACY_RELATIVE_STRENGTH_NULL_MODEL_NAME,
        TRANSACTION_COST,
        adjust_trade_returns,
        build_context_preserving_schedule_batch,
        build_entry_context_matrices,
        build_legacy_trade_schedule_batch,
        build_structure_preserving_schedule_batch,
        calculate_actual_percentile,
        calculate_cumulative_return_from_log_returns,
        calculate_p_value,
        calculate_same_bar_turnover_share,
        calculate_weighted_exposure_share,
        convert_to_log_returns,
        load_market_data,
        load_trade_data,
        prepare_agent_null_model_inputs,
        resolve_data_clean_dir,
        simulate_legacy_agent_null_cumulative_returns,
        simulate_structure_preserving_cumulative_returns,
    )
    from Code.single_ticker_agent_common import load_regime_data
    from Code.strategy_artifact_utils import ensure_trade_file_exists
    from Code.strategy_config import AGENT_ORDER
    from Code.strategy_simulator import run_strategy
    from Code.strategy_verdicts import classify_metrics


ticker = os.environ.get("TICKER", "SPY").strip().upper()
COMPARISON_SIMULATIONS = int(os.environ.get("NULL_VALIDATION_SIMULATIONS", "2000"))
RANDOM_BASELINE_RUNS = int(os.environ.get("NULL_VALIDATION_RANDOM_RUNS", "30"))
RANDOM_BASELINE_SIMULATIONS = int(
    os.environ.get("NULL_VALIDATION_RANDOM_SIMULATIONS", str(COMPARISON_SIMULATIONS))
)
BASE_SEED = int(os.environ.get("NULL_VALIDATION_BASE_SEED", "700"))
MIN_RESEARCH_GRADE_VALIDATION_RUNS = int(os.environ.get("MIN_RESEARCH_GRADE_VALIDATION_RUNS", "30"))
MIN_RESEARCH_GRADE_VALIDATION_SIMULATIONS = int(
    os.environ.get("MIN_RESEARCH_GRADE_VALIDATION_SIMULATIONS", "1000")
)


def safe_nanmean(values: np.ndarray) -> float:
    """Return a scalar nanmean without emitting warnings for all-NaN slices."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.isnan(values).all():
        return np.nan
    return float(np.nanmean(values))


def rowwise_nanmean(values: np.ndarray) -> np.ndarray:
    """Return one nanmean per row without warning on all-NaN rows."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("rowwise_nanmean expects a 2D array.")

    valid_mask = ~np.isnan(values)
    valid_counts = valid_mask.sum(axis=1)
    row_sums = np.nansum(values, axis=1)
    row_means = np.full(values.shape[0], np.nan, dtype=float)
    non_empty_rows = valid_counts > 0
    row_means[non_empty_rows] = row_sums[non_empty_rows] / valid_counts[non_empty_rows]
    return row_means


def calculate_actual_cumulative_return(trade_df: pd.DataFrame, input_path: Path | str) -> float:
    """Calculate the realized cumulative return from the saved trade log."""
    if trade_df.empty:
        return 0.0

    raw_returns = trade_df["return"].to_numpy(dtype=float)
    adjusted_returns = adjust_trade_returns(
        raw_returns=raw_returns,
        transaction_cost=TRANSACTION_COST,
        input_path=input_path,
    )
    log_returns = convert_to_log_returns(adjusted_returns)
    return calculate_cumulative_return_from_log_returns(log_returns)


def build_fairness_rows(
    *,
    agent_name: str,
    null_variant: str,
    null_model_name: str,
    actual_cumulative_return: float,
    null_model_inputs,
    simulated_returns: pd.Series,
    schedule_rng: np.random.Generator,
) -> dict[str, float | int | str | bool]:
    """Summarize one null model's structural fairness for one strategy."""
    trade_structure = null_model_inputs.trade_structure
    open_price_matrix = null_model_inputs.open_price_matrix
    max_open_index = open_price_matrix.shape[1] - 1
    trade_count = len(trade_structure.durations)
    actual_has_same_bar_turnover = bool(
        np.any(trade_structure.entry_indices[1:] == trade_structure.exit_indices[:-1])
    )

    trailing_returns, trailing_volatility = build_entry_context_matrices(open_price_matrix)
    actual_entry_trailing_returns = trailing_returns[
        trade_structure.asset_indices,
        trade_structure.entry_indices,
    ]
    actual_entry_trailing_volatility = trailing_volatility[
        trade_structure.asset_indices,
        trade_structure.entry_indices,
    ]

    exposure_share_samples: list[np.ndarray] = []
    same_bar_share_samples: list[np.ndarray] = []
    entry_trailing_return_samples: list[np.ndarray] = []
    entry_trailing_volatility_samples: list[np.ndarray] = []

    for batch_start in range(0, COMPARISON_SIMULATIONS, 512):
        batch_size = min(512, COMPARISON_SIMULATIONS - batch_start)
        if null_variant == "structure_preserving":
            if CONTEXT_MATCHING_ENABLED and null_model_inputs.context_entry_candidate_pools:
                entry_indices, exit_indices = build_context_preserving_schedule_batch(
                    trade_structure=trade_structure,
                    max_open_index=max_open_index,
                    batch_size=batch_size,
                    rng=schedule_rng,
                    candidate_pools=null_model_inputs.context_entry_candidate_pools,
                )
            else:
                entry_indices, exit_indices = build_structure_preserving_schedule_batch(
                    trade_structure=trade_structure,
                    max_open_index=max_open_index,
                    batch_size=batch_size,
                    rng=schedule_rng,
                )
            durations_batch = np.broadcast_to(
                trade_structure.durations[np.newaxis, :],
                entry_indices.shape,
            )
            fractions_batch = np.broadcast_to(
                trade_structure.position_value_fractions[np.newaxis, :],
                entry_indices.shape,
            )
            asset_indices_batch = np.broadcast_to(
                trade_structure.asset_indices[np.newaxis, :],
                entry_indices.shape,
            )
            same_bar_capability_preserved = (
                (not actual_has_same_bar_turnover)
                or bool(np.any(trade_structure.transition_gap_floors == 0))
            )
            ordered_durations_preserved = True
            ordered_assets_preserved = True
        else:
            entry_indices, exit_indices, permutation_indices = build_legacy_trade_schedule_batch(
                durations=trade_structure.durations,
                max_open_index=max_open_index,
                batch_size=batch_size,
                rng=schedule_rng,
            )
            durations_batch = trade_structure.durations[permutation_indices]
            fractions_batch = trade_structure.position_value_fractions[permutation_indices]
            if agent_name == "momentum_relative_strength":
                asset_indices_batch = schedule_rng.integers(
                    low=0,
                    high=open_price_matrix.shape[0],
                    size=entry_indices.shape,
                    dtype=np.int64,
                )
            else:
                asset_indices_batch = np.zeros_like(entry_indices, dtype=np.int64)
            same_bar_capability_preserved = not actual_has_same_bar_turnover
            ordered_durations_preserved = False
            ordered_assets_preserved = agent_name != "momentum_relative_strength"

        exposure_share_samples.append(
            np.sum(durations_batch * fractions_batch, axis=1) / float(max_open_index)
        )
        if trade_count > 1:
            same_bar_share_samples.append(
                np.mean(entry_indices[:, 1:] == exit_indices[:, :-1], axis=1)
            )
        else:
            same_bar_share_samples.append(np.zeros(batch_size, dtype=float))
        entry_trailing_return_samples.append(
            rowwise_nanmean(trailing_returns[asset_indices_batch, entry_indices])
        )
        entry_trailing_volatility_samples.append(
            rowwise_nanmean(trailing_volatility[asset_indices_batch, entry_indices])
        )

    simulated_array = simulated_returns.to_numpy(dtype=float)
    mean_simulated_return = float(np.mean(simulated_array))
    std_simulated_return = float(np.std(simulated_array, ddof=0))
    actual_percentile = calculate_actual_percentile(simulated_array, actual_cumulative_return)
    p_value = calculate_p_value(simulated_array, actual_cumulative_return)
    rcsi = actual_cumulative_return - mean_simulated_return
    rcsi_z = np.nan
    if std_simulated_return > 0:
        rcsi_z = rcsi / std_simulated_return

    return {
        "agent": agent_name,
        "null_variant": null_variant,
        "null_model": null_model_name,
        "comparison_simulations": COMPARISON_SIMULATIONS,
        "actual_trade_count": trade_count,
        "simulated_mean_trade_count": float(trade_count),
        "actual_mean_holding_bars": float(np.mean(trade_structure.durations)),
        "simulated_mean_holding_bars": float(np.mean(trade_structure.durations)),
        "actual_exposure_share": calculate_weighted_exposure_share(
            trade_structure.durations,
            trade_structure.position_value_fractions,
            max_open_index,
        ),
        "simulated_mean_exposure_share": float(np.mean(np.concatenate(exposure_share_samples))),
        "actual_same_bar_turnover_share": calculate_same_bar_turnover_share(
            trade_structure.entry_indices,
            trade_structure.exit_indices,
        ),
        "simulated_mean_same_bar_turnover_share": float(
            np.mean(np.concatenate(same_bar_share_samples))
        ),
        "actual_entry_trailing_return_mean": safe_nanmean(actual_entry_trailing_returns),
        "simulated_entry_trailing_return_mean": safe_nanmean(
            np.concatenate(entry_trailing_return_samples)
        ),
        "actual_entry_trailing_volatility_mean": safe_nanmean(
            actual_entry_trailing_volatility
        ),
        "simulated_entry_trailing_volatility_mean": safe_nanmean(
            np.concatenate(entry_trailing_volatility_samples)
        ),
        "ordered_durations_preserved": ordered_durations_preserved,
        "ordered_assets_preserved": ordered_assets_preserved,
        "same_bar_turnover_capability_preserved": same_bar_capability_preserved,
        "lookback_bars_for_context": ENTRY_CONTEXT_LOOKBACK_BARS,
        "actual_cumulative_return": actual_cumulative_return,
        "mean_simulated_return": mean_simulated_return,
        "std_simulated_return": std_simulated_return,
        "actual_percentile": actual_percentile,
        "p_value": p_value,
        "RCSI": rcsi,
        "RCSI_z": rcsi_z,
        "evidence_bucket": classify_metrics(
            rcsi_z=rcsi_z,
            p_value=p_value,
            percentile=actual_percentile,
        ),
    }


def validate_against_legacy(
    market_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Compare the new null against the legacy null for every saved strategy."""
    comparison_rows: list[dict[str, float | int | str | bool]] = []

    for agent_name in AGENT_ORDER:
        input_path = ensure_trade_file_exists(ticker, agent_name)
        trade_df = load_trade_data(input_path, allow_empty=True)
        if trade_df.empty:
            continue

        null_model_inputs = prepare_agent_null_model_inputs(
            agent_name=agent_name,
            current_ticker=ticker,
            trade_df=trade_df,
            market_df=market_df,
            input_path=input_path,
        )
        actual_cumulative_return = calculate_actual_cumulative_return(trade_df, input_path)

        structure_rng = np.random.default_rng(BASE_SEED + (len(comparison_rows) + 1) * 11)
        structure_simulated_returns = simulate_structure_preserving_cumulative_returns(
            null_model_inputs=null_model_inputs,
            simulation_count=COMPARISON_SIMULATIONS,
            rng=structure_rng,
        )
        comparison_rows.append(
            build_fairness_rows(
                agent_name=agent_name,
                null_variant="structure_preserving",
                null_model_name=null_model_inputs.null_model_name,
                actual_cumulative_return=actual_cumulative_return,
                null_model_inputs=null_model_inputs,
                simulated_returns=structure_simulated_returns,
                schedule_rng=np.random.default_rng(BASE_SEED + (len(comparison_rows) + 1) * 17),
            )
        )

        legacy_simulated_returns, legacy_null_model_name = (
            simulate_legacy_agent_null_cumulative_returns(
                agent_name=agent_name,
                null_model_inputs=null_model_inputs,
                simulation_count=COMPARISON_SIMULATIONS,
                rng=np.random.default_rng(BASE_SEED + (len(comparison_rows) + 1) * 23),
            )
        )
        comparison_rows.append(
            build_fairness_rows(
                agent_name=agent_name,
                null_variant="legacy",
                null_model_name=legacy_null_model_name,
                actual_cumulative_return=actual_cumulative_return,
                null_model_inputs=null_model_inputs,
                simulated_returns=legacy_simulated_returns,
                schedule_rng=np.random.default_rng(BASE_SEED + (len(comparison_rows) + 1) * 29),
            )
        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df["exposure_share_error"] = (
        comparison_df["simulated_mean_exposure_share"] - comparison_df["actual_exposure_share"]
    )
    comparison_df["entry_trailing_return_shift"] = (
        comparison_df["simulated_entry_trailing_return_mean"]
        - comparison_df["actual_entry_trailing_return_mean"]
    )
    comparison_df["entry_trailing_volatility_shift"] = (
        comparison_df["simulated_entry_trailing_volatility_mean"]
        - comparison_df["actual_entry_trailing_volatility_mean"]
    )
    write_dataframe_artifact(
        comparison_df,
        output_path,
        producer="monte_carlo_null_validation.validate_against_legacy",
        current_ticker=ticker,
        research_grade=COMPARISON_SIMULATIONS >= MIN_RESEARCH_GRADE_VALIDATION_SIMULATIONS,
        canonical_policy="auto",
        parameters={
            "comparison_simulations": COMPARISON_SIMULATIONS,
        },
    )
    return comparison_df


def validate_random_baseline(
    market_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    output_runs_path: Path,
    output_summary_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check that the random strategy mostly maps to the null-like classification."""
    rows: list[dict[str, float | int | str]] = []
    for run_number in range(1, RANDOM_BASELINE_RUNS + 1):
        seed_used = BASE_SEED + 1000 + run_number
        random_trade_df = run_strategy(
            "random",
            regime_df.copy(),
            random_decision_seed=seed_used,
            ticker=ticker,
        )
        if random_trade_df.empty:
            rows.append(
                {
                    "seed_used": seed_used,
                    "number_of_trades": 0,
                    "actual_cumulative_return": 0.0,
                    "mean_simulated_return": 0.0,
                    "std_simulated_return": 0.0,
                    "actual_percentile": np.nan,
                    "p_value": np.nan,
                    "RCSI": np.nan,
                    "RCSI_z": np.nan,
                    "evidence_bucket": "no_trades",
                    "null_model": LEGACY_NULL_MODEL_NAME,
                }
            )
            continue

        input_path = f"{ticker}_random_validation_seed_{seed_used}"
        null_model_inputs = prepare_agent_null_model_inputs(
            agent_name="random",
            current_ticker=ticker,
            trade_df=random_trade_df,
            market_df=market_df,
            input_path=input_path,
        )
        actual_cumulative_return = calculate_actual_cumulative_return(random_trade_df, input_path)
        simulated_returns = simulate_structure_preserving_cumulative_returns(
            null_model_inputs=null_model_inputs,
            simulation_count=RANDOM_BASELINE_SIMULATIONS,
            rng=np.random.default_rng(seed_used),
        )
        simulated_array = simulated_returns.to_numpy(dtype=float)
        mean_simulated_return = float(np.mean(simulated_array))
        std_simulated_return = float(np.std(simulated_array, ddof=0))
        actual_percentile = calculate_actual_percentile(simulated_array, actual_cumulative_return)
        p_value = calculate_p_value(simulated_array, actual_cumulative_return)
        rcsi = actual_cumulative_return - mean_simulated_return
        rcsi_z = np.nan
        if std_simulated_return > 0:
            rcsi_z = rcsi / std_simulated_return

        rows.append(
            {
                "seed_used": seed_used,
                "number_of_trades": int(len(random_trade_df)),
                "actual_cumulative_return": actual_cumulative_return,
                "mean_simulated_return": mean_simulated_return,
                "std_simulated_return": std_simulated_return,
                "actual_percentile": actual_percentile,
                "p_value": p_value,
                "RCSI": rcsi,
                "RCSI_z": rcsi_z,
                "evidence_bucket": classify_metrics(
                    rcsi_z=rcsi_z,
                    p_value=p_value,
                    percentile=actual_percentile,
                ),
                "null_model": null_model_inputs.null_model_name,
            }
        )

    runs_df = pd.DataFrame(rows)
    research_grade = (
        RANDOM_BASELINE_RUNS >= MIN_RESEARCH_GRADE_VALIDATION_RUNS
        and RANDOM_BASELINE_SIMULATIONS >= MIN_RESEARCH_GRADE_VALIDATION_SIMULATIONS
    )
    write_dataframe_artifact(
        runs_df,
        output_runs_path,
        producer="monte_carlo_null_validation.validate_random_baseline",
        current_ticker=ticker,
        research_grade=research_grade,
        canonical_policy="auto",
        parameters={
            "random_baseline_runs": RANDOM_BASELINE_RUNS,
            "random_baseline_simulations": RANDOM_BASELINE_SIMULATIONS,
        },
    )

    summary_df = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "random_baseline_runs": RANDOM_BASELINE_RUNS,
                "random_baseline_simulations": RANDOM_BASELINE_SIMULATIONS,
                "mean_percentile": safe_nanmean(runs_df["actual_percentile"].to_numpy(dtype=float)),
                "median_percentile": float(
                    np.nanmedian(runs_df["actual_percentile"].to_numpy(dtype=float))
                ),
                "mean_p_value": safe_nanmean(runs_df["p_value"].to_numpy(dtype=float)),
                "false_positive_rate_p_le_0_05": float(
                    np.nanmean((runs_df["p_value"].to_numpy(dtype=float) <= 0.05).astype(float))
                ),
                "random_luck_rate": float(
                    np.mean(runs_df["evidence_bucket"].astype(str) == "random_luck")
                ),
                "weak_skill_rate": float(
                    np.mean(runs_df["evidence_bucket"].astype(str) == "weak_skill")
                ),
                "moderate_or_strong_skill_rate": float(
                    np.mean(
                        runs_df["evidence_bucket"].astype(str).isin(
                            ["moderate_skill", "strong_skill"]
                        )
                    )
                ),
            }
        ]
    )
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_dataframe_artifact(
        summary_df,
        output_summary_path,
        producer="monte_carlo_null_validation.validate_random_baseline",
        current_ticker=ticker,
        dependencies=[output_runs_path],
        research_grade=research_grade,
        canonical_policy="auto",
        parameters={
            "random_baseline_runs": RANDOM_BASELINE_RUNS,
            "random_baseline_simulations": RANDOM_BASELINE_SIMULATIONS,
        },
    )
    return runs_df, summary_df


def main() -> None:
    """Run the fairness comparison and the random-baseline validation."""
    project_root = Path(__file__).resolve().parents[1]
    data_clean_dir = resolve_data_clean_dir(project_root)
    market_path = data_clean_dir / f"{ticker}_regimes.csv"
    market_df = load_market_data(market_path)
    regime_df = load_regime_data(
        market_path,
        required_columns=["Date", "Open", "High", "Low", "Close", "avg_volume_20", "regime"],
    )
    regime_df.attrs["ticker"] = ticker

    comparison_output_path = data_clean_dir / f"{ticker}_monte_carlo_null_comparison.csv"
    random_runs_output_path = (
        data_clean_dir / f"{ticker}_monte_carlo_random_baseline_validation.csv"
    )
    random_summary_output_path = (
        data_clean_dir / f"{ticker}_monte_carlo_random_baseline_summary.csv"
    )

    comparison_df = validate_against_legacy(
        market_df=market_df,
        output_path=comparison_output_path,
    )
    _, random_summary_df = validate_random_baseline(
        market_df=market_df,
        regime_df=regime_df,
        output_runs_path=random_runs_output_path,
        output_summary_path=random_summary_output_path,
    )

    print("\nNull-model comparison:")
    print(comparison_df.to_string(index=False))
    print("\nRandom-baseline summary:")
    print(random_summary_df.to_string(index=False))
    print(
        "\nSaved validation outputs to:\n"
        f"  {comparison_output_path}\n"
        f"  {random_runs_output_path}\n"
        f"  {random_summary_output_path}"
    )


if __name__ == "__main__":
    main()
