"""Run the optional experimental strategy suite through the core testing stack."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from artifact_provenance import write_dataframe_artifact
    from monte_carlo import (
        NULL_MODEL_NAME,
        NUMBER_OF_SIMULATIONS,
        REPRODUCIBLE,
        SEED,
        TRANSACTION_COST,
        benjamini_hochberg_adjusted_p_values,
        build_agent_summary,
        build_no_trade_simulated_returns,
        build_no_trade_summary,
        build_random_generator,
        calculate_cumulative_return_from_log_returns,
        convert_to_log_returns,
        load_market_data,
        load_trade_data,
        save_simulation_results,
        simulate_agent_null_cumulative_returns,
        adjust_trade_returns,
    )
    from single_ticker_agent_common import load_regime_data, save_trade_outputs
    from strategy_config import EXPERIMENTAL_AGENT_ORDER, format_strategy_name
    from strategy_metrics_common import create_and_save_metrics
    from strategy_simulator import resolve_data_clean_dir, run_strategy
    from strategy_verdicts import classify_metrics, evidence_label
except ModuleNotFoundError:
    from Code.artifact_provenance import write_dataframe_artifact
    from Code.monte_carlo import (
        NULL_MODEL_NAME,
        NUMBER_OF_SIMULATIONS,
        REPRODUCIBLE,
        SEED,
        TRANSACTION_COST,
        benjamini_hochberg_adjusted_p_values,
        build_agent_summary,
        build_no_trade_simulated_returns,
        build_no_trade_summary,
        build_random_generator,
        calculate_cumulative_return_from_log_returns,
        convert_to_log_returns,
        load_market_data,
        load_trade_data,
        save_simulation_results,
        simulate_agent_null_cumulative_returns,
        adjust_trade_returns,
    )
    from Code.single_ticker_agent_common import load_regime_data, save_trade_outputs
    from Code.strategy_config import EXPERIMENTAL_AGENT_ORDER, format_strategy_name
    from Code.strategy_metrics_common import create_and_save_metrics
    from Code.strategy_simulator import resolve_data_clean_dir, run_strategy
    from Code.strategy_verdicts import classify_metrics, evidence_label


ticker = os.environ.get("TICKER", "SPY").strip().upper()


def _required_columns() -> list[str]:
    """Return the union of features needed by the experimental strategy set."""
    return [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "avg_volume_20",
        "ema_20",
        "sma_50",
        "sma_200",
        "atr_14",
        "adx_14",
        "plus_di_14",
        "minus_di_14",
        "rsi_14",
        "macd_line",
        "macd_signal",
        "atr_percent_ratio_60",
        "volume_ratio_20",
        "rolling_high_20_prev",
        "rolling_low_10_prev",
        "trailing_return_20",
        "zscore_20",
        "bollinger_lower",
        "bollinger_width_ratio_60",
        "regime",
    ]


def _build_monte_carlo_row(
    *,
    agent_name: str,
    current_ticker: str,
    trade_path: Path,
    market_df: pd.DataFrame,
    rng: np.random.Generator,
    data_clean_dir: Path,
) -> dict[str, float | int | str | bool]:
    """Run structure-preserving Monte Carlo for one strategy trade log."""
    trade_df = load_trade_data(trade_path, allow_empty=True)
    results_output_path = data_clean_dir / f"{current_ticker}_{agent_name}_monte_carlo_results.csv"
    market_path = data_clean_dir / f"{current_ticker}_regimes.csv"

    if trade_df.empty:
        simulated_returns = build_no_trade_simulated_returns(NUMBER_OF_SIMULATIONS)
        save_simulation_results(
            results_output_path,
            simulated_returns,
            current_ticker=current_ticker,
            agent_name=agent_name,
            dependencies=[trade_path, market_path],
        )
        return build_no_trade_summary(agent_name, NULL_MODEL_NAME)

    raw_returns = trade_df["return"].to_numpy(dtype=float)
    adjusted_returns = adjust_trade_returns(
        raw_returns=raw_returns,
        transaction_cost=TRANSACTION_COST,
        input_path=trade_path,
    )
    log_returns = convert_to_log_returns(adjusted_returns)
    actual_cumulative_return = calculate_cumulative_return_from_log_returns(log_returns)

    simulated_returns, null_model_name = simulate_agent_null_cumulative_returns(
        agent_name=agent_name,
        current_ticker=current_ticker,
        trade_df=trade_df,
        market_df=market_df,
        input_path=trade_path,
        simulation_count=NUMBER_OF_SIMULATIONS,
        rng=rng,
    )
    save_simulation_results(
        results_output_path,
        simulated_returns,
        current_ticker=current_ticker,
        agent_name=agent_name,
        dependencies=[trade_path, market_path],
    )
    return build_agent_summary(
        agent_name=agent_name,
        actual_cumulative_return=actual_cumulative_return,
        simulated_returns=simulated_returns,
        number_of_trades=len(trade_df),
        null_model_name=null_model_name,
    )


def main() -> None:
    """Run the full experimental suite for one ticker."""
    project_root = Path(__file__).resolve().parents[1]
    data_clean_dir = resolve_data_clean_dir(project_root)
    regimes_path = data_clean_dir / f"{ticker}_regimes.csv"
    regime_df = load_regime_data(regimes_path, _required_columns())
    regime_df.attrs["ticker"] = ticker
    market_df = load_market_data(regimes_path)

    seed_sequence = np.random.SeedSequence(SEED)
    child_sequences = seed_sequence.spawn(len(EXPERIMENTAL_AGENT_ORDER))

    summary_rows: list[dict[str, float | int | str | bool]] = []
    result_dependencies: list[Path] = []
    trade_dependencies: list[Path] = [regimes_path]

    print(f"Running experimental strategy suite for {ticker}...")
    for agent_name, child_sequence in zip(EXPERIMENTAL_AGENT_ORDER, child_sequences):
        trades_df = run_strategy(agent_name, regime_df, ticker=ticker)
        save_trade_outputs(
            current_ticker=ticker,
            agent_name=agent_name,
            trades_df=trades_df,
            output_dir=data_clean_dir,
        )

        trade_path = data_clean_dir / f"{ticker}_{agent_name}_trades.csv"
        metrics_output_path = data_clean_dir / f"{ticker}_{agent_name}_metrics.csv"
        create_and_save_metrics(input_path=trade_path, output_path=metrics_output_path)

        rng = build_random_generator(
            reproducible=REPRODUCIBLE,
            seed=int(child_sequence.generate_state(1, dtype=np.uint64)[0]),
        )
        summary_row = _build_monte_carlo_row(
            agent_name=agent_name,
            current_ticker=ticker,
            trade_path=trade_path,
            market_df=market_df,
            rng=rng,
            data_clean_dir=data_clean_dir,
        )
        summary_rows.append(summary_row)
        result_dependencies.append(data_clean_dir / f"{ticker}_{agent_name}_monte_carlo_results.csv")
        trade_dependencies.append(trade_path)

    summary_df = pd.DataFrame(summary_rows)
    summary_df["bh_adjusted_p_value"] = benjamini_hochberg_adjusted_p_values(
        summary_df["p_value"]
    )
    summary_output_path = data_clean_dir / f"{ticker}_experimental_monte_carlo_summary.csv"
    write_dataframe_artifact(
        summary_df,
        summary_output_path,
        producer="experimental_strategy_tester.main",
        current_ticker=ticker,
        dependencies=[*trade_dependencies, *result_dependencies],
        research_grade=NUMBER_OF_SIMULATIONS >= 5000,
        canonical_policy="always",
        parameters={
            "artifact_type": "experimental_monte_carlo_summary",
            "strategies": list(EXPERIMENTAL_AGENT_ORDER),
            "simulation_count": NUMBER_OF_SIMULATIONS,
        },
    )

    comparison_rows: list[dict[str, object]] = []
    for _, row in summary_df.iterrows():
        rcsi = float(row["actual_cumulative_return"] - row["mean_simulated_return"])
        std = float(row["std_simulated_return"])
        if std > 0:
            rcsi_z = float(rcsi / std)
        elif np.isclose(rcsi, 0.0):
            rcsi_z = 0.0
        else:
            rcsi_z = np.nan
        classification_bucket = classify_metrics(
            rcsi_z=rcsi_z,
            p_value=float(row["p_value"]),
            percentile=float(row["actual_percentile"]),
        )
        comparison_rows.append(
            {
                "agent": row["agent"],
                "strategy_name": format_strategy_name(str(row["agent"])),
                "number_of_trades": int(row["number_of_trades"]),
                "actual_cumulative_return": float(row["actual_cumulative_return"]),
                "RCSI": rcsi,
                "RCSI_z": rcsi_z,
                "p_value": float(row["p_value"]),
                "adjusted_p_value": float(row["bh_adjusted_p_value"]),
                "actual_percentile": float(row["actual_percentile"]),
                "final_classification": evidence_label(classification_bucket).lower(),
                "classification_bucket": classification_bucket,
                "null_model": str(row["null_model"]),
                "simulation_count": int(row["simulation_count"]),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        by=["RCSI_z", "actual_percentile"],
        ascending=[False, False],
    )
    comparison_output_path = data_clean_dir / f"{ticker}_experimental_strategy_comparison.csv"
    write_dataframe_artifact(
        comparison_df,
        comparison_output_path,
        producer="experimental_strategy_tester.main",
        current_ticker=ticker,
        dependencies=[summary_output_path],
        research_grade=NUMBER_OF_SIMULATIONS >= 5000,
        canonical_policy="always",
        parameters={
            "artifact_type": "experimental_strategy_comparison",
            "strategies": list(EXPERIMENTAL_AGENT_ORDER),
        },
    )

    print("\nExperimental strategy comparison:")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
