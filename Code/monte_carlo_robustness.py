"""Run repeated Monte Carlo evaluations and aggregate stability statistics."""

import os
import time

import numpy as np
import pandas as pd

from plot_config import data_clean_dir, format_agent_name, load_csv_checked

try:
    from monte_carlo import (
        NULL_MODEL_NAME,
        RELATIVE_STRENGTH_NULL_MODEL_NAME,
        TRANSACTION_COST,
        adjust_trade_returns,
        calculate_cumulative_return_from_log_returns,
        calculate_p_value,
        calculate_trade_durations,
        convert_to_log_returns,
        extract_position_value_fractions,
        interpret_p_value,
        load_market_data,
        load_trade_data,
        simulate_agent_null_cumulative_returns,
    )
    from research_metrics import (
        build_daily_strategy_curve,
        calculate_p_value_prominence,
        calculate_trade_level_return_ratio,
        summarize_daily_curve,
    )
    from strategy_config import AGENT_ORDER
    from strategy_artifact_utils import ensure_trade_file_exists
    from strategy_curve_utils import load_saved_strategy_curve
    from strategy_verdicts import (
        adjust_for_evaluation_power,
        classify_robustness_evidence,
        confidence_bucket_from_score,
        confidence_label,
        evidence_label,
        evaluation_power_label,
        evaluation_power_score,
        format_verdict_label,
        verdict_from_evidence_bucket,
    )
except ModuleNotFoundError:
    from Code.monte_carlo import (
        NULL_MODEL_NAME,
        RELATIVE_STRENGTH_NULL_MODEL_NAME,
        TRANSACTION_COST,
        adjust_trade_returns,
        calculate_cumulative_return_from_log_returns,
        calculate_p_value,
        calculate_trade_durations,
        convert_to_log_returns,
        extract_position_value_fractions,
        interpret_p_value,
        load_market_data,
        load_trade_data,
        simulate_agent_null_cumulative_returns,
    )
    from Code.research_metrics import (
        build_daily_strategy_curve,
        calculate_p_value_prominence,
        calculate_trade_level_return_ratio,
        summarize_daily_curve,
    )
    from Code.strategy_config import AGENT_ORDER
    from Code.strategy_artifact_utils import ensure_trade_file_exists
    from Code.strategy_curve_utils import load_saved_strategy_curve
    from Code.strategy_verdicts import (
        adjust_for_evaluation_power,
        classify_robustness_evidence,
        confidence_bucket_from_score,
        confidence_label,
        evidence_label,
        evaluation_power_label,
        evaluation_power_score,
        format_verdict_label,
        verdict_from_evidence_bucket,
    )


# Read the active ticker from the environment, or fall back to SPY.
ticker = os.environ.get("TICKER", "SPY")

# Monte Carlo settings. These are the main knobs you may want to change.
SIMULATIONS_PER_RUN = int(os.environ.get("ROBUSTNESS_SIMULATIONS_PER_RUN", "5000"))
OUTER_RUNS = int(os.environ.get("ROBUSTNESS_OUTER_RUNS", "100"))
BASE_SEED = int(os.environ.get("ROBUSTNESS_BASE_SEED", "100"))
PROGRESS_EVERY = max(1, int(os.environ.get("ROBUSTNESS_PROGRESS_EVERY", "1")))

# These thresholds control the plain-English stability interpretation.
STABLE_PERCENTILE_RANGE_MAX = 5.0
MODERATE_PERCENTILE_RANGE_MAX = 15.0
STABLE_RCSI_RELATIVE_STD_MAX = 0.10
MODERATE_RCSI_RELATIVE_STD_MAX = 0.25
RCSI_RELATIVE_STD_FLOOR = 0.05

SUMMARY_METRICS = [
    "actual_cumulative_return",
    "annualized_return",
    "annualized_sharpe",
    "trade_level_return_ratio",
    "max_drawdown",
    "number_of_trades",
    "p_value",
    "p_value_prominence",
    "actual_percentile",
    "RCSI",
    "RCSI_z",
    "median_simulated_return",
]


def calculate_max_drawdown_from_log_returns(log_returns: np.ndarray) -> float:
    """Calculate max drawdown from a sequence of log returns."""
    wealth_index = np.exp(np.cumsum(log_returns))
    rolling_peak = np.maximum.accumulate(wealth_index)
    drawdowns = (wealth_index / rolling_peak) - 1.0
    return float(np.min(drawdowns))


def series_std_or_nan(series: pd.Series) -> float:
    """Return a population standard deviation or NaN for an empty numeric slice."""
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


def calculate_annualized_return(
    log_returns: np.ndarray,
    trade_df: pd.DataFrame,
    market_df: pd.DataFrame,
    input_path,
) -> float:
    """Annualize the realized compounded return across the strategy's active span."""
    if len(log_returns) == 0:
        return 0.0

    date_to_index = pd.Series(market_df.index.to_numpy(), index=market_df["Date"])
    first_entry_index = trade_df["entry_date"].min()
    last_exit_index = trade_df["exit_date"].max()
    start_bar = date_to_index.get(first_entry_index)
    end_bar = date_to_index.get(last_exit_index)

    if pd.isna(start_bar) or pd.isna(end_bar):
        raise ValueError(
            f"{input_path} contains trade dates that do not align with the market data."
        )

    trading_bars = int(end_bar) - int(start_bar)
    if trading_bars <= 0:
        return calculate_cumulative_return_from_log_returns(log_returns)

    annualized_log_return = float(log_returns.sum()) * (252.0 / trading_bars)
    return float(np.expm1(annualized_log_return))


def load_agent_inputs() -> tuple[pd.DataFrame, np.ndarray, dict[str, dict[str, object]]]:
    """Load each strategy's realized trades and matched duration profile."""
    market_path = data_clean_dir() / f"{ticker}_regimes.csv"
    market_df = load_market_data(market_path)
    market_curve_df = load_csv_checked(
        market_path,
        required_columns=["Date", "Close"],
    )
    market_curve_df["Date"] = pd.to_datetime(market_curve_df["Date"], errors="coerce")
    market_curve_df["Close"] = pd.to_numeric(market_curve_df["Close"], errors="coerce")
    market_curve_df = market_curve_df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(
        drop=True
    )
    open_prices = market_df["Open"].to_numpy(dtype=float)
    agent_inputs: dict[str, dict[str, object]] = {}

    for agent_name in AGENT_ORDER:
        input_path = ensure_trade_file_exists(ticker, agent_name)
        trade_df = load_trade_data(input_path, allow_empty=True)
        if trade_df.empty:
            adjusted_returns = np.array([], dtype=float)
            log_returns = np.array([], dtype=float)
            actual_cumulative_return = 0.0
            durations = np.array([], dtype=int)
            position_value_fractions = np.array([], dtype=float)
            annualized_return = 0.0
        else:
            raw_returns = trade_df["return"].to_numpy(dtype=float)
            adjusted_returns = adjust_trade_returns(
                raw_returns=raw_returns,
                transaction_cost=TRANSACTION_COST,
                input_path=input_path,
            )
            log_returns = convert_to_log_returns(adjusted_returns)
            actual_cumulative_return = calculate_cumulative_return_from_log_returns(log_returns)
            durations = calculate_trade_durations(
                trade_df=trade_df,
                market_df=market_df,
                input_path=input_path,
            )
            position_value_fractions = extract_position_value_fractions(trade_df)
            annualized_return = calculate_annualized_return(
                log_returns=log_returns,
                trade_df=trade_df,
                market_df=market_df,
                input_path=input_path,
            )
        daily_curve_df = load_saved_strategy_curve(ticker, agent_name)
        if daily_curve_df is None:
            daily_curve_df = build_daily_strategy_curve(trade_df, market_curve_df)
        curve_summary = summarize_daily_curve(daily_curve_df)
        trade_level_return_ratio = calculate_trade_level_return_ratio(adjusted_returns)
        max_drawdown = float(curve_summary["max_drawdown"])

        agent_inputs[agent_name] = {
            "actual_cumulative_return": actual_cumulative_return,
            "annualized_return": annualized_return,
            "annualized_sharpe": float(curve_summary["annualized_sharpe"]),
            "trade_level_return_ratio": trade_level_return_ratio,
            "max_drawdown": max_drawdown,
            "number_of_trades": len(trade_df),
            "durations": durations,
            "position_value_fractions": position_value_fractions,
            "null_model": (
                RELATIVE_STRENGTH_NULL_MODEL_NAME
                if agent_name == "momentum_relative_strength"
                else NULL_MODEL_NAME
            ),
        }

    return market_df, open_prices, agent_inputs


def build_run_row(
    agent_name: str,
    outer_run: int,
    seed_used: int,
    actual_metrics: dict[str, float | int | np.ndarray],
    simulated_returns: pd.Series,
    number_of_trades: int,
    null_model_name: str,
) -> dict[str, float | int | str]:
    """Create one per-seed result row for one strategy."""
    simulated_array = simulated_returns.to_numpy(dtype=float)
    actual_cumulative_return = float(actual_metrics["actual_cumulative_return"])
    mean_simulated_return = float(np.mean(simulated_array))
    std_simulated_return = float(np.std(simulated_array, ddof=0))
    median_simulated_return = float(np.median(simulated_array))
    actual_percentile = float((simulated_array <= actual_cumulative_return).mean() * 100)
    p_value = calculate_p_value(simulated_array, actual_cumulative_return)
    p_value_prominence = calculate_p_value_prominence(p_value)
    rcsi = actual_cumulative_return - median_simulated_return

    rcsi_z = np.nan
    if std_simulated_return > 0:
        rcsi_z = (actual_cumulative_return - mean_simulated_return) / std_simulated_return

    return {
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
        "actual_percentile": actual_percentile,
        "p_value": p_value,
        "p_value_prominence": p_value_prominence,
        "RCSI": rcsi,
        "RCSI_z": rcsi_z,
        "significant_run": int(p_value <= 0.05),
        "outperform_null_median": int(actual_cumulative_return > median_simulated_return),
        "lower_5pct": float(np.percentile(simulated_array, 5)),
        "upper_95pct": float(np.percentile(simulated_array, 95)),
        "number_of_trades": number_of_trades,
        "transaction_cost": TRANSACTION_COST,
        "simulations_per_run": SIMULATIONS_PER_RUN,
        "null_model": null_model_name,
    }


def build_no_trade_run_row(
    agent_name: str,
    outer_run: int,
    seed_used: int,
    actual_metrics: dict[str, float | int | np.ndarray],
    null_model_name: str,
) -> dict[str, float | int | str]:
    """Create one placeholder repeated-run row for a no-trade strategy."""
    return {
        "agent": agent_name,
        "outer_run": outer_run,
        "seed_used": seed_used,
        "actual_cumulative_return": float(actual_metrics["actual_cumulative_return"]),
        "annualized_return": float(actual_metrics["annualized_return"]),
        "annualized_sharpe": float(actual_metrics["annualized_sharpe"]),
        "trade_level_return_ratio": float(actual_metrics["trade_level_return_ratio"]),
        "max_drawdown": float(actual_metrics["max_drawdown"]),
        "median_simulated_return": 0.0,
        "mean_simulated_return": 0.0,
        "std_simulated_return": 0.0,
        "actual_percentile": np.nan,
        "p_value": np.nan,
        "p_value_prominence": np.nan,
        "RCSI": np.nan,
        "RCSI_z": np.nan,
        "significant_run": 0,
        "outperform_null_median": 0,
        "lower_5pct": 0.0,
        "upper_95pct": 0.0,
        "number_of_trades": 0,
        "transaction_cost": TRANSACTION_COST,
        "simulations_per_run": SIMULATIONS_PER_RUN,
        "null_model": null_model_name,
    }


def classify_stability(summary_row: pd.Series) -> str:
    """Translate numeric variability into a simple plain-English label."""
    mean_number_of_trades = pd.to_numeric(
        pd.Series([summary_row.get("mean_number_of_trades", np.nan)]),
        errors="coerce",
    ).iloc[0]
    if pd.notna(mean_number_of_trades) and float(mean_number_of_trades) <= 0:
        return "not_applicable"

    percentile_range = float(summary_row["max_percentile"] - summary_row["min_percentile"])
    rcsi_scale = max(abs(float(summary_row["mean_RCSI"])), RCSI_RELATIVE_STD_FLOOR)
    rcsi_relative_std = float(summary_row["std_RCSI"]) / rcsi_scale

    if (
        percentile_range <= STABLE_PERCENTILE_RANGE_MAX
        and rcsi_relative_std <= STABLE_RCSI_RELATIVE_STD_MAX
    ):
        return "stable"

    if (
        percentile_range <= MODERATE_PERCENTILE_RANGE_MAX
        and rcsi_relative_std <= MODERATE_RCSI_RELATIVE_STD_MAX
    ):
        return "moderately variable"

    return "unstable"


def aggregate_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-seed results into one summary row per strategy."""
    grouped = runs_df.groupby("agent", as_index=False)
    summary_df = grouped[["agent"]].first()

    for metric_name in SUMMARY_METRICS:
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

    summary_df["proportion_significant"] = grouped["significant_run"].mean()[
        "significant_run"
    ].to_numpy(dtype=float)
    summary_df["proportion_outperforming_null_median"] = grouped[
        "outperform_null_median"
    ].mean()["outperform_null_median"].to_numpy(dtype=float)

    summary_df["mean_percentile"] = summary_df["mean_actual_percentile"]
    summary_df["median_percentile"] = summary_df["median_actual_percentile"]
    summary_df["std_percentile"] = summary_df["std_actual_percentile"]
    summary_df["min_percentile"] = summary_df["min_actual_percentile"]
    summary_df["max_percentile"] = summary_df["max_actual_percentile"]
    summary_df["p25_percentile"] = summary_df["p25_actual_percentile"]
    summary_df["p75_percentile"] = summary_df["p75_actual_percentile"]

    summary_df["number_of_outer_runs"] = OUTER_RUNS
    summary_df["simulations_per_run"] = SIMULATIONS_PER_RUN
    summary_df["transaction_cost"] = TRANSACTION_COST
    summary_df["p_value_interpretation"] = summary_df["mean_p_value"].apply(
        lambda value: interpret_p_value(value) if pd.notna(value) else "no trades"
    )
    summary_df["null_model"] = grouped["null_model"].first()["null_model"].to_numpy(dtype=str)
    summary_df["stability_classification"] = summary_df.apply(classify_stability, axis=1)
    evidence_and_confidence = summary_df.apply(
        lambda row: classify_robustness_evidence(
            p_value=row["mean_p_value"],
            rcsi=row["mean_RCSI"],
            rcsi_z=row["mean_RCSI_z"],
            percentile=row["mean_percentile"],
            proportion_significant=row["proportion_significant"],
            proportion_outperforming_null_median=row["proportion_outperforming_null_median"],
            stability_classification=str(row["stability_classification"]),
        ),
        axis=1,
    )
    summary_df["evidence_bucket"] = evidence_and_confidence.map(lambda value: value[0])
    summary_df["confidence_score"] = evidence_and_confidence.map(lambda value: value[1])
    summary_df["evaluation_power_score"] = summary_df.apply(
        lambda row: evaluation_power_score(
            number_of_outer_runs=OUTER_RUNS,
            simulations_per_run=SIMULATIONS_PER_RUN,
        ),
        axis=1,
    )
    adjusted_evidence = summary_df.apply(
        lambda row: adjust_for_evaluation_power(
            evidence_bucket=row["evidence_bucket"],
            confidence_score=row["confidence_score"],
            power_score=row["evaluation_power_score"],
        ),
        axis=1,
    )
    summary_df["evidence_bucket"] = adjusted_evidence.map(lambda value: value[0])
    summary_df["confidence_score"] = adjusted_evidence.map(lambda value: value[1])
    no_trade_mask = summary_df["mean_number_of_trades"].fillna(np.nan).le(0)
    summary_df.loc[no_trade_mask, "evidence_bucket"] = "no_trades"
    summary_df.loc[no_trade_mask, "confidence_score"] = np.nan
    summary_df["confidence_bucket"] = summary_df["confidence_score"].apply(confidence_bucket_from_score)
    summary_df.loc[no_trade_mask, "confidence_bucket"] = "not_applicable"
    summary_df["evaluation_power_label"] = summary_df["evaluation_power_score"].apply(
        evaluation_power_label
    )
    summary_df.loc[no_trade_mask, "evaluation_power_score"] = np.nan
    summary_df.loc[no_trade_mask, "evaluation_power_label"] = "Not Applicable"
    summary_df["evidence_label"] = summary_df["evidence_bucket"].apply(evidence_label)
    summary_df["confidence_label"] = summary_df["confidence_bucket"].apply(confidence_label)
    summary_df["skill_luck_verdict"] = summary_df["evidence_bucket"].apply(verdict_from_evidence_bucket)
    summary_df["verdict_label"] = summary_df["skill_luck_verdict"].apply(format_verdict_label)
    summary_df["final_classification"] = summary_df["evidence_label"].str.lower()

    summary_df["agent"] = pd.Categorical(
        summary_df["agent"],
        categories=AGENT_ORDER,
        ordered=True,
    )
    summary_df = summary_df.sort_values("agent").reset_index(drop=True)
    summary_df["agent"] = summary_df["agent"].astype(str)

    return summary_df


def print_agent_summary(summary_row: pd.Series) -> None:
    """Print a compact interpretation for one strategy."""
    print(f"\nAgent: {format_agent_name(summary_row['agent'])}")
    mean_number_of_trades = pd.to_numeric(
        pd.Series([summary_row.get("mean_number_of_trades", np.nan)]),
        errors="coerce",
    ).iloc[0]
    if pd.notna(mean_number_of_trades) and float(mean_number_of_trades) <= 0:
        print("No completed trades were generated across the robustness evaluation.")
        print("Repeated-run skill-versus-luck inference is not applicable for this strategy.")
        print(f"Evaluation power: {summary_row['evaluation_power_label']}")
        print(f"Evidence: {summary_row['evidence_label']}")
        print(f"Confidence: {summary_row['confidence_label']}")
        print(f"Final classification: {summary_row['final_classification']}")
        print(f"Skill vs luck verdict: {summary_row['verdict_label']}")
        print(f"Null model: {summary_row['null_model']}")
        return

    print(f"Mean cumulative return: {summary_row['mean_actual_cumulative_return']:.6f}")
    print(f"Mean annualized return: {summary_row['mean_annualized_return']:.6f}")
    print(f"Mean annualized Sharpe: {summary_row['mean_annualized_sharpe']:.6f}")
    print(
        "Mean trade-level return ratio: "
        f"{summary_row['mean_trade_level_return_ratio']:.6f}"
    )
    print(f"Mean max drawdown: {summary_row['mean_max_drawdown']:.6f}")
    print(f"Mean percentile across runs: {summary_row['mean_percentile']:.2f}")
    print(f"Std percentile across runs: {summary_row['std_percentile']:.2f}")
    print(f"Mean p-value across runs: {summary_row['mean_p_value']:.6f}")
    print(f"Mean p-value prominence across runs: {summary_row['mean_p_value_prominence']:.6f}")
    print(f"Std p-value across runs: {summary_row['std_p_value']:.6f}")
    print(f"Mean RCSI across runs: {summary_row['mean_RCSI']:.6f}")
    print(f"Std RCSI across runs: {summary_row['std_RCSI']:.6f}")
    print(
        "Range of RCSI: "
        f"{summary_row['min_RCSI']:.6f} to {summary_row['max_RCSI']:.6f}"
    )
    print(
        "Proportion of runs with p <= 0.05: "
        f"{summary_row['proportion_significant']:.2%}"
    )
    print(
        "Proportion of runs above null median: "
        f"{summary_row['proportion_outperforming_null_median']:.2%}"
    )
    print(f"P-value evidence: {summary_row['p_value_interpretation']}")
    print(f"Stability: {summary_row['stability_classification']}")
    print(
        "Evaluation power: "
        f"{summary_row['evaluation_power_label']} "
        f"(score={summary_row['evaluation_power_score']:.2f})"
    )
    print(f"Evidence: {summary_row['evidence_label']}")
    print(f"Confidence: {summary_row['confidence_label']}")
    print(f"Final classification: {summary_row['final_classification']}")
    print(
        "Skill vs luck verdict: "
        f"{summary_row['verdict_label']}"
    )
    print(f"Null model: {summary_row['null_model']}")


def main() -> None:
    """Run the outer-loop Monte Carlo robustness test across many seeds."""
    runs_output_path = data_clean_dir() / f"{ticker}_monte_carlo_robustness_runs.csv"
    summary_output_path = data_clean_dir() / f"{ticker}_monte_carlo_robustness_summary.csv"

    _, open_prices, agent_inputs = load_agent_inputs()
    run_rows: list[dict[str, float | int | str]] = []
    start_time = time.perf_counter()

    for outer_run in range(1, OUTER_RUNS + 1):
        seed_used = BASE_SEED + outer_run
        if outer_run == 1 or outer_run % PROGRESS_EVERY == 0:
            elapsed_seconds = time.perf_counter() - start_time
            print(
                f"[robustness] ticker={ticker} run {outer_run}/{OUTER_RUNS} "
                f"(seed={seed_used}) elapsed={elapsed_seconds:.1f}s",
                flush=True,
            )
        seed_sequence = np.random.SeedSequence(seed_used)
        child_sequences = seed_sequence.spawn(len(AGENT_ORDER))

        for agent_name, child_sequence in zip(AGENT_ORDER, child_sequences):
            agent_input = agent_inputs[agent_name]
            null_model_name = str(agent_input["null_model"])
            if int(agent_input["number_of_trades"]) == 0:
                run_rows.append(
                    build_no_trade_run_row(
                        agent_name=agent_name,
                        outer_run=outer_run,
                        seed_used=seed_used,
                        actual_metrics=agent_input,
                        null_model_name=null_model_name,
                    )
                )
                continue
            rng = np.random.default_rng(child_sequence)
            simulated_returns, null_model_name = simulate_agent_null_cumulative_returns(
                agent_name=agent_name,
                current_ticker=ticker,
                single_asset_open_prices=open_prices,
                durations=agent_input["durations"],
                position_value_fractions=agent_input["position_value_fractions"],
                simulation_count=SIMULATIONS_PER_RUN,
                rng=rng,
            )

            run_rows.append(
                build_run_row(
                    agent_name=agent_name,
                    outer_run=outer_run,
                    seed_used=seed_used,
                    actual_metrics=agent_input,
                    simulated_returns=simulated_returns,
                    number_of_trades=int(agent_input["number_of_trades"]),
                    null_model_name=null_model_name,
                )
            )

    runs_df = pd.DataFrame(
        run_rows,
        columns=[
            "agent",
            "outer_run",
            "seed_used",
            "actual_cumulative_return",
            "annualized_return",
            "annualized_sharpe",
            "trade_level_return_ratio",
            "max_drawdown",
            "median_simulated_return",
            "mean_simulated_return",
            "std_simulated_return",
            "actual_percentile",
            "p_value",
            "p_value_prominence",
            "RCSI",
            "RCSI_z",
            "significant_run",
            "outperform_null_median",
            "lower_5pct",
            "upper_95pct",
            "number_of_trades",
            "transaction_cost",
            "simulations_per_run",
            "null_model",
        ],
    )
    runs_df.to_csv(runs_output_path, index=False)

    summary_df = aggregate_runs(runs_df)
    summary_df.to_csv(summary_output_path, index=False)

    total_elapsed_seconds = time.perf_counter() - start_time
    print(
        f"[robustness] completed {OUTER_RUNS} runs in {total_elapsed_seconds:.1f}s",
        flush=True,
    )

    for _, summary_row in summary_df.iterrows():
        print_agent_summary(summary_row)


if __name__ == "__main__":
    main()
