"""Run repeated multi-asset walk-forward evaluations across rolling test windows."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from plot_config import data_clean_dir, format_agent_name, load_csv_checked

try:
    from monte_carlo import (
        NULL_MODEL_NAME,
        TRANSACTION_COST,
        adjust_trade_returns,
        calculate_p_value,
        calculate_trade_durations,
        extract_position_value_fractions,
        load_market_data,
        simulate_random_timing_cumulative_returns,
    )
    from research_metrics import (
        build_daily_strategy_curve,
        calculate_p_value_prominence,
        calculate_trade_level_return_ratio,
        summarize_daily_curve,
    )
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
        evidence_label,
        evaluation_power_label,
        evaluation_power_score,
        format_verdict_label,
        verdict_from_evidence_bucket,
    )
except ModuleNotFoundError:
    from Code.monte_carlo import (
        NULL_MODEL_NAME,
        TRANSACTION_COST,
        adjust_trade_returns,
        calculate_p_value,
        calculate_trade_durations,
        extract_position_value_fractions,
        load_market_data,
        simulate_random_timing_cumulative_returns,
    )
    from Code.research_metrics import (
        build_daily_strategy_curve,
        calculate_p_value_prominence,
        calculate_trade_level_return_ratio,
        summarize_daily_curve,
    )
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
        evidence_label,
        evaluation_power_label,
        evaluation_power_score,
        format_verdict_label,
        verdict_from_evidence_bucket,
    )


PREREQUISITE_SCRIPTS = [
    "data_loader.py",
    "features.py",
    "regimes.py",
    "trend_agent.py",
    "mean_reversion_agent.py",
    "random_agent.py",
    "momentum_agent.py",
    "breakout_agent.py",
]
DEFAULT_TICKERS = [
    "SPY",
    "QQQ",
    "AAPL",
    "VOO",
    "NVDA",
    "TSM",
    "MRNA",
    "NVAX",
    "BTC-USD",
    "NQ=F",
    "ES=F",
]
TICKERS = [
    ticker.strip().upper()
    for ticker in os.environ.get("WALK_FORWARD_TICKERS", ",".join(DEFAULT_TICKERS)).split(",")
    if ticker.strip()
]
TEST_BARS = int(os.environ.get("WALK_FORWARD_TEST_BARS", "504"))
STEP_BARS = int(os.environ.get("WALK_FORWARD_STEP_BARS", "252"))
MIN_TRADES_PER_PANEL = int(os.environ.get("WALK_FORWARD_MIN_TRADES", "5"))
SIMULATIONS_PER_RUN = int(os.environ.get("WALK_FORWARD_SIMULATIONS_PER_RUN", "1000"))
OUTER_RUNS = int(os.environ.get("WALK_FORWARD_OUTER_RUNS", "50"))
BASE_SEED = int(os.environ.get("WALK_FORWARD_BASE_SEED", "5000"))
PROGRESS_EVERY = max(1, int(os.environ.get("WALK_FORWARD_PROGRESS_EVERY", "5")))
REFRESH_DATA = os.environ.get("WALK_FORWARD_REFRESH_DATA", "1") == "1"
RCSI_Z_STABILITY_FLOOR = 0.10


def code_dir() -> Path:
    """Return the folder that contains the project scripts."""
    return Path(__file__).resolve().parent


def required_market_columns() -> list[str]:
    """Return the minimum columns needed from the regime-tagged market file."""
    return [
        "Date",
        "Open",
        "Close",
        "ma_20",
        "ma_50",
        "price_std_20",
        "zscore_20",
        "avg_volume_20",
        "regime",
        MOMENTUM_RETURN_COLUMN,
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOW_COLUMN,
    ]


def ensure_csv_columns(input_path: Path, required_columns: list[str]) -> bool:
    """Return True when a CSV exists and contains the columns we need."""
    if not input_path.exists():
        return False

    try:
        df = pd.read_csv(input_path, nrows=1)
    except Exception:
        return False

    return all(column in df.columns for column in required_columns)


def run_script_for_ticker(script_name: str, ticker: str) -> None:
    """Execute one project script for a chosen ticker."""
    env = os.environ.copy()
    env["TICKER"] = ticker
    env["SHOW_PLOTS"] = "0"
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(code_dir().parent / ".matplotlib")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [sys.executable, script_name],
        cwd=code_dir(),
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def ensure_ticker_inputs(ticker: str) -> None:
    """Refresh the files needed for walk-forward analysis when required."""
    market_path = data_clean_dir() / f"{ticker}_regimes.csv"
    has_market = ensure_csv_columns(market_path, required_market_columns())

    if not REFRESH_DATA and has_market:
        return

    print(f"[walk-forward] refreshing inputs for {ticker}", flush=True)
    for script_name in PREREQUISITE_SCRIPTS:
        run_script_for_ticker(script_name, ticker)


def calculate_fold_annualized_return(cumulative_return: float, test_bar_count: int) -> float:
    """Annualize a fold return across the full out-of-sample window length."""
    if test_bar_count <= 0:
        return 0.0
    if cumulative_return <= -1.0:
        return -1.0
    return float((1.0 + cumulative_return) ** (252.0 / test_bar_count) - 1.0)


def load_fold_market_data(ticker: str) -> pd.DataFrame:
    """Load the regime-tagged market data used to build rolling folds."""
    market_path = data_clean_dir() / f"{ticker}_regimes.csv"
    market_df = load_csv_checked(
        market_path,
        required_columns=required_market_columns(),
    )
    market_df["Date"] = pd.to_datetime(market_df["Date"], errors="coerce")
    numeric_columns = [column for column in required_market_columns() if column not in {"Date", "regime"}]
    for column in numeric_columns:
        market_df[column] = pd.to_numeric(market_df[column], errors="coerce")
    market_df["regime"] = market_df["regime"].astype(str)
    market_df = market_df.dropna(subset=required_market_columns()).sort_values("Date").reset_index(drop=True)
    return market_df


def build_folds(market_df: pd.DataFrame) -> list[dict[str, object]]:
    """Create rolling out-of-sample folds from market data."""
    folds: list[dict[str, object]] = []

    if len(market_df) < TEST_BARS:
        return folds

    fold_number = 0
    for start_index in range(0, len(market_df) - TEST_BARS + 1, STEP_BARS):
        end_index = start_index + TEST_BARS - 1
        fold_number += 1
        fold_market_df = market_df.iloc[start_index : end_index + 1].reset_index(drop=True)
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


def actual_random_decision_seed(ticker: str, fold_id: str) -> int:
    """Create a deterministic random-strategy decision seed for one panel."""
    ticker_component = sum((index + 1) * ord(character) for index, character in enumerate(ticker))
    fold_component = sum((index + 1) * ord(character) for index, character in enumerate(fold_id))
    return BASE_SEED + (97 * ticker_component) + (13 * fold_component)


def build_fold_actual_metrics(
    ticker: str,
    fold_id: str,
    agent_name: str,
    fold_market_df: pd.DataFrame,
    test_bar_count: int,
) -> dict[str, object] | None:
    """Rerun one strategy from the fold start with fresh capital and compute metrics."""
    random_seed = actual_random_decision_seed(ticker, fold_id) if agent_name == "random" else None
    trade_df = run_strategy(agent_name, fold_market_df, random_decision_seed=random_seed)
    if len(trade_df) < MIN_TRADES_PER_PANEL:
        return None

    input_path = Path(f"{ticker}_{agent_name}_{fold_id}")
    raw_returns = trade_df["return"].to_numpy(dtype=float)
    adjusted_returns = adjust_trade_returns(
        raw_returns=raw_returns,
        transaction_cost=TRANSACTION_COST,
        input_path=input_path,
    )
    durations = calculate_trade_durations(
        trade_df=trade_df,
        market_df=fold_market_df[["Date", "Open"]].copy(),
        input_path=input_path,
    )
    position_value_fractions = extract_position_value_fractions(trade_df)
    daily_curve_df = build_daily_strategy_curve(
        trade_df=trade_df,
        market_df=fold_market_df[["Date", "Close"]].copy(),
    )
    curve_summary = summarize_daily_curve(daily_curve_df)
    cumulative_return = float(curve_summary["cumulative_return"])

    return {
        "trade_df": trade_df,
        "actual_cumulative_return": cumulative_return,
        "annualized_return": calculate_fold_annualized_return(cumulative_return, test_bar_count),
        "annualized_sharpe": float(curve_summary["annualized_sharpe"]),
        "trade_level_return_ratio": calculate_trade_level_return_ratio(adjusted_returns),
        "max_drawdown": float(curve_summary["max_drawdown"]),
        "number_of_trades": int(len(trade_df)),
        "durations": durations,
        "position_value_fractions": position_value_fractions,
    }


def build_panel_run_row(
    ticker: str,
    fold_id: str,
    fold_start: pd.Timestamp,
    fold_end: pd.Timestamp,
    agent_name: str,
    outer_run: int,
    seed_used: int,
    actual_metrics: dict[str, object],
    test_bar_count: int,
    simulated_returns: pd.Series,
) -> dict[str, float | int | str]:
    """Build one repeated-run row for one ticker/fold/strategy panel."""
    actual_cumulative_return = float(actual_metrics["actual_cumulative_return"])
    simulated_array = simulated_returns.to_numpy(dtype=float)
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
        "actual_percentile": actual_percentile,
        "p_value": p_value,
        "p_value_prominence": p_value_prominence,
        "RCSI": rcsi,
        "RCSI_z": rcsi_z,
        "significant_run": int(p_value <= 0.05),
        "outperform_null_median": int(actual_cumulative_return > median_simulated_return),
        "number_of_trades": int(actual_metrics["number_of_trades"]),
        "test_bar_count": int(test_bar_count),
        "transaction_cost": TRANSACTION_COST,
        "simulations_per_run": SIMULATIONS_PER_RUN,
        "null_model": NULL_MODEL_NAME,
    }


def aggregate_panel_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated runs for each ticker/fold/strategy panel."""
    group_columns = ["ticker", "fold_id", "fold_start", "fold_end", "agent"]
    grouped = runs_df.groupby(group_columns, as_index=False)
    panel_df = grouped[group_columns].first()

    metrics = [
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
    for metric_name in metrics:
        metric_summary = grouped[metric_name].agg(
            mean="mean",
            median="median",
            std=lambda series: float(np.std(series.to_numpy(dtype=float), ddof=0)),
            min="min",
            max="max",
            p25=lambda series: float(np.percentile(series.to_numpy(dtype=float), 25)),
            p75=lambda series: float(np.percentile(series.to_numpy(dtype=float), 75)),
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
        panel_df = panel_df.merge(metric_summary, on=group_columns, how="left")

    panel_df["mean_number_of_trades"] = grouped["number_of_trades"].mean()["number_of_trades"].to_numpy(
        dtype=float
    )
    panel_df["proportion_significant_runs"] = grouped["significant_run"].mean()[
        "significant_run"
    ].to_numpy(dtype=float)
    panel_df["proportion_outperforming_null_runs"] = grouped["outperform_null_median"].mean()[
        "outperform_null_median"
    ].to_numpy(dtype=float)
    panel_df["number_of_outer_runs"] = OUTER_RUNS
    panel_df["simulations_per_run"] = SIMULATIONS_PER_RUN
    panel_df["transaction_cost"] = TRANSACTION_COST
    panel_df["null_model"] = NULL_MODEL_NAME

    panel_df["agent"] = pd.Categorical(panel_df["agent"], categories=AGENT_ORDER, ordered=True)
    panel_df = panel_df.sort_values(["ticker", "fold_start", "agent"]).reset_index(drop=True)
    panel_df["agent"] = panel_df["agent"].astype(str)
    return panel_df


def aggregate_agent_summary(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate panel summaries into one overall repeated-evaluation table per agent."""
    grouped = panel_df.groupby("agent", as_index=False)
    summary_df = grouped[["agent"]].first()

    panel_metrics = [
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
    for metric_name in panel_metrics:
        metric_summary = grouped[metric_name].agg(
            mean="mean",
            median="median",
            std=lambda series: float(np.std(series.to_numpy(dtype=float), ddof=0)),
            min="min",
            max="max",
            p25=lambda series: float(np.percentile(series.to_numpy(dtype=float), 25)),
            p75=lambda series: float(np.percentile(series.to_numpy(dtype=float), 75)),
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
    summary_df["mean_proportion_significant_runs"] = grouped["proportion_significant_runs"].mean()[
        "proportion_significant_runs"
    ].to_numpy(dtype=float)
    summary_df["mean_proportion_outperforming_null_runs"] = grouped[
        "proportion_outperforming_null_runs"
    ].mean()["proportion_outperforming_null_runs"].to_numpy(dtype=float)
    summary_df["proportion_significant_panels"] = (
        grouped["mean_p_value"].apply(lambda series: float((series <= 0.05).mean()))["mean_p_value"]
        .to_numpy(dtype=float)
    )
    summary_df["proportion_positive_rcsi_z_panels"] = (
        grouped["mean_RCSI_z"].apply(lambda series: float((series > 0).mean()))["mean_RCSI_z"]
        .to_numpy(dtype=float)
    )
    summary_df["stability_classification"] = summary_df.apply(classify_stability, axis=1)

    evidence_and_confidence = summary_df.apply(
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
    summary_df["evidence_bucket"] = evidence_and_confidence.map(lambda value: value[0])
    summary_df["confidence_score"] = evidence_and_confidence.map(lambda value: value[1])
    summary_df["evaluation_power_score"] = summary_df.apply(
        lambda _: evaluation_power_score(
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
    summary_df["confidence_bucket"] = summary_df["confidence_score"].apply(confidence_bucket_from_score)
    summary_df["evaluation_power_label"] = summary_df["evaluation_power_score"].apply(
        evaluation_power_label
    )
    summary_df["evidence_label"] = summary_df["evidence_bucket"].apply(evidence_label)
    summary_df["confidence_label"] = summary_df["confidence_bucket"].apply(confidence_label)
    summary_df["skill_luck_verdict"] = summary_df["evidence_bucket"].apply(verdict_from_evidence_bucket)
    summary_df["verdict_label"] = summary_df["skill_luck_verdict"].apply(format_verdict_label)
    summary_df["final_classification"] = summary_df["evidence_label"].str.lower()
    summary_df["number_of_outer_runs"] = OUTER_RUNS
    summary_df["simulations_per_run"] = SIMULATIONS_PER_RUN
    summary_df["test_bars"] = TEST_BARS
    summary_df["step_bars"] = STEP_BARS
    summary_df["min_trades_per_panel"] = MIN_TRADES_PER_PANEL
    summary_df["transaction_cost"] = TRANSACTION_COST
    summary_df["null_model"] = NULL_MODEL_NAME

    summary_df["agent"] = pd.Categorical(summary_df["agent"], categories=AGENT_ORDER, ordered=True)
    summary_df = summary_df.sort_values("agent").reset_index(drop=True)
    summary_df["agent"] = summary_df["agent"].astype(str)
    return summary_df


def print_agent_summary(summary_row: pd.Series) -> None:
    """Print a compact final summary for one strategy."""
    print(f"\nAgent: {format_agent_name(summary_row['agent'])}")
    print(f"Panels evaluated: {int(summary_row['panel_count'])}")
    print(f"Mean cumulative return across panels: {summary_row['mean_mean_actual_cumulative_return']:.6f}")
    print(f"Mean annualized return across panels: {summary_row['mean_mean_annualized_return']:.6f}")
    print(f"Mean annualized Sharpe across panels: {summary_row['mean_mean_annualized_sharpe']:.6f}")
    print(
        "Mean trade-level return ratio across panels: "
        f"{summary_row['mean_mean_trade_level_return_ratio']:.6f}"
    )
    print(f"Mean max drawdown across panels: {summary_row['mean_mean_max_drawdown']:.6f}")
    print(f"Mean p-value across panels: {summary_row['mean_mean_p_value']:.6f}")
    print(
        "Mean p-value prominence across panels: "
        f"{summary_row['mean_mean_p_value_prominence']:.6f}"
    )
    print(f"Mean percentile across panels: {summary_row['mean_mean_actual_percentile']:.2f}")
    print(f"Mean RCSI_z across panels: {summary_row['mean_mean_RCSI_z']:.6f}")
    print(
        "Average proportion of significant runs within panels: "
        f"{summary_row['mean_proportion_significant_runs']:.2%}"
    )
    print(
        "Average proportion of runs above null median within panels: "
        f"{summary_row['mean_proportion_outperforming_null_runs']:.2%}"
    )
    print(
        "Share of panels with mean p-value <= 0.05: "
        f"{summary_row['proportion_significant_panels']:.2%}"
    )
    print(
        "Share of panels with positive mean RCSI_z: "
        f"{summary_row['proportion_positive_rcsi_z_panels']:.2%}"
    )
    print(f"Stability: {summary_row['stability_classification']}")
    print(
        "Evaluation power: "
        f"{summary_row['evaluation_power_label']} "
        f"(score={summary_row['evaluation_power_score']:.2f})"
    )
    print(f"Evidence: {summary_row['evidence_label']}")
    print(f"Confidence: {summary_row['confidence_label']}")
    print(f"Final classification: {summary_row['final_classification']}")
    print(f"Skill vs luck verdict: {summary_row['verdict_label']}")


def main() -> None:
    """Run repeated multi-asset walk-forward evaluation and save the results."""
    runs_output_path = data_clean_dir() / "multi_asset_walk_forward_runs.csv"
    panel_output_path = data_clean_dir() / "multi_asset_walk_forward_panel_summary.csv"
    summary_output_path = data_clean_dir() / "multi_asset_walk_forward_agent_summary.csv"

    start_time = time.perf_counter()
    run_rows: list[dict[str, float | int | str]] = []
    panel_counter = 0

    for ticker in TICKERS:
        ensure_ticker_inputs(ticker)
        market_df = load_fold_market_data(ticker)
        folds = build_folds(market_df)

        if not folds:
            print(f"[walk-forward] skipping {ticker}: not enough history for one test fold.", flush=True)
            continue

        for fold in folds:
            fold_market_df = fold["market_df"]
            fold_start = fold["start_date"]
            fold_end = fold["end_date"]
            test_bar_count = int(fold["test_bar_count"])
            fold_open_prices = fold_market_df["Open"].to_numpy(dtype=float)

            for agent_name in AGENT_ORDER:
                actual_metrics = build_fold_actual_metrics(
                    ticker=ticker,
                    fold_id=str(fold["fold_id"]),
                    agent_name=agent_name,
                    fold_market_df=fold_market_df,
                    test_bar_count=test_bar_count,
                )
                if actual_metrics is None:
                    continue

                panel_counter += 1
                for outer_run in range(1, OUTER_RUNS + 1):
                    seed_used = BASE_SEED + outer_run
                    if panel_counter == 1 and (outer_run == 1 or outer_run % PROGRESS_EVERY == 0):
                        elapsed_seconds = time.perf_counter() - start_time
                        print(
                            f"[walk-forward] warm-up run {outer_run}/{OUTER_RUNS} "
                            f"elapsed={elapsed_seconds:.1f}s",
                            flush=True,
                        )

                    seed_sequence = np.random.SeedSequence(
                        [seed_used, panel_counter, sum(ord(char) for char in ticker), AGENT_ORDER.index(agent_name)]
                    )
                    rng = np.random.default_rng(seed_sequence)
                    simulated_returns = simulate_random_timing_cumulative_returns(
                        open_prices=fold_open_prices,
                        durations=actual_metrics["durations"],
                        position_value_fractions=actual_metrics["position_value_fractions"],
                        simulation_count=SIMULATIONS_PER_RUN,
                        rng=rng,
                    )
                    run_rows.append(
                        build_panel_run_row(
                            ticker=ticker,
                            fold_id=str(fold["fold_id"]),
                            fold_start=fold_start,
                            fold_end=fold_end,
                            agent_name=agent_name,
                            outer_run=outer_run,
                            seed_used=seed_used,
                            actual_metrics=actual_metrics,
                            test_bar_count=test_bar_count,
                            simulated_returns=simulated_returns,
                        )
                    )

            elapsed_seconds = time.perf_counter() - start_time
            print(
                f"[walk-forward] ticker={ticker} fold={fold['fold_id']} "
                f"window={fold_start.date()}->{fold_end.date()} "
                f"elapsed={elapsed_seconds:.1f}s",
                flush=True,
            )

    if not run_rows:
        raise ValueError("No walk-forward panels met the minimum trade requirement.")

    runs_df = pd.DataFrame(run_rows)
    runs_df.to_csv(runs_output_path, index=False)

    panel_df = aggregate_panel_runs(runs_df)
    panel_df.to_csv(panel_output_path, index=False)

    summary_df = aggregate_agent_summary(panel_df)
    summary_df.to_csv(summary_output_path, index=False)

    total_elapsed_seconds = time.perf_counter() - start_time
    print(f"[walk-forward] completed in {total_elapsed_seconds:.1f}s", flush=True)
    print(f"[walk-forward] tickers: {', '.join(TICKERS)}", flush=True)
    print(f"[walk-forward] panels evaluated: {len(panel_df)}", flush=True)

    for _, summary_row in summary_df.iterrows():
        print_agent_summary(summary_row)


if __name__ == "__main__":
    main()
