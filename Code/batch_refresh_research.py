"""Refresh research outputs for multiple tickers under the current schema."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys

import pandas as pd

try:
    from pipeline_utils import data_clean_dir, data_raw_dir
    from strategy_config import AGENT_ORDER, BENCHMARK_NAME
except ModuleNotFoundError:
    from Code.pipeline_utils import data_clean_dir, data_raw_dir
    from Code.strategy_config import AGENT_ORDER, BENCHMARK_NAME


DOWNLOAD_DATA = os.environ.get("BATCH_REFRESH_DOWNLOAD_DATA", "0") == "1"
INCLUDE_CHARTS = os.environ.get("BATCH_REFRESH_INCLUDE_CHARTS", "1") == "1"
ONLY_STALE = os.environ.get("BATCH_REFRESH_ONLY_STALE", "1") == "1"
CONTINUE_ON_ERROR = os.environ.get("BATCH_REFRESH_CONTINUE_ON_ERROR", "1") == "1"
CONFIGURED_TICKERS = os.environ.get("BATCH_TICKERS", "").strip()
MIN_CURRENT_OUTER_RUNS = int(os.environ.get("BATCH_REFRESH_MIN_OUTER_RUNS", "30"))
MIN_CURRENT_SIMULATIONS_PER_RUN = int(
    os.environ.get("BATCH_REFRESH_MIN_SIMULATIONS_PER_RUN", "1000")
)

CORE_STEPS = [
    "features.py",
    "regimes.py",
    "trend_agent.py",
    "mean_reversion_agent.py",
    "random_agent.py",
    "momentum_agent.py",
    "breakout_agent.py",
    "trend_metrics.py",
    "mean_reversion_metrics.py",
    "random_metrics.py",
    "momentum_metrics.py",
    "breakout_metrics.py",
    "buy_and_hold.py",
    "regime_analysis.py",
    "monte_carlo.py",
    "monte_carlo_robustness.py",
    "rcsi.py",
    "compare_agents.py",
]

CHART_STEPS = [
    "strategy_verdict_plot.py",
    "p_value_plot.py",
    "rcsi_plot.py",
    "regime_plot.py",
    "rcsi_heatmap.py",
    "monte_carlo_robustness_plot.py",
    "monte_carlo_plot.py",
    "equity_curve.py",
]

REQUIRED_ROBUSTNESS_COLUMNS = {
    "agent",
    "mean_p_value",
    "mean_RCSI",
    "mean_RCSI_z",
    "mean_percentile",
    "proportion_significant",
    "proportion_outperforming_null_median",
    "stability_classification",
    "evidence_label",
    "confidence_label",
    "skill_luck_verdict",
    "number_of_outer_runs",
    "simulations_per_run",
}
REQUIRED_COMPARISON_COLUMNS = {
    "agent",
    "skill_luck_verdict",
    "evidence_label",
    "confidence_label",
    "RCSI",
    "RCSI_z",
    "p_value",
    "p_value_prominence",
    "actual_percentile",
    "annualized_sharpe",
}


def code_dir() -> Path:
    """Return the scripts folder."""
    return Path(__file__).resolve().parent


def configured_ticker_list() -> list[str]:
    """Return the refresh universe."""
    if CONFIGURED_TICKERS:
        return [ticker.strip().upper() for ticker in CONFIGURED_TICKERS.split(",") if ticker.strip()]

    return sorted(path.stem.upper() for path in data_raw_dir().glob("*.csv"))


def build_env(ticker: str) -> dict[str, str]:
    """Build the child-process environment for one ticker."""
    env = os.environ.copy()
    env["TICKER"] = ticker
    env["SHOW_PLOTS"] = "0"
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(code_dir().parent / ".matplotlib")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    """Load one CSV when it exists, otherwise return None."""
    if not path.exists():
        return None

    return pd.read_csv(path)


def output_paths_for_ticker(ticker: str) -> tuple[Path, Path]:
    """Return the robustness and comparison paths for one ticker."""
    clean_dir = data_clean_dir()
    return (
        clean_dir / f"{ticker}_monte_carlo_robustness_summary.csv",
        clean_dir / f"{ticker}_full_comparison.csv",
    )


def ticker_is_current(ticker: str) -> bool:
    """Check whether a ticker already has the current research schema."""
    robustness_path, comparison_path = output_paths_for_ticker(ticker)
    robustness_df = load_csv_if_exists(robustness_path)
    comparison_df = load_csv_if_exists(comparison_path)

    if robustness_df is None or comparison_df is None:
        return False

    if not REQUIRED_ROBUSTNESS_COLUMNS.issubset(set(robustness_df.columns)):
        return False
    if not REQUIRED_COMPARISON_COLUMNS.issubset(set(comparison_df.columns)):
        return False

    robustness_agents = set(robustness_df["agent"].astype(str).str.strip())
    comparison_agents = set(comparison_df["agent"].astype(str).str.strip())
    expected_robustness_agents = set(AGENT_ORDER)
    expected_comparison_agents = set(AGENT_ORDER) | {BENCHMARK_NAME}

    outer_runs = pd.to_numeric(robustness_df["number_of_outer_runs"], errors="coerce")
    simulations_per_run = pd.to_numeric(robustness_df["simulations_per_run"], errors="coerce")
    has_enough_runs = not outer_runs.isna().any() and float(outer_runs.min()) >= MIN_CURRENT_OUTER_RUNS
    has_enough_simulations = (
        not simulations_per_run.isna().any()
        and float(simulations_per_run.min()) >= MIN_CURRENT_SIMULATIONS_PER_RUN
    )

    return (
        robustness_agents == expected_robustness_agents
        and comparison_agents == expected_comparison_agents
        and has_enough_runs
        and has_enough_simulations
    )


def validate_ticker_outputs(ticker: str) -> None:
    """Raise an error if the refreshed outputs are incomplete or stale."""
    if not ticker_is_current(ticker):
        raise RuntimeError(
            f"{ticker} did not finish with the current schema. "
            "The robustness summary or comparison file is still stale or underpowered."
        )


def run_step(script_name: str, env: dict[str, str]) -> None:
    """Run one script inside the Code folder."""
    subprocess.run(
        [sys.executable, script_name],
        cwd=code_dir(),
        env=env,
        check=True,
    )


def refresh_ticker(ticker: str) -> None:
    """Refresh one ticker under the current research logic."""
    env = build_env(ticker)
    print(f"\n[batch-refresh] {ticker}: starting", flush=True)

    if DOWNLOAD_DATA:
        print(f"[batch-refresh] {ticker}: data_loader.py", flush=True)
        run_step("data_loader.py", env)

    for script_name in CORE_STEPS:
        print(f"[batch-refresh] {ticker}: {script_name}", flush=True)
        run_step(script_name, env)

    if INCLUDE_CHARTS:
        for script_name in CHART_STEPS:
            print(f"[batch-refresh] {ticker}: {script_name}", flush=True)
            run_step(script_name, env)

    validate_ticker_outputs(ticker)
    print(f"[batch-refresh] {ticker}: complete", flush=True)


def main() -> None:
    """Refresh all requested tickers and report any failures."""
    tickers = configured_ticker_list()
    if not tickers:
        raise RuntimeError("No tickers were found to refresh.")

    stale_tickers = [ticker for ticker in tickers if not ticker_is_current(ticker)]
    selected_tickers = stale_tickers if ONLY_STALE else tickers

    if not selected_tickers:
        print("All selected tickers already match the current research schema.", flush=True)
        return

    print(
        "[batch-refresh] tickers to refresh: "
        + ", ".join(selected_tickers),
        flush=True,
    )

    failures: list[tuple[str, str]] = []
    for ticker in selected_tickers:
        try:
            refresh_ticker(ticker)
        except Exception as error:  # noqa: BLE001 - batch run should continue when configured
            failures.append((ticker, str(error)))
            print(f"[batch-refresh] {ticker}: FAILED -> {error}", flush=True)
            if not CONTINUE_ON_ERROR:
                break

    if failures:
        print("\n[batch-refresh] completed with failures:", flush=True)
        for ticker, error_text in failures:
            print(f"- {ticker}: {error_text}", flush=True)
        raise SystemExit(1)

    print("\n[batch-refresh] all selected tickers refreshed successfully.", flush=True)


if __name__ == "__main__":
    main()
