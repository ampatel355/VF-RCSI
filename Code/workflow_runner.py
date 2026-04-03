"""Reusable workflow runners and artifact helpers for CLI and UI entrypoints."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable

import pandas as pd

try:
    from artifact_provenance import current_run_id, pipeline_profile
    from pipeline_utils import charts_dir, data_clean_dir, data_raw_dir
    from strategy_config import AGENT_ORDER, BENCHMARK_NAME
except ModuleNotFoundError:
    from Code.artifact_provenance import current_run_id, pipeline_profile
    from Code.pipeline_utils import charts_dir, data_clean_dir, data_raw_dir
    from Code.strategy_config import AGENT_ORDER, BENCHMARK_NAME


PRE_STRATEGY_STEPS: list[tuple[str, str]] = [
    ("Running data loader...", "data_loader.py"),
    ("Running features...", "features.py"),
    ("Running regimes...", "regimes.py"),
]

POST_STRATEGY_STEPS: list[tuple[str, str]] = [
    ("Running trade activity validation...", "trade_activity_validation.py"),
    ("Running buy-and-hold benchmark...", "buy_and_hold.py"),
    ("Running Monte Carlo analysis...", "monte_carlo.py"),
    ("Running RCSI analysis...", "rcsi.py"),
    ("Running full comparison table...", "compare_agents.py"),
]

AGENT_WORKFLOW_STEPS: dict[str, dict[str, tuple[str, str]]] = {
    "trend_pullback": {
        "agent": ("Running Trend + Pullback agent...", "trend_pullback_agent.py"),
        "metrics": ("Running Trend + Pullback metrics...", "trend_pullback_metrics.py"),
    },
    "breakout_volume_momentum": {
        "agent": ("Running Breakout + Volume + Momentum agent...", "breakout_volume_momentum_agent.py"),
        "metrics": ("Running Breakout + Volume + Momentum metrics...", "breakout_volume_momentum_metrics.py"),
    },
    "mean_reversion_vol_filter": {
        "agent": ("Running Mean Reversion + Volatility Filter agent...", "mean_reversion_vol_filter_agent.py"),
        "metrics": ("Running Mean Reversion + Volatility Filter metrics...", "mean_reversion_vol_filter_metrics.py"),
    },
    "momentum_relative_strength": {
        "agent": ("Running Momentum + Relative Strength agent...", "momentum_relative_strength_agent.py"),
        "metrics": ("Running Momentum + Relative Strength metrics...", "momentum_relative_strength_metrics.py"),
    },
    "trend_momentum_verification": {
        "agent": (
            "Running Validation Strategy (Adaptive Volatility Momentum) agent...",
            "trend_momentum_verification_agent.py",
        ),
        "metrics": (
            "Running Validation Strategy (Adaptive Volatility Momentum) metrics...",
            "trend_momentum_verification_metrics.py",
        ),
    },
    "adx_trend_following": {
        "agent": ("Running ADX Trend Following agent...", "adx_trend_following_agent.py"),
        "metrics": ("Running ADX Trend Following metrics...", "adx_trend_following_metrics.py"),
    },
    "uptrend_oversold_reversion": {
        "agent": ("Running Uptrend Oversold Reversion agent...", "uptrend_oversold_reversion_agent.py"),
        "metrics": ("Running Uptrend Oversold Reversion metrics...", "uptrend_oversold_reversion_metrics.py"),
    },
    "volatility_squeeze_breakout": {
        "agent": ("Running Volatility Squeeze Breakout agent...", "volatility_squeeze_breakout_agent.py"),
        "metrics": ("Running Volatility Squeeze Breakout metrics...", "volatility_squeeze_breakout_metrics.py"),
    },
    "connors_rsi2_pullback": {
        "agent": ("Running Connors RSI(2) Pullback agent...", "connors_rsi2_pullback_agent.py"),
        "metrics": ("Running Connors RSI(2) Pullback metrics...", "connors_rsi2_pullback_metrics.py"),
    },
    "donchian_trend_reentry": {
        "agent": ("Running Donchian Trend Reentry agent...", "donchian_trend_reentry_agent.py"),
        "metrics": ("Running Donchian Trend Reentry metrics...", "donchian_trend_reentry_metrics.py"),
    },
    "turn_of_month_seasonality": {
        "agent": ("Running Turn-of-Month Seasonality agent...", "turn_of_month_seasonality_agent.py"),
        "metrics": ("Running Turn-of-Month Seasonality metrics...", "turn_of_month_seasonality_metrics.py"),
    },
    "random": {
        "agent": ("Running random agent...", "random_agent.py"),
        "metrics": ("Running random metrics...", "random_metrics.py"),
    },
}


def build_core_single_ticker_steps() -> list[tuple[str, str]]:
    """Build the active core workflow from the configured strategy order."""
    steps = list(PRE_STRATEGY_STEPS)
    for agent_name in AGENT_ORDER:
        step_set = AGENT_WORKFLOW_STEPS.get(agent_name)
        if step_set is None:
            raise KeyError(
                "workflow_runner is missing script mappings for active strategy "
                f"'{agent_name}'."
            )
        steps.append(step_set["agent"])
    for agent_name in AGENT_ORDER:
        step_set = AGENT_WORKFLOW_STEPS.get(agent_name)
        if step_set is None:
            raise KeyError(
                "workflow_runner is missing metrics mappings for active strategy "
                f"'{agent_name}'."
            )
        steps.append(step_set["metrics"])
    steps.extend(POST_STRATEGY_STEPS)
    return steps


CORE_SINGLE_TICKER_STEPS: list[tuple[str, str]] = build_core_single_ticker_steps()

VISUALIZATION_STEPS: list[tuple[str, str]] = [
    ("Displaying equity curve...", "equity_curve.py"),
    ("Displaying skill vs luck summary...", "strategy_verdict_plot.py"),
]

EXTENDED_ANALYSIS_STEPS: list[tuple[str, str]] = CORE_SINGLE_TICKER_STEPS + [
    ("Running regime analysis...", "regime_analysis.py"),
    ("Running Monte Carlo robustness analysis...", "monte_carlo_robustness.py"),
    ("Refreshing full comparison table...", "compare_agents.py"),
]

EXTENDED_PLOTTING_STEPS: list[tuple[str, str]] = [
    ("Running RCSI plot...", "rcsi_plot.py"),
    ("Running regime plot...", "regime_plot.py"),
    ("Running regime heatmap...", "rcsi_heatmap.py"),
    ("Running Monte Carlo plots...", "monte_carlo_plot.py"),
    ("Running robustness plots...", "monte_carlo_robustness_plot.py"),
    ("Running p-value plot...", "p_value_plot.py"),
]


def build_single_ticker_steps(
    *,
    minimal_mode: bool,
    include_visuals: bool,
    include_extended_plots: bool,
) -> list[tuple[str, str]]:
    """Return the single-ticker workflow steps for the chosen run profile."""
    base_steps = CORE_SINGLE_TICKER_STEPS if minimal_mode else EXTENDED_ANALYSIS_STEPS
    steps = list(base_steps)
    if include_extended_plots and not minimal_mode:
        steps.extend(EXTENDED_PLOTTING_STEPS)
    if include_visuals:
        steps.extend(VISUALIZATION_STEPS)
    return steps

DEFAULT_WALK_FORWARD_TICKERS = [
    "SPY",
    "QQQ",
    "AAPL",
    "VOO",
    "NVDA",
    "TSM",
    "MRNA",
    "NVAX",
    "BTC-USD",
    "EURUSD=X",
    "JPYUSD=X",
    "NQ=F",
    "ES=F",
]

FAST_TEST_ENV_OVERRIDES: dict[str, str] = {
    # Keep the strategy logic identical, but shrink the expensive sampling layers
    # so the full pipeline is practical while debugging.
    "FAST_TEST_MODE": "1",
    "MONTE_CARLO_SIMULATIONS": "2000",
    "MONTE_CARLO_BATCH_SIZE": "512",
    "ROBUSTNESS_SIMULATIONS_PER_RUN": "500",
    "ROBUSTNESS_OUTER_RUNS": "10",
    "ROBUSTNESS_PROGRESS_EVERY": "2",
}

WALK_FORWARD_FAST_TEST_ENV_OVERRIDES: dict[str, str] = {
    "TEST_BARS": "126",
    "STEP_BARS": "63",
    "SIMULATIONS_PER_RUN": "25",
    "OUTER_RUNS": "2",
    "PROGRESS_EVERY": "1",
    "REFRESH_DATA": "0",
}


@dataclass(slots=True)
class WorkflowEvent:
    """One progress event emitted while a workflow is running."""

    kind: str
    workflow_name: str
    message: str = ""
    step_index: int | None = None
    total_steps: int | None = None
    step_label: str | None = None
    script_name: str | None = None
    returncode: int | None = None


@dataclass(slots=True)
class StepRunResult:
    """Captured result for one workflow step."""

    label: str
    script_name: str
    returncode: int
    log_text: str
    started_at: float
    finished_at: float

    @property
    def duration_seconds(self) -> float:
        """Return the elapsed runtime for this step."""
        return self.finished_at - self.started_at


@dataclass(slots=True)
class WorkflowRunResult:
    """Final result for a workflow execution."""

    workflow_name: str
    success: bool
    started_at: float
    finished_at: float
    step_results: list[StepRunResult] = field(default_factory=list)
    error_message: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Return the elapsed runtime for the workflow."""
        return self.finished_at - self.started_at

    @property
    def combined_log(self) -> str:
        """Concatenate the logs from all executed steps."""
        sections: list[str] = []
        for step in self.step_results:
            sections.append(f"$ {step.script_name}\n{step.log_text}".rstrip())
        if self.error_message:
            sections.append(f"ERROR\n{self.error_message}")
        return "\n\n".join(section for section in sections if section)


EventCallback = Callable[[WorkflowEvent], None]


def code_dir() -> Path:
    """Return the Code directory."""
    return Path(__file__).resolve().parent


def build_runtime_context(
    env_overrides: dict[str, str] | None = None,
    show_plots: bool = False,
    save_outputs: bool = False,
) -> tuple[Path, str, dict[str, str]]:
    """Build the shared runtime context used by subprocess workflows."""
    script_dir = code_dir()
    python_executable = sys.executable
    env = os.environ.copy()
    env["SHOW_PLOTS"] = "1" if show_plots else "0"
    env["SAVE_OUTPUTS"] = "1" if save_outputs else "0"
    env.setdefault("PIPELINE_RUN_ID", current_run_id())
    # Prevent stale trade logs from prior runs from being reused silently.
    env.setdefault("REQUIRE_TRADE_RUN_ID_MATCH", "1")
    if not show_plots:
        env["MPLBACKEND"] = "Agg"
    else:
        env.pop("MPLBACKEND", None)
    env["MPLCONFIGDIR"] = str(script_dir.parent / ".matplotlib")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    if env_overrides:
        for key, value in env_overrides.items():
            env[key] = str(value)

    env.setdefault("PIPELINE_PROFILE", pipeline_profile())

    return script_dir, python_executable, env


def _emit(callback: EventCallback | None, event: WorkflowEvent) -> None:
    """Send one progress event when a callback is available."""
    if callback is not None:
        callback(event)


def prune_inactive_strategy_artifacts(ticker: str) -> int:
    """Delete stale per-ticker files for strategies not in the active order."""
    inactive_agents = sorted(set(AGENT_WORKFLOW_STEPS) - set(AGENT_ORDER))
    if not inactive_agents:
        return 0

    removed_count = 0
    ticker_prefix = f"{ticker.upper()}_"
    for directory in (data_clean_dir(), charts_dir()):
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if not path.is_file():
                continue
            filename = path.name
            if not filename.startswith(ticker_prefix):
                continue
            for agent_name in inactive_agents:
                if f"_{agent_name}_" in filename or filename.startswith(
                    f"{ticker_prefix}{agent_name}."
                ):
                    path.unlink(missing_ok=True)
                    removed_count += 1
                    break

    return removed_count


def merge_fast_test_overrides(
    env_overrides: dict[str, str] | None,
    *,
    fast_test_mode: bool,
) -> dict[str, str]:
    """Merge the optional fast-test preset with any caller-provided overrides.

    Fast mode is an explicit request for speed, so its preset wins over any
    conflicting Monte Carlo or robustness values supplied by the caller.
    """
    merged: dict[str, str] = {}
    if env_overrides:
        merged.update({key: str(value) for key, value in env_overrides.items()})
    if fast_test_mode:
        merged.update(FAST_TEST_ENV_OVERRIDES)
    return merged


def _run_script_capture(
    *,
    workflow_name: str,
    script_dir: Path,
    python_executable: str,
    env: dict[str, str],
    label: str,
    script_name: str,
    step_index: int,
    total_steps: int,
    event_callback: EventCallback | None,
) -> StepRunResult:
    """Run one Python script and capture merged stdout and stderr."""
    _emit(
        event_callback,
        WorkflowEvent(
            kind="step_started",
            workflow_name=workflow_name,
            step_index=step_index,
            total_steps=total_steps,
            step_label=label,
            script_name=script_name,
            message=label,
        ),
    )

    started_at = time.time()
    process = subprocess.Popen(
        [python_executable, script_name],
        cwd=script_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    captured_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        captured_lines.append(line)
        _emit(
            event_callback,
            WorkflowEvent(
                kind="log",
                workflow_name=workflow_name,
                step_index=step_index,
                total_steps=total_steps,
                step_label=label,
                script_name=script_name,
                message=line.rstrip("\n"),
            ),
        )

    process.wait()
    finished_at = time.time()
    log_text = "".join(captured_lines)

    _emit(
        event_callback,
        WorkflowEvent(
            kind="step_finished",
            workflow_name=workflow_name,
            step_index=step_index,
            total_steps=total_steps,
            step_label=label,
            script_name=script_name,
            returncode=process.returncode,
            message=f"{label} finished with exit code {process.returncode}.",
        ),
    )

    return StepRunResult(
        label=label,
        script_name=script_name,
        returncode=process.returncode,
        log_text=log_text,
        started_at=started_at,
        finished_at=finished_at,
    )


def run_single_ticker_pipeline(
    ticker: str,
    *,
    env_overrides: dict[str, str] | None = None,
    event_callback: EventCallback | None = None,
    fast_test_mode: bool = False,
    show_plots: bool = False,
    save_outputs: bool = False,
    minimal_mode: bool = True,
) -> WorkflowRunResult:
    """Run the full single-ticker workflow using the existing project scripts."""
    workflow_name = "Single-Ticker Full Pipeline"
    started_at = time.time()
    runtime_overrides = merge_fast_test_overrides(
        env_overrides,
        fast_test_mode=fast_test_mode,
    )
    script_dir, python_executable, env = build_runtime_context(
        env_overrides={**runtime_overrides, "TICKER": ticker.upper()},
        show_plots=show_plots,
        save_outputs=save_outputs,
    )
    removed_stale_files = prune_inactive_strategy_artifacts(ticker)
    single_ticker_steps = build_single_ticker_steps(
        minimal_mode=minimal_mode,
        include_visuals=(show_plots or save_outputs),
        include_extended_plots=(show_plots or save_outputs),
    )

    _emit(
        event_callback,
        WorkflowEvent(
            kind="workflow_started",
            workflow_name=workflow_name,
            message=(
                f"Starting full single-ticker pipeline for {ticker.upper()} "
                f"({'fast test mode' if fast_test_mode else 'full research mode'}, "
                f"{'minimal' if minimal_mode else 'extended'} profile)."
            ),
        ),
    )
    if removed_stale_files > 0:
        _emit(
            event_callback,
            WorkflowEvent(
                kind="log",
                workflow_name=workflow_name,
                message=(
                    f"Removed {removed_stale_files} stale artifact file(s) for disabled "
                    f"strategies before running {ticker.upper()}."
                ),
            ),
        )

    step_results: list[StepRunResult] = []
    total_steps = len(single_ticker_steps)
    error_message: str | None = None

    for step_index, (label, script_name) in enumerate(single_ticker_steps, start=1):
        step_result = _run_script_capture(
            workflow_name=workflow_name,
            script_dir=script_dir,
            python_executable=python_executable,
            env=env,
            label=label,
            script_name=script_name,
            step_index=step_index,
            total_steps=total_steps,
            event_callback=event_callback,
        )
        step_results.append(step_result)
        if step_result.returncode != 0:
            error_message = (
                f"{script_name} failed with exit code {step_result.returncode}. "
                "See the captured logs for details."
            )
            break

    finished_at = time.time()
    success = error_message is None
    _emit(
        event_callback,
        WorkflowEvent(
            kind="workflow_finished",
            workflow_name=workflow_name,
            message=(
                f"Completed single-ticker pipeline for {ticker.upper()}."
                if success
                else error_message or "Pipeline failed."
            ),
        ),
    )
    return WorkflowRunResult(
        workflow_name=workflow_name,
        success=success,
        started_at=started_at,
        finished_at=finished_at,
        step_results=step_results,
        error_message=error_message,
        metadata={
            "ticker": ticker.upper(),
            "run_mode": "fast_test" if fast_test_mode else "full",
            "profile": "minimal" if minimal_mode else "extended",
        },
    )


def run_walk_forward_pipeline(
    tickers: list[str] | None = None,
    *,
    env_overrides: dict[str, str] | None = None,
    event_callback: EventCallback | None = None,
    fast_test_mode: bool = False,
) -> WorkflowRunResult:
    """Run the multi-asset walk-forward workflow."""
    workflow_name = "Multi-Asset Walk-Forward Evaluation"
    started_at = time.time()
    normalized_tickers = [ticker.strip().upper() for ticker in (tickers or DEFAULT_WALK_FORWARD_TICKERS)]
    normalized_tickers = [ticker for ticker in normalized_tickers if ticker]
    env_payload = merge_fast_test_overrides(
        env_overrides,
        fast_test_mode=fast_test_mode,
    )
    if fast_test_mode:
        for key, value in WALK_FORWARD_FAST_TEST_ENV_OVERRIDES.items():
            env_payload.setdefault(key, value)
    env_payload["WALK_FORWARD_TICKERS"] = ",".join(normalized_tickers)
    script_dir, python_executable, env = build_runtime_context(
        env_overrides=env_payload,
        show_plots=False,
        save_outputs=False,
    )

    _emit(
        event_callback,
        WorkflowEvent(
            kind="workflow_started",
            workflow_name=workflow_name,
            message=(
                "Starting multi-asset walk-forward evaluation "
                f"({'fast test mode' if fast_test_mode else 'full research mode'})."
            ),
        ),
    )

    step_result = _run_script_capture(
        workflow_name=workflow_name,
        script_dir=script_dir,
        python_executable=python_executable,
        env=env,
        label="Running multi-asset walk-forward evaluation...",
        script_name="multi_asset_walk_forward.py",
        step_index=1,
        total_steps=1,
        event_callback=event_callback,
    )
    finished_at = time.time()
    error_message = None
    if step_result.returncode != 0:
        error_message = (
            "multi_asset_walk_forward.py failed with exit code "
            f"{step_result.returncode}. See the captured logs for details."
        )

    _emit(
        event_callback,
        WorkflowEvent(
            kind="workflow_finished",
            workflow_name=workflow_name,
            message=(
                "Completed multi-asset walk-forward evaluation."
                if error_message is None
                else error_message
            ),
        ),
    )
    return WorkflowRunResult(
        workflow_name=workflow_name,
        success=error_message is None,
        started_at=started_at,
        finished_at=finished_at,
        step_results=[step_result],
        error_message=error_message,
        metadata={
            "tickers": ",".join(normalized_tickers),
            "run_mode": "fast_test" if fast_test_mode else "full",
        },
    )


def existing_tickers() -> list[str]:
    """Return the tickers currently available from raw or clean project files."""
    ticker_names = {
        path.stem.upper()
        for path in data_raw_dir().glob("*.csv")
        if path.is_file()
    }
    for path in data_clean_dir().glob("*_regimes.csv"):
        ticker_names.add(path.name.replace("_regimes.csv", "").upper())
    return sorted(ticker_names)


def ticker_chart_files(ticker: str) -> list[Path]:
    """Return every saved chart file for one ticker."""
    ticker = ticker.upper()
    paths = sorted(charts_dir().glob(f"{ticker}_*"))
    return [path for path in paths if path.is_file()]


def ticker_data_files(ticker: str) -> list[Path]:
    """Return every saved CSV artifact associated with one ticker."""
    ticker = ticker.upper()
    clean_paths = sorted(data_clean_dir().glob(f"{ticker}_*.csv"))
    raw_path = data_raw_dir() / f"{ticker}.csv"
    files = [path for path in clean_paths if path.is_file()]
    if raw_path.exists():
        files.insert(0, raw_path)
    return files


def ticker_trade_files(ticker: str) -> list[Path]:
    """Return every trade log file for one ticker."""
    ticker = ticker.upper()
    return sorted(
        path
        for path in data_clean_dir().glob(f"{ticker}_*trade*.csv")
        if path.is_file()
    )


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    """Load one CSV file when it exists and is readable."""
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def single_ticker_summary_tables(ticker: str) -> dict[str, pd.DataFrame]:
    """Load the most important summary tables for one ticker."""
    ticker = ticker.upper()
    table_paths = {
        "Full Comparison": data_clean_dir() / f"{ticker}_full_comparison.csv",
        "Monte Carlo Summary": data_clean_dir() / f"{ticker}_monte_carlo_summary.csv",
        "RCSI": data_clean_dir() / f"{ticker}_rcsi.csv",
        "Trade Activity Validation": data_clean_dir() / f"{ticker}_trade_activity_validation.csv",
        "Buy and Hold Metrics": data_clean_dir() / f"{ticker}_buy_hold_metrics.csv",
    }
    loaded: dict[str, pd.DataFrame] = {}
    for label, path in table_paths.items():
        df = read_csv_if_exists(path)
        if df is not None:
            loaded[label] = df
    return loaded


def walk_forward_output_paths() -> dict[str, Path]:
    """Return the standard saved walk-forward artifact paths."""
    clean_dir = data_clean_dir()
    return {
        "Walk-Forward Runs": clean_dir / "multi_asset_walk_forward_runs.csv",
        "Walk-Forward Panel Summary": clean_dir / "multi_asset_walk_forward_panel_summary.csv",
        "Walk-Forward Agent Summary": clean_dir / "multi_asset_walk_forward_agent_summary.csv",
    }


def walk_forward_tables() -> dict[str, pd.DataFrame]:
    """Load the walk-forward output tables that currently exist."""
    loaded: dict[str, pd.DataFrame] = {}
    for label, path in walk_forward_output_paths().items():
        df = read_csv_if_exists(path)
        if df is not None:
            loaded[label] = df
    return loaded


def walk_forward_artifact_files() -> list[Path]:
    """Return the saved walk-forward files currently present in the project."""
    matches = [
        *data_clean_dir().glob("*walk_forward*"),
        *charts_dir().glob("*walk_forward*"),
    ]
    unique_paths = {
        path.resolve(): path
        for path in matches
        if path.is_file()
    }
    return sorted(unique_paths.values())


def agent_metric_paths(ticker: str) -> dict[str, Path]:
    """Return the per-strategy metric files for one ticker."""
    ticker = ticker.upper()
    return {
        agent_name: data_clean_dir() / f"{ticker}_{agent_name}_metrics.csv"
        for agent_name in AGENT_ORDER
    }


def benchmark_curve_path(ticker: str) -> Path:
    """Return the buy-and-hold curve path for one ticker."""
    normalized_ticker = ticker.upper()
    preferred_path = data_clean_dir() / f"{normalized_ticker}_buy_hold_curve.csv"
    if preferred_path.exists():
        return preferred_path
    return data_clean_dir() / f"{normalized_ticker}_{BENCHMARK_NAME}_curve.csv"
