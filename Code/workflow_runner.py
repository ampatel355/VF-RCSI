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
from PIL import Image

try:
    from pipeline_utils import charts_dir, data_clean_dir, data_raw_dir, pipeline_chart_paths
    from strategy_config import AGENT_ORDER, BENCHMARK_NAME
except ModuleNotFoundError:
    from Code.pipeline_utils import charts_dir, data_clean_dir, data_raw_dir, pipeline_chart_paths
    from Code.strategy_config import AGENT_ORDER, BENCHMARK_NAME


DEFAULT_SINGLE_TICKER_STEPS: list[tuple[str, str]] = [
    ("Running data loader...", "data_loader.py"),
    ("Running features...", "features.py"),
    ("Running regimes...", "regimes.py"),
    ("Running trend agent...", "trend_agent.py"),
    ("Running mean reversion agent...", "mean_reversion_agent.py"),
    ("Running random agent...", "random_agent.py"),
    ("Running momentum agent...", "momentum_agent.py"),
    ("Running breakout agent...", "breakout_agent.py"),
    ("Running trend metrics...", "trend_metrics.py"),
    ("Running mean reversion metrics...", "mean_reversion_metrics.py"),
    ("Running random metrics...", "random_metrics.py"),
    ("Running momentum metrics...", "momentum_metrics.py"),
    ("Running breakout metrics...", "breakout_metrics.py"),
    ("Running buy-and-hold benchmark...", "buy_and_hold.py"),
    ("Running regime analysis...", "regime_analysis.py"),
    ("Running Monte Carlo analysis...", "monte_carlo.py"),
    ("Running Monte Carlo robustness analysis...", "monte_carlo_robustness.py"),
    ("Running RCSI analysis...", "rcsi.py"),
    ("Running full comparison table...", "compare_agents.py"),
    ("Running RCSI plot...", "rcsi_plot.py"),
    ("Running regime plot...", "regime_plot.py"),
    ("Running regime heatmap...", "rcsi_heatmap.py"),
    ("Running equity curve charts...", "equity_curve.py"),
    ("Running Monte Carlo plots...", "monte_carlo_plot.py"),
    ("Running Monte Carlo robustness plots...", "monte_carlo_robustness_plot.py"),
    ("Running p-value plot...", "p_value_plot.py"),
    ("Running skill vs luck summary...", "strategy_verdict_plot.py"),
    ("Building combined chart PDF...", "open_charts.py"),
]

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
    interactive_chart_open: bool = False,
) -> tuple[Path, str, dict[str, str]]:
    """Build the shared runtime context used by subprocess workflows."""
    script_dir = code_dir()
    python_executable = sys.executable
    env = os.environ.copy()
    env["SHOW_PLOTS"] = "1" if show_plots else "0"
    env["OPEN_CHARTS_INTERACTIVE"] = "1" if interactive_chart_open else "0"
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(script_dir.parent / ".matplotlib")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    if env_overrides:
        for key, value in env_overrides.items():
            env[key] = str(value)

    return script_dir, python_executable, env


def _emit(callback: EventCallback | None, event: WorkflowEvent) -> None:
    """Send one progress event when a callback is available."""
    if callback is not None:
        callback(event)


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
    interactive_chart_open: bool = False,
) -> WorkflowRunResult:
    """Run the full single-ticker workflow using the existing project scripts."""
    workflow_name = "Single-Ticker Full Pipeline"
    started_at = time.time()
    script_dir, python_executable, env = build_runtime_context(
        env_overrides={**(env_overrides or {}), "TICKER": ticker.upper()},
        show_plots=False,
        interactive_chart_open=interactive_chart_open,
    )

    _emit(
        event_callback,
        WorkflowEvent(
            kind="workflow_started",
            workflow_name=workflow_name,
            message=f"Starting full single-ticker pipeline for {ticker.upper()}",
        ),
    )

    step_results: list[StepRunResult] = []
    total_steps = len(DEFAULT_SINGLE_TICKER_STEPS)
    error_message: str | None = None

    for step_index, (label, script_name) in enumerate(DEFAULT_SINGLE_TICKER_STEPS, start=1):
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
        metadata={"ticker": ticker.upper()},
    )


def run_walk_forward_pipeline(
    tickers: list[str] | None = None,
    *,
    env_overrides: dict[str, str] | None = None,
    event_callback: EventCallback | None = None,
) -> WorkflowRunResult:
    """Run the preserved multi-asset walk-forward workflow."""
    workflow_name = "Multi-Asset Walk-Forward Evaluation"
    started_at = time.time()
    normalized_tickers = [ticker.strip().upper() for ticker in (tickers or DEFAULT_WALK_FORWARD_TICKERS)]
    normalized_tickers = [ticker for ticker in normalized_tickers if ticker]
    env_payload = dict(env_overrides or {})
    env_payload["WALK_FORWARD_TICKERS"] = ",".join(normalized_tickers)
    script_dir, python_executable, env = build_runtime_context(
        env_overrides=env_payload,
        show_plots=False,
        interactive_chart_open=False,
    )

    _emit(
        event_callback,
        WorkflowEvent(
            kind="workflow_started",
            workflow_name=workflow_name,
            message="Starting multi-asset walk-forward evaluation.",
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
        metadata={"tickers": ",".join(normalized_tickers)},
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
        for path in data_clean_dir().glob(f"{ticker}_*_trades.csv")
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
        "Agent Comparison": data_clean_dir() / f"{ticker}_agent_comparison.csv",
        "Monte Carlo Summary": data_clean_dir() / f"{ticker}_monte_carlo_summary.csv",
        "Monte Carlo Robustness Summary": data_clean_dir() / f"{ticker}_monte_carlo_robustness_summary.csv",
        "RCSI": data_clean_dir() / f"{ticker}_rcsi.csv",
        "Regime Analysis": data_clean_dir() / f"{ticker}_regime_analysis.csv",
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


def ensure_combined_chart_pdf(ticker: str) -> Path | None:
    """Build the combined chart PDF for one ticker when its PNG charts exist."""
    try:
        chart_files = [
            chart_path
            for chart_path in pipeline_chart_paths(ticker.upper())
            if chart_path.exists()
        ]
        if not chart_files:
            return None

        pdf_path = chart_files[0].parent / f"{ticker.upper()}_pipeline_charts.pdf"
        image_pages = []
        for chart_path in chart_files:
            with Image.open(chart_path) as image:
                image_pages.append(image.convert("RGB"))

        first_page, *remaining_pages = image_pages
        first_page.save(pdf_path, save_all=True, append_images=remaining_pages)
        return pdf_path
    except Exception:
        return None


def agent_metric_paths(ticker: str) -> dict[str, Path]:
    """Return the per-strategy metric files for one ticker."""
    ticker = ticker.upper()
    return {
        agent_name: data_clean_dir() / f"{ticker}_{agent_name}_metrics.csv"
        for agent_name in AGENT_ORDER
    }


def benchmark_curve_path(ticker: str) -> Path:
    """Return the buy-and-hold curve path for one ticker."""
    return data_clean_dir() / f"{ticker.upper()}_{BENCHMARK_NAME}_curve.csv"
