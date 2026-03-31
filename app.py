"""Streamlit interface for the Virtu Fortuna research workflows."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st
from streamlit.runtime import exists as streamlit_runtime_exists


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Code.pipeline_utils import data_clean_dir, data_raw_dir
from Code.strategy_config import AGENT_DISPLAY_NAMES, AGENT_ORDER, BENCHMARK_NAME
from Code.workflow_runner import (
    DEFAULT_WALK_FORWARD_TICKERS,
    WorkflowEvent,
    WorkflowRunResult,
    agent_metric_paths,
    benchmark_curve_path,
    ensure_combined_chart_pdf,
    existing_tickers,
    read_csv_if_exists,
    run_single_ticker_pipeline,
    run_walk_forward_pipeline,
    single_ticker_summary_tables,
    ticker_chart_files,
    ticker_data_files,
    ticker_trade_files,
    walk_forward_output_paths,
    walk_forward_tables,
)


APP_TITLE = "Virtu Fortuna Research Interface"
APP_SUBTITLE = (
    "Run the real backtesting and walk-forward workflows through a web interface "
    "without stripping out the project’s existing metrics, charts, tables, files, or logs."
)

DEFAULT_SINGLE_TICKER = "SPY"


def to_env_flag(value: bool) -> str:
    """Convert a checkbox state into the string flag expected by the scripts."""
    return "1" if value else "0"


def split_tickers(text: str) -> list[str]:
    """Parse a comma-separated ticker list."""
    return [ticker.strip().upper() for ticker in text.split(",") if ticker.strip()]


def render_file_download(path: Path, *, label: str, key: str) -> None:
    """Render a download button for one existing file."""
    if not path.exists():
        return

    st.download_button(
        label=label,
        data=path.read_bytes(),
        file_name=path.name,
        key=key,
    )


def render_dataframe_with_download(title: str, path: Path, *, key_prefix: str) -> None:
    """Render one CSV file as a dataframe plus a download button."""
    df = read_csv_if_exists(path)
    if df is None:
        st.info(f"{title} is not available yet.")
        return

    st.markdown(f"**{title}**")
    st.dataframe(df, use_container_width=True, height=320)
    render_file_download(path, label=f"Download {path.name}", key=f"{key_prefix}_download")


def best_strategy_row(comparison_df: pd.DataFrame) -> pd.Series | None:
    """Return the strongest strategy row by cumulative return, excluding the benchmark."""
    if comparison_df.empty or "agent" not in comparison_df.columns:
        return None

    working_df = comparison_df.copy()
    working_df = working_df.loc[working_df["agent"].astype(str) != BENCHMARK_NAME].copy()
    if working_df.empty or "cumulative_return" not in working_df.columns:
        return None

    working_df["cumulative_return"] = pd.to_numeric(
        working_df["cumulative_return"],
        errors="coerce",
    )
    working_df = working_df.dropna(subset=["cumulative_return"])
    if working_df.empty:
        return None

    return working_df.sort_values("cumulative_return", ascending=False).iloc[0]


def lowest_p_value_row(comparison_df: pd.DataFrame) -> pd.Series | None:
    """Return the strategy row with the lowest available p-value."""
    if comparison_df.empty or "p_value" not in comparison_df.columns:
        return None

    working_df = comparison_df.copy()
    working_df["p_value"] = pd.to_numeric(working_df["p_value"], errors="coerce")
    working_df = working_df.dropna(subset=["p_value"])
    if working_df.empty:
        return None

    return working_df.sort_values("p_value", ascending=True).iloc[0]


def render_single_workflow_summary(ticker: str, result: WorkflowRunResult | None) -> None:
    """Render the top-line summary cards for one ticker."""
    summary_tables = single_ticker_summary_tables(ticker)
    comparison_df = summary_tables.get("Full Comparison")
    robustness_df = summary_tables.get("Monte Carlo Robustness Summary")
    chart_files = ticker_chart_files(ticker)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ticker", ticker)
    col2.metric("Charts Found", len(chart_files))
    if result is not None:
        col3.metric("Last Run Status", "Success" if result.success else "Failed")
        col4.metric("Last Run Time", f"{result.duration_seconds:.1f}s")
    else:
        col3.metric("Last Run Status", "Existing Files")
        col4.metric("Last Run Time", "n/a")

    if comparison_df is None:
        st.info("Run the single-ticker pipeline to populate the comparison and summary outputs.")
        return

    best_row = best_strategy_row(comparison_df)
    p_value_row = lowest_p_value_row(comparison_df)
    summary_cols = st.columns(3)
    if best_row is not None:
        summary_cols[0].metric(
            "Best Strategy by Return",
            AGENT_DISPLAY_NAMES.get(str(best_row["agent"]), str(best_row["agent"]).title()),
            f"{float(best_row['cumulative_return']):.2%}",
        )
    else:
        summary_cols[0].metric("Best Strategy by Return", "n/a")

    if p_value_row is not None:
        summary_cols[1].metric(
            "Lowest p-value",
            AGENT_DISPLAY_NAMES.get(str(p_value_row["agent"]), str(p_value_row["agent"]).title()),
            f"{float(p_value_row['p_value']):.4f}",
        )
    else:
        summary_cols[1].metric("Lowest p-value", "n/a")

    if robustness_df is not None and "agent" in robustness_df.columns:
        robust_row = robustness_df.copy()
        if "mean_p_value" in robust_row.columns:
            robust_row["mean_p_value"] = pd.to_numeric(robust_row["mean_p_value"], errors="coerce")
            robust_row = robust_row.dropna(subset=["mean_p_value"])
        if not robust_row.empty:
            best_robust = robust_row.sort_values("mean_p_value", ascending=True).iloc[0]
            summary_cols[2].metric(
                "Strongest Robustness Case",
                AGENT_DISPLAY_NAMES.get(str(best_robust["agent"]), str(best_robust["agent"]).title()),
                f"mean p={float(best_robust['mean_p_value']):.4f}",
            )
        else:
            summary_cols[2].metric("Strongest Robustness Case", "n/a")
    else:
        summary_cols[2].metric("Strongest Robustness Case", "n/a")

    st.markdown("**Primary comparison table**")
    st.dataframe(comparison_df, use_container_width=True, height=340)


def render_single_metrics_tab(ticker: str) -> None:
    """Render the metrics and table outputs for one ticker."""
    ordered_labels = [
        "Full Comparison",
        "Agent Comparison",
        "Monte Carlo Summary",
        "Monte Carlo Robustness Summary",
        "RCSI",
        "Regime Analysis",
        "Buy and Hold Metrics",
    ]
    for label in ordered_labels:
        path_name = {
            "Full Comparison": f"{ticker}_full_comparison.csv",
            "Agent Comparison": f"{ticker}_agent_comparison.csv",
            "Monte Carlo Summary": f"{ticker}_monte_carlo_summary.csv",
            "Monte Carlo Robustness Summary": f"{ticker}_monte_carlo_robustness_summary.csv",
            "RCSI": f"{ticker}_rcsi.csv",
            "Regime Analysis": f"{ticker}_regime_analysis.csv",
            "Buy and Hold Metrics": f"{ticker}_buy_hold_metrics.csv",
        }[label]
        path = data_clean_dir() / path_name
        render_dataframe_with_download(label, path, key_prefix=f"{ticker}_{path.stem}")

    st.markdown("**Per-strategy metric tables**")
    for agent_name, path in agent_metric_paths(ticker).items():
        with st.expander(AGENT_DISPLAY_NAMES.get(agent_name, agent_name.title()), expanded=False):
            render_dataframe_with_download(
                f"{AGENT_DISPLAY_NAMES.get(agent_name, agent_name.title())} metrics",
                path,
                key_prefix=f"{ticker}_{agent_name}_metrics",
            )

    raw_path = data_raw_dir() / f"{ticker}.csv"
    features_path = data_clean_dir() / f"{ticker}_features.csv"
    regimes_path = data_clean_dir() / f"{ticker}_regimes.csv"
    for label, path in [
        ("Raw Price History", raw_path),
        ("Feature Table", features_path),
        ("Regime Table", regimes_path),
    ]:
        with st.expander(label, expanded=False):
            render_dataframe_with_download(label, path, key_prefix=f"{ticker}_{path.stem}")


def render_single_charts_tab(ticker: str) -> None:
    """Render every saved chart artifact for one ticker."""
    combined_pdf = ensure_combined_chart_pdf(ticker)
    if combined_pdf is not None and combined_pdf.exists():
        render_file_download(
            combined_pdf,
            label=f"Download {combined_pdf.name}",
            key=f"{ticker}_combined_pdf",
        )

    chart_paths = ticker_chart_files(ticker)
    if not chart_paths:
        st.info("No chart files are available yet for this ticker.")
        return

    pdf_paths = [path for path in chart_paths if path.suffix.lower() == ".pdf"]
    image_paths = [path for path in chart_paths if path.suffix.lower() != ".pdf"]

    if pdf_paths:
        st.markdown("**PDF outputs**")
        for index, path in enumerate(pdf_paths):
            render_file_download(path, label=f"Download {path.name}", key=f"{ticker}_pdf_{index}")

    st.markdown("**Image outputs**")
    for index, path in enumerate(image_paths):
        st.markdown(f"**{path.name}**")
        st.image(str(path), use_container_width=True)
        render_file_download(path, label=f"Download {path.name}", key=f"{ticker}_img_{index}")


def render_single_trades_tab(ticker: str) -> None:
    """Render the saved trade logs for one ticker."""
    trade_paths = ticker_trade_files(ticker)
    if not trade_paths:
        st.info("No trade logs are available yet for this ticker.")
        return

    for path in trade_paths:
        agent_name = path.name.replace(f"{ticker}_", "").replace("_trades.csv", "")
        title = AGENT_DISPLAY_NAMES.get(agent_name, agent_name.replace("_", " ").title())
        with st.expander(title, expanded=False):
            render_dataframe_with_download(title, path, key_prefix=f"{ticker}_{agent_name}_trades")

    curve_path = benchmark_curve_path(ticker)
    with st.expander("Buy and Hold Curve", expanded=False):
        render_dataframe_with_download("Buy and Hold Curve", curve_path, key_prefix=f"{ticker}_buy_hold_curve")


def render_logs_tab(result: WorkflowRunResult | None) -> None:
    """Render captured logs and per-step diagnostics."""
    if result is None:
        st.info("Run a workflow from the interface to capture live logs here.")
        return

    st.markdown("**Combined raw output**")
    st.code(result.combined_log or "No logs were captured.", language="text")
    st.download_button(
        "Download combined log",
        data=(result.combined_log or "").encode("utf-8"),
        file_name=f"{result.workflow_name.lower().replace(' ', '_')}_log.txt",
        key=f"{result.workflow_name}_combined_log",
    )

    st.markdown("**Step diagnostics**")
    for index, step in enumerate(result.step_results, start=1):
        with st.expander(f"{index}. {step.label} ({step.duration_seconds:.1f}s)", expanded=False):
            st.caption(f"Script: {step.script_name} | Exit code: {step.returncode}")
            st.code(step.log_text or "No step output captured.", language="text")


def render_files_tab(paths: list[Path], *, key_prefix: str) -> None:
    """Render a flat list of downloadable artifact files."""
    if not paths:
        st.info("No files are available for download yet.")
        return

    for index, path in enumerate(sorted(paths)):
        cols = st.columns([4, 1])
        cols[0].write(path.name)
        with cols[1]:
            render_file_download(path, label="Download", key=f"{key_prefix}_{index}")


def render_walk_forward_summary(result: WorkflowRunResult | None) -> None:
    """Render top-line walk-forward metrics and summary tables."""
    tables = walk_forward_tables()
    agent_df = tables.get("Walk-Forward Agent Summary")
    panel_df = tables.get("Walk-Forward Panel Summary")
    runs_df = tables.get("Walk-Forward Runs")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Saved Tables", len(tables))
    col2.metric("Panels", len(panel_df) if panel_df is not None else 0)
    col3.metric("Runs", len(runs_df) if runs_df is not None else 0)
    if result is not None:
        col4.metric("Last Run Time", f"{result.duration_seconds:.1f}s")
    else:
        col4.metric("Last Run Time", "n/a")

    if agent_df is None:
        st.info("Run the walk-forward workflow to populate its saved output tables.")
        return

    if "mean_p_value" in agent_df.columns:
        working_df = agent_df.copy()
        working_df["mean_p_value"] = pd.to_numeric(working_df["mean_p_value"], errors="coerce")
        working_df = working_df.dropna(subset=["mean_p_value"])
        if not working_df.empty:
            top_row = working_df.sort_values("mean_p_value", ascending=True).iloc[0]
            st.metric(
                "Lowest mean p-value",
                AGENT_DISPLAY_NAMES.get(str(top_row["agent"]), str(top_row["agent"]).title()),
                f"{float(top_row['mean_p_value']):.4f}",
            )

    st.markdown("**Walk-forward agent summary**")
    st.dataframe(agent_df, use_container_width=True, height=340)


def render_walk_forward_tab(title: str, path: Path, *, key_prefix: str) -> None:
    """Render one walk-forward CSV output."""
    render_dataframe_with_download(title, path, key_prefix=key_prefix)


def render_shared_parameter_controls(prefix: str) -> dict[str, str]:
    """Render the shared parameter controls and map them to environment variables."""
    env_overrides: dict[str, str] = {}

    with st.expander("Strategy Parameters", expanded=False):
        momentum_lookback = st.number_input(
            "Momentum lookback days",
            min_value=1,
            value=120,
            key=f"{prefix}_momentum_lookback",
        )
        breakout_lookback = st.number_input(
            "Breakout lookback days",
            min_value=1,
            value=20,
            key=f"{prefix}_breakout_lookback",
        )
        min_regime_history = st.number_input(
            "Minimum regime history",
            min_value=1,
            value=252,
            key=f"{prefix}_min_regime_history",
        )
        regime_min_trades = st.number_input(
            "Regime analysis minimum trades",
            min_value=1,
            value=20,
            key=f"{prefix}_regime_min_trades",
        )

    with st.expander("Execution Model", expanded=False):
        starting_capital = st.number_input(
            "Starting capital",
            min_value=1.0,
            value=100000.0,
            step=1000.0,
            key=f"{prefix}_starting_capital",
        )
        max_capital_fraction = st.number_input(
            "Max capital fraction",
            min_value=0.01,
            max_value=1.0,
            value=1.0,
            step=0.01,
            key=f"{prefix}_max_capital_fraction",
        )
        max_avg_daily_volume_fraction = st.number_input(
            "Max avg daily volume fraction",
            min_value=0.001,
            max_value=1.0,
            value=0.05,
            step=0.001,
            format="%.3f",
            key=f"{prefix}_max_avg_daily_volume_fraction",
        )
        half_spread_bps = st.number_input(
            "Half spread (bps)",
            min_value=0.0,
            value=0.5,
            step=0.1,
            key=f"{prefix}_half_spread_bps",
        )
        min_slippage_bps = st.number_input(
            "Min slippage (bps)",
            min_value=0.0,
            value=0.5,
            step=0.1,
            key=f"{prefix}_min_slippage_bps",
        )
        max_slippage_bps = st.number_input(
            "Max slippage (bps)",
            min_value=0.0,
            value=3.0,
            step=0.1,
            key=f"{prefix}_max_slippage_bps",
        )
        commission_per_share = st.number_input(
            "Commission per share",
            min_value=0.0,
            value=0.005,
            step=0.001,
            format="%.4f",
            key=f"{prefix}_commission_per_share",
        )
        min_commission_per_order = st.number_input(
            "Minimum commission per order",
            min_value=0.0,
            value=1.0,
            step=0.1,
            key=f"{prefix}_min_commission_per_order",
        )
        expected_commission_rate = st.number_input(
            "Expected commission rate",
            min_value=0.0,
            value=0.00002,
            step=0.00001,
            format="%.5f",
            key=f"{prefix}_expected_commission_rate",
        )
        execution_reproducible = st.checkbox(
            "Execution model reproducible",
            value=True,
            key=f"{prefix}_execution_reproducible",
        )
        execution_seed = st.number_input(
            "Execution model seed",
            min_value=0,
            value=20260330,
            key=f"{prefix}_execution_seed",
        )

    with st.expander("Random Agent", expanded=False):
        random_reproducible = st.checkbox(
            "Random agent reproducible",
            value=True,
            key=f"{prefix}_random_reproducible",
        )
        random_seed = st.number_input(
            "Random agent seed",
            min_value=0,
            value=42,
            key=f"{prefix}_random_seed",
        )

    with st.expander("Monte Carlo", expanded=False):
        simulated_transaction_cost = st.number_input(
            "Simulated transaction cost",
            min_value=0.0,
            value=0.00082,
            step=0.00001,
            format="%.5f",
            key=f"{prefix}_simulated_transaction_cost",
        )
        monte_carlo_reproducible = st.checkbox(
            "Monte Carlo reproducible",
            value=True,
            key=f"{prefix}_monte_carlo_reproducible",
        )
        monte_carlo_seed = st.number_input(
            "Monte Carlo seed",
            min_value=0,
            value=42,
            key=f"{prefix}_monte_carlo_seed",
        )
        monte_carlo_batch_size = st.number_input(
            "Monte Carlo batch size",
            min_value=1,
            value=512,
            key=f"{prefix}_monte_carlo_batch_size",
        )

    with st.expander("Robustness", expanded=False):
        robustness_simulations = st.number_input(
            "Robustness simulations per run",
            min_value=1,
            value=5000,
            key=f"{prefix}_robustness_simulations",
        )
        robustness_outer_runs = st.number_input(
            "Robustness outer runs",
            min_value=1,
            value=100,
            key=f"{prefix}_robustness_outer_runs",
        )
        robustness_base_seed = st.number_input(
            "Robustness base seed",
            min_value=0,
            value=100,
            key=f"{prefix}_robustness_base_seed",
        )
        robustness_progress_every = st.number_input(
            "Robustness progress every",
            min_value=1,
            value=1,
            key=f"{prefix}_robustness_progress_every",
        )
        min_reliable_outer_runs = st.number_input(
            "Minimum reliable outer runs",
            min_value=1,
            value=30,
            key=f"{prefix}_min_reliable_outer_runs",
        )
        min_reliable_sims = st.number_input(
            "Minimum reliable simulations per run",
            min_value=1,
            value=1000,
            key=f"{prefix}_min_reliable_sims",
        )

    env_overrides.update(
        {
            "MOMENTUM_LOOKBACK_DAYS": str(int(momentum_lookback)),
            "BREAKOUT_LOOKBACK_DAYS": str(int(breakout_lookback)),
            "MIN_REGIME_HISTORY": str(int(min_regime_history)),
            "REGIME_MIN_TRADES": str(int(regime_min_trades)),
            "STARTING_CAPITAL": str(float(starting_capital)),
            "MAX_CAPITAL_FRACTION": str(float(max_capital_fraction)),
            "MAX_AVG_DAILY_VOLUME_FRACTION": str(float(max_avg_daily_volume_fraction)),
            "HALF_SPREAD_BPS": str(float(half_spread_bps)),
            "MIN_SLIPPAGE_BPS": str(float(min_slippage_bps)),
            "MAX_SLIPPAGE_BPS": str(float(max_slippage_bps)),
            "COMMISSION_PER_SHARE": str(float(commission_per_share)),
            "MIN_COMMISSION_PER_ORDER": str(float(min_commission_per_order)),
            "EXPECTED_COMMISSION_RATE": str(float(expected_commission_rate)),
            "EXECUTION_MODEL_REPRODUCIBLE": to_env_flag(execution_reproducible),
            "EXECUTION_MODEL_SEED": str(int(execution_seed)),
            "RANDOM_AGENT_REPRODUCIBLE": to_env_flag(random_reproducible),
            "RANDOM_AGENT_SEED": str(int(random_seed)),
            "SIMULATED_TRANSACTION_COST": str(float(simulated_transaction_cost)),
            "MONTE_CARLO_REPRODUCIBLE": to_env_flag(monte_carlo_reproducible),
            "MONTE_CARLO_SEED": str(int(monte_carlo_seed)),
            "MONTE_CARLO_BATCH_SIZE": str(int(monte_carlo_batch_size)),
            "ROBUSTNESS_SIMULATIONS_PER_RUN": str(int(robustness_simulations)),
            "ROBUSTNESS_OUTER_RUNS": str(int(robustness_outer_runs)),
            "ROBUSTNESS_BASE_SEED": str(int(robustness_base_seed)),
            "ROBUSTNESS_PROGRESS_EVERY": str(int(robustness_progress_every)),
            "MIN_RELIABLE_OUTER_RUNS": str(int(min_reliable_outer_runs)),
            "MIN_RELIABLE_SIMULATIONS_PER_RUN": str(int(min_reliable_sims)),
        }
    )
    return env_overrides


def run_single_from_ui(ticker: str, env_overrides: dict[str, str]) -> WorkflowRunResult:
    """Execute the single-ticker workflow and stream updates into the page."""
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    log_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0.0, text="Waiting to start...")
    log_lines: list[str] = []
    total_steps = 28

    def handle_event(event: WorkflowEvent) -> None:
        nonlocal total_steps
        if event.total_steps is not None:
            total_steps = event.total_steps
        if event.kind == "step_started":
            step_position = event.step_index or 1
            progress_bar.progress(
                min(step_position / max(total_steps, 1), 1.0),
                text=f"Step {step_position}/{total_steps}: {event.step_label}",
            )
            status_placeholder.info(event.message)
        elif event.kind == "log" and event.message:
            log_lines.append(event.message)
            log_placeholder.code("\n".join(log_lines[-200:]), language="text")
        elif event.kind == "workflow_finished":
            progress_bar.progress(1.0, text=event.message)
            if "completed" in event.message.lower():
                status_placeholder.success(event.message)
            else:
                status_placeholder.error(event.message)

    result = run_single_ticker_pipeline(
        ticker=ticker,
        env_overrides=env_overrides,
        event_callback=handle_event,
        interactive_chart_open=False,
    )
    return result


def run_walk_forward_from_ui(tickers: list[str], env_overrides: dict[str, str]) -> WorkflowRunResult:
    """Execute the walk-forward workflow and stream updates into the page."""
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    log_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0.0, text="Waiting to start...")
    log_lines: list[str] = []

    def handle_event(event: WorkflowEvent) -> None:
        if event.kind == "step_started":
            progress_bar.progress(0.25, text=event.message)
            status_placeholder.info(event.message)
        elif event.kind == "log" and event.message:
            log_lines.append(event.message)
            log_placeholder.code("\n".join(log_lines[-250:]), language="text")
        elif event.kind == "workflow_finished":
            progress_bar.progress(1.0, text=event.message)
            if "completed" in event.message.lower():
                status_placeholder.success(event.message)
            else:
                status_placeholder.error(event.message)

    return run_walk_forward_pipeline(
        tickers=tickers,
        env_overrides=env_overrides,
        event_callback=handle_event,
    )


def main() -> None:
    """Render the Streamlit interface."""
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None

    workflow = st.sidebar.radio(
        "Workflow",
        [
            "Single-Ticker Full Pipeline",
            "Multi-Asset Walk-Forward Evaluation",
        ],
    )

    available_tickers = existing_tickers()
    default_ticker = available_tickers[0] if available_tickers else DEFAULT_SINGLE_TICKER
    st.sidebar.markdown("**Current strategy scope**")
    st.sidebar.multiselect(
        "Strategies preserved from the real project",
        options=AGENT_ORDER + [BENCHMARK_NAME],
        default=AGENT_ORDER + [BENCHMARK_NAME],
        format_func=lambda name: AGENT_DISPLAY_NAMES.get(name, name.replace("_", " ").title()),
        disabled=True,
        help="The current project workflow always runs the full strategy set plus the buy-and-hold benchmark.",
    )
    st.sidebar.info(
        "Date range controls are not shown because the underlying project runs on full-history inputs. "
        "The interface preserves that behavior instead of silently changing it."
    )

    current_result: WorkflowRunResult | None = st.session_state.get("last_result")

    if workflow == "Single-Ticker Full Pipeline":
        with st.sidebar.form("single_workflow_form"):
            ticker = st.text_input("Ticker", value=default_ticker)
            shared_env = render_shared_parameter_controls("single")
            run_pipeline = st.form_submit_button("Run Full Pipeline", type="primary")

        if run_pipeline:
            normalized_ticker = ticker.strip().upper()
            if not normalized_ticker:
                st.error("Enter a ticker symbol before running the pipeline.")
            else:
                result = run_single_from_ui(normalized_ticker, shared_env)
                st.session_state["last_result"] = result
                st.session_state["last_single_ticker"] = normalized_ticker
                current_result = result

        active_ticker = ticker.strip().upper() or st.session_state.get("last_single_ticker", default_ticker)
        tabs = st.tabs(["Summary", "Metrics", "Charts", "Trades", "Logs / Diagnostics", "Files"])
        with tabs[0]:
            render_single_workflow_summary(
                active_ticker,
                current_result if current_result and current_result.metadata.get("ticker") == active_ticker else None,
            )
        with tabs[1]:
            render_single_metrics_tab(active_ticker)
        with tabs[2]:
            render_single_charts_tab(active_ticker)
        with tabs[3]:
            render_single_trades_tab(active_ticker)
        with tabs[4]:
            if current_result and current_result.metadata.get("ticker") == active_ticker:
                render_logs_tab(current_result)
            else:
                render_logs_tab(None)
        with tabs[5]:
            render_files_tab(
                ticker_data_files(active_ticker) + ticker_chart_files(active_ticker),
                key_prefix=f"{active_ticker}_files",
            )

    else:
        default_walk_forward_text = ",".join(DEFAULT_WALK_FORWARD_TICKERS)
        with st.sidebar.form("walk_forward_form"):
            tickers_text = st.text_area(
                "Tickers",
                value=default_walk_forward_text,
                help="Comma-separated tickers for the preserved walk-forward workflow.",
            )
            with st.expander("Walk-Forward Parameters", expanded=False):
                test_bars = st.number_input("Test bars", min_value=1, value=504, key="wf_test_bars")
                step_bars = st.number_input("Step bars", min_value=1, value=252, key="wf_step_bars")
                min_trades = st.number_input("Minimum trades per panel", min_value=0, value=5, key="wf_min_trades")
                simulations_per_run = st.number_input(
                    "Simulations per run",
                    min_value=1,
                    value=1000,
                    key="wf_simulations_per_run",
                )
                outer_runs = st.number_input("Outer runs", min_value=1, value=50, key="wf_outer_runs")
                base_seed = st.number_input("Base seed", min_value=0, value=5000, key="wf_base_seed")
                progress_every = st.number_input(
                    "Progress every",
                    min_value=1,
                    value=5,
                    key="wf_progress_every",
                )
                refresh_data = st.checkbox("Refresh prerequisite data", value=True, key="wf_refresh_data")

            shared_env = render_shared_parameter_controls("walk_forward_shared")
            run_walk_forward = st.form_submit_button("Run Walk-Forward Evaluation", type="primary")

        if run_walk_forward:
            selected_tickers = split_tickers(tickers_text)
            if not selected_tickers:
                st.error("Enter at least one ticker for the walk-forward workflow.")
            else:
                walk_forward_env = {
                    **shared_env,
                    "WALK_FORWARD_TEST_BARS": str(int(test_bars)),
                    "WALK_FORWARD_STEP_BARS": str(int(step_bars)),
                    "WALK_FORWARD_MIN_TRADES": str(int(min_trades)),
                    "WALK_FORWARD_SIMULATIONS_PER_RUN": str(int(simulations_per_run)),
                    "WALK_FORWARD_OUTER_RUNS": str(int(outer_runs)),
                    "WALK_FORWARD_BASE_SEED": str(int(base_seed)),
                    "WALK_FORWARD_PROGRESS_EVERY": str(int(progress_every)),
                    "WALK_FORWARD_REFRESH_DATA": to_env_flag(refresh_data),
                }
                result = run_walk_forward_from_ui(selected_tickers, walk_forward_env)
                st.session_state["last_result"] = result
                st.session_state["last_walk_forward_tickers"] = ",".join(selected_tickers)
                current_result = result

        wf_tabs = st.tabs(["Summary", "Panel Results", "Agent Summary", "Raw Runs", "Logs / Diagnostics", "Files"])
        with wf_tabs[0]:
            render_walk_forward_summary(
                current_result if current_result and current_result.workflow_name == "Multi-Asset Walk-Forward Evaluation" else None
            )
        with wf_tabs[1]:
            render_walk_forward_tab(
                "Walk-Forward Panel Summary",
                walk_forward_output_paths()["Walk-Forward Panel Summary"],
                key_prefix="wf_panel",
            )
        with wf_tabs[2]:
            render_walk_forward_tab(
                "Walk-Forward Agent Summary",
                walk_forward_output_paths()["Walk-Forward Agent Summary"],
                key_prefix="wf_agent",
            )
        with wf_tabs[3]:
            render_walk_forward_tab(
                "Walk-Forward Runs",
                walk_forward_output_paths()["Walk-Forward Runs"],
                key_prefix="wf_runs",
            )
        with wf_tabs[4]:
            if current_result and current_result.workflow_name == "Multi-Asset Walk-Forward Evaluation":
                render_logs_tab(current_result)
            else:
                render_logs_tab(None)
        with wf_tabs[5]:
            render_files_tab(
                list(walk_forward_output_paths().values()),
                key_prefix="wf_files",
            )


if __name__ == "__main__":
    if streamlit_runtime_exists():
        main()
    else:
        print("This interface app must be launched with: streamlit run app.py")
