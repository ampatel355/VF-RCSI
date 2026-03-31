"""Run the project's CLI workflows while delegating execution to shared runners."""

from __future__ import annotations

import os

try:
    from workflow_runner import (
        DEFAULT_WALK_FORWARD_TICKERS,
        WorkflowEvent,
        run_single_ticker_pipeline,
        run_walk_forward_pipeline,
    )
except ModuleNotFoundError:
    from Code.workflow_runner import (
        DEFAULT_WALK_FORWARD_TICKERS,
        WorkflowEvent,
        run_single_ticker_pipeline,
        run_walk_forward_pipeline,
    )


def ask_for_mode() -> str:
    """Ask which top-level workflow to run."""
    mode_input = input(
        "\nChoose a workflow:\n"
        "1. Single-ticker full pipeline\n"
        "2. Multi-asset walk-forward evaluation\n"
        "Enter 1 or 2: "
    ).strip()

    if not mode_input:
        return "1"

    if mode_input not in {"1", "2"}:
        raise ValueError("You must enter 1 for the ticker pipeline or 2 for walk-forward.")

    return mode_input


def ask_for_ticker() -> str:
    """Ask the user for a ticker symbol and clean up the input."""
    default_ticker = os.environ.get("TICKER", "").strip().upper()
    prompt = "Enter a ticker symbol (for example SPY, QQQ, AAPL, BTC-USD, EURUSD=X)"
    if default_ticker:
        prompt += f" [Enter for {default_ticker}]"
    prompt += ": "

    ticker_input = input(prompt).strip()

    if not ticker_input and default_ticker:
        return default_ticker

    if not ticker_input:
        raise ValueError("You must enter a ticker symbol before the pipeline can run.")

    return ticker_input.upper()


def ask_for_walk_forward_tickers() -> list[str]:
    """Ask for an optional comma-separated walk-forward universe."""
    default_universe = os.environ.get(
        "WALK_FORWARD_TICKERS",
        ",".join(DEFAULT_WALK_FORWARD_TICKERS),
    ).strip()
    ticker_input = input(
        "Enter comma-separated tickers for walk-forward "
        f"[Enter for {default_universe}]: "
    ).strip()

    if not ticker_input:
        ticker_input = default_universe

    tickers = [ticker.strip().upper() for ticker in ticker_input.split(",") if ticker.strip()]
    if not tickers:
        raise ValueError("You must enter at least one valid ticker symbol.")

    return tickers


def print_event(event: WorkflowEvent) -> None:
    """Render workflow progress events to the terminal."""
    if event.kind == "step_started":
        print(event.message, flush=True)
    elif event.kind == "log" and event.message:
        print(event.message, flush=True)


def run_single_ticker_cli() -> None:
    """Run the single-ticker CLI flow."""
    try:
        ticker = ask_for_ticker()
    except ValueError as error:
        print(f"Error: {error}")
        return

    result = run_single_ticker_pipeline(
        ticker=ticker,
        event_callback=print_event,
        interactive_chart_open=True,
    )
    if not result.success:
        print(result.error_message or "Pipeline stopped.", flush=True)
        return

    print("Pipeline complete", flush=True)
    print(f"Outputs saved for {ticker}", flush=True)
    print("Charts saved in Charts/", flush=True)


def run_walk_forward_cli() -> None:
    """Run the multi-asset walk-forward CLI flow."""
    try:
        tickers = ask_for_walk_forward_tickers()
    except ValueError as error:
        print(f"Error: {error}")
        return

    result = run_walk_forward_pipeline(
        tickers=tickers,
        event_callback=print_event,
    )
    if not result.success:
        print(result.error_message or "Walk-forward workflow stopped.", flush=True)
        return

    print("Walk-forward evaluation complete", flush=True)
    print("Outputs saved in Data_Clean/", flush=True)
    print("Key files:", flush=True)
    print("- Data_Clean/multi_asset_walk_forward_runs.csv", flush=True)
    print("- Data_Clean/multi_asset_walk_forward_panel_summary.csv", flush=True)
    print("- Data_Clean/multi_asset_walk_forward_agent_summary.csv", flush=True)


def main() -> None:
    """Run the selected CLI workflow and stop on the first error."""
    try:
        mode = ask_for_mode()
    except ValueError as error:
        print(f"Error: {error}")
        return

    if mode == "2":
        run_walk_forward_cli()
        return

    run_single_ticker_cli()


if __name__ == "__main__":
    main()
