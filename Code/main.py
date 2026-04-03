"""Primary CLI entrypoint for the simplified research pipeline."""

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


def ask_yes_no(prompt: str, *, default: bool) -> bool:
    """Ask a simple yes/no question with a readable default."""
    default_label = "Y/n" if default else "y/N"
    user_input = input(f"{prompt} [{default_label}]: ").strip().lower()
    if not user_input:
        return default
    if user_input in {"y", "yes"}:
        return True
    if user_input in {"n", "no"}:
        return False
    raise ValueError("Please answer yes or no.")


def ask_for_mode() -> str:
    """Ask which workflow to run."""
    mode_input = input(
        "\nChoose a workflow:\n"
        "1. Single-ticker research pipeline\n"
        "2. Multi-asset walk-forward evaluation\n"
        "Enter 1 or 2: "
    ).strip()
    if not mode_input:
        return "1"
    if mode_input not in {"1", "2"}:
        raise ValueError("You must enter 1 or 2.")
    return mode_input


def ask_for_ticker() -> str:
    """Ask for the active research ticker."""
    default_ticker = os.environ.get("TICKER", "SPY").strip().upper()
    ticker_input = input(
        "Enter a ticker symbol "
        f"[Enter for {default_ticker}]: "
    ).strip().upper()
    return ticker_input or default_ticker


def ask_for_walk_forward_tickers() -> list[str]:
    """Ask for the walk-forward universe."""
    default_universe = os.environ.get(
        "WALK_FORWARD_TICKERS",
        ",".join(DEFAULT_WALK_FORWARD_TICKERS),
    ).strip()
    ticker_input = input(
        "Enter comma-separated tickers for walk-forward "
        f"[Enter for {default_universe}]: "
    ).strip()
    ticker_input = ticker_input or default_universe
    tickers = [ticker.strip().upper() for ticker in ticker_input.split(",") if ticker.strip()]
    if not tickers:
        raise ValueError("You must enter at least one ticker.")
    return tickers


TIMEFRAME_CHOICES = {
    "1": ("1d", "Daily"),
    "2": ("4h", "4-Hour"),
    "3": ("1h", "Hourly"),
}


def ask_for_timeframe() -> tuple[str, str]:
    """Ask which research timeframe to use.

    Returns (interval, label) — e.g. ("1h", "Hourly").
    """
    current_interval = os.environ.get("RESEARCH_INTERVAL", "1d").strip().lower()
    default_key = "1"
    for key, (interval, _label) in TIMEFRAME_CHOICES.items():
        if interval == current_interval:
            default_key = key
            break

    print("\nSelect a research timeframe:")
    for key, (interval, label) in TIMEFRAME_CHOICES.items():
        marker = " (default)" if key == default_key else ""
        print(f"  {key}. {label} ({interval}){marker}")

    choice = input(f"Enter 1, 2, or 3 [Enter for {default_key}]: ").strip()
    if not choice:
        choice = default_key
    if choice not in TIMEFRAME_CHOICES:
        raise ValueError("Please enter 1, 2, or 3.")
    return TIMEFRAME_CHOICES[choice]


def ask_for_fast_mode() -> bool:
    """Ask whether to use the reduced-cost fast test preset."""
    env_default = os.environ.get("FAST_TEST_MODE", "").strip().lower()
    default_fast = env_default in {"1", "true", "yes", "y", "on"}
    return ask_yes_no(
        "Use fast testing mode? This keeps the strategy logic the same but reduces Monte Carlo depth.",
        default=default_fast,
    )


def print_event(event: WorkflowEvent) -> None:
    """Render workflow progress events to the terminal."""
    if event.kind == "step_started":
        print(event.message, flush=True)
    elif event.kind == "log" and event.message:
        print(event.message, flush=True)


def run_single_ticker_cli() -> None:
    """Run the simplified single-ticker workflow."""
    try:
        ticker = ask_for_ticker()
        interval, timeframe_label = ask_for_timeframe()
        fast_mode = ask_for_fast_mode()
        minimal_mode = ask_yes_no(
            "Use minimal research mode? This runs only the core analysis and two display charts.",
            default=True,
        )
        show_plots = ask_yes_no(
            "Display charts during the run?",
            default=True,
        )
        save_outputs = ask_yes_no(
            "Save charts to disk? This is off by default to avoid clutter.",
            default=False,
        )
    except ValueError as error:
        print(f"Error: {error}")
        return

    # Pass the chosen timeframe into the pipeline subprocess environment so
    # every script in the pipeline sees the same RESEARCH_INTERVAL.
    timeframe_overrides = {
        "RESEARCH_INTERVAL": interval,
        "RESEARCH_TIMEFRAME_LABEL": timeframe_label,
    }

    result = run_single_ticker_pipeline(
        ticker=ticker,
        env_overrides=timeframe_overrides,
        event_callback=print_event,
        fast_test_mode=fast_mode,
        show_plots=show_plots,
        save_outputs=save_outputs,
        minimal_mode=minimal_mode,
    )
    if not result.success:
        print(result.error_message or "Pipeline stopped.", flush=True)
        return

    suffix = "" if interval == "1d" else f"_{interval}"
    print("\nPipeline complete.", flush=True)
    print(f"Ticker: {ticker}", flush=True)
    print(f"Timeframe: {timeframe_label} ({interval})", flush=True)
    print(f"Run mode: {'fast test' if fast_mode else 'full research'}", flush=True)
    print(f"Profile: {'minimal' if minimal_mode else 'extended'}", flush=True)
    print(f"Charts displayed: {'yes' if show_plots else 'no'}", flush=True)
    print(f"Charts saved: {'yes' if save_outputs else 'no'}", flush=True)
    print(f"Core summary file: Data_Clean{suffix}/{ticker}_full_comparison.csv", flush=True)


def run_walk_forward_cli() -> None:
    """Run the walk-forward workflow."""
    try:
        tickers = ask_for_walk_forward_tickers()
        interval, timeframe_label = ask_for_timeframe()
        fast_mode = ask_for_fast_mode()
    except ValueError as error:
        print(f"Error: {error}")
        return

    timeframe_overrides = {
        "RESEARCH_INTERVAL": interval,
        "RESEARCH_TIMEFRAME_LABEL": timeframe_label,
    }

    result = run_walk_forward_pipeline(
        tickers=tickers,
        env_overrides=timeframe_overrides,
        event_callback=print_event,
        fast_test_mode=fast_mode,
    )
    if not result.success:
        print(result.error_message or "Walk-forward workflow stopped.", flush=True)
        return

    suffix = "" if interval == "1d" else f"_{interval}"
    print("\nWalk-forward evaluation complete.", flush=True)
    print(f"Timeframe: {timeframe_label} ({interval})", flush=True)
    print(f"Run mode: {'fast test' if fast_mode else 'full research'}", flush=True)
    print("Key files:", flush=True)
    print(f"- Data_Clean{suffix}/multi_asset_walk_forward_runs.csv", flush=True)
    print(f"- Data_Clean{suffix}/multi_asset_walk_forward_panel_summary.csv", flush=True)
    print(f"- Data_Clean{suffix}/multi_asset_walk_forward_agent_summary.csv", flush=True)


def main() -> None:
    """Run the requested research workflow."""
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
