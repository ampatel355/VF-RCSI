"""Convenience CLI entrypoint for the fast single-ticker test pipeline."""

from __future__ import annotations

import os

try:
    from workflow_runner import run_single_ticker_pipeline
except ModuleNotFoundError:
    from Code.workflow_runner import run_single_ticker_pipeline


def main() -> None:
    """Run the single-ticker workflow with the fast-test preset enabled."""
    ticker = os.environ.get("TICKER", "SPY").strip().upper() or "SPY"
    # Fast pipeline runs are usually interactive debugging sessions, so charts
    # should display by default unless the caller explicitly turns them off.
    show_plots = os.environ.get("SHOW_PLOTS", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
    save_outputs = os.environ.get("SAVE_OUTPUTS", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
    minimal_mode = os.environ.get("MINIMAL_MODE", "1").strip().lower() not in {"0", "false", "no", "n", "off"}
    result = run_single_ticker_pipeline(
        ticker=ticker,
        fast_test_mode=True,
        show_plots=show_plots,
        save_outputs=save_outputs,
        minimal_mode=minimal_mode,
    )
    print(
        {
            "ticker": ticker,
            "success": result.success,
            "steps": len(result.step_results),
            "duration_seconds": result.duration_seconds,
            "error_message": result.error_message,
            "run_mode": result.metadata.get("run_mode", "fast_test"),
            "show_plots": show_plots,
            "save_outputs": save_outputs,
            "profile": result.metadata.get("profile", "minimal"),
        }
    )


if __name__ == "__main__":
    main()
