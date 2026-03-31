"""Run the single-ticker research pipeline."""

from pathlib import Path
import os
import subprocess
import sys


def ask_for_ticker() -> str:
    """Ask the user for a ticker symbol and clean up the input."""
    ticker_input = input(
        "Enter a ticker symbol (for example SPY, QQQ, AAPL, BTC-USD): "
    ).strip()

    if not ticker_input:
        raise ValueError("You must enter a ticker symbol before the pipeline can run.")

    return ticker_input.upper()


def build_pipeline_steps() -> list[tuple[str, str]]:
    """Return the scripts in the order they should be run."""
    return [
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
        ("Opening charts...", "open_charts.py"),
    ]


def build_runtime_context() -> tuple[Path, str, dict[str, str]]:
    """Build the shared runtime context for child scripts."""
    script_dir = Path(__file__).resolve().parent
    python_executable = sys.executable
    env = os.environ.copy()
    env["SHOW_PLOTS"] = "0"
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(script_dir.parent / ".matplotlib")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return script_dir, python_executable, env


def run_script(
    python_executable: str,
    script_dir: Path,
    script_name: str,
    env: dict[str, str],
) -> None:
    """Run one Python script using the current interpreter."""
    subprocess.run(
        [python_executable, script_name],
        cwd=script_dir,
        env=env,
        check=True,
    )


def run_single_ticker_pipeline(
    script_dir: Path,
    python_executable: str,
    env: dict[str, str],
) -> None:
    """Run the original single-ticker pipeline."""
    try:
        ticker = ask_for_ticker()
    except ValueError as error:
        print(f"Error: {error}")
        return

    env["TICKER"] = ticker

    for progress_message, script_name in build_pipeline_steps():
        print(progress_message, flush=True)

        try:
            run_script(
                python_executable=python_executable,
                script_dir=script_dir,
                script_name=script_name,
                env=env,
            )
        except subprocess.CalledProcessError as error:
            print(f"Error: {script_name} failed with exit code {error.returncode}.")
            print("Pipeline stopped.")
            return
        except FileNotFoundError as error:
            print(f"Error: {error}")
            print("Pipeline stopped.")
            return

    print("Pipeline complete", flush=True)
    print(f"Outputs saved for {ticker}", flush=True)
    print("Charts saved in Charts/", flush=True)


def main() -> None:
    """Run the single-ticker workflow and stop on the first error."""
    script_dir, python_executable, env = build_runtime_context()
    run_single_ticker_pipeline(
        script_dir=script_dir,
        python_executable=python_executable,
        env=env,
    )


if __name__ == "__main__":
    main()
