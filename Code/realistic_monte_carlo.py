"""Run a more realistic Monte Carlo simulation for all active trading agents."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from strategy_config import AGENT_ORDER
except ModuleNotFoundError:
    from Code.strategy_config import AGENT_ORDER


# Read the active ticker from the environment, or fall back to SPY.
ticker = os.environ.get("TICKER", "SPY")

# These settings are easy to change at the top of the script.
NUMBER_OF_SIMULATIONS = 5000
TRANSACTION_COST = 0.001  # 0.001 means 0.1% cost per trade.
RANDOM_SEED = 42


def resolve_data_clean_dir(project_root: Path) -> Path:
    """Return the clean-data folder, supporting either lower- or upper-case names."""
    uppercase_dir = project_root / "Data_Clean"
    lowercase_dir = project_root / "data_clean"

    if uppercase_dir.exists():
        return uppercase_dir
    if lowercase_dir.exists():
        return lowercase_dir

    # If neither folder exists yet, create the project's preferred folder name.
    uppercase_dir.mkdir(parents=True, exist_ok=True)
    return uppercase_dir


def load_csv_checked(input_path: Path, required_columns: list[str]) -> pd.DataFrame:
    """Load a CSV file and confirm that it has the columns we need."""
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError(f"The input file is empty: {input_path}")

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(
            f"Missing required columns in {input_path}: {', '.join(missing_columns)}"
        )

    return df


def load_trade_data(input_path: Path) -> pd.DataFrame:
    """Load one trade file and prepare the return column for simulation."""
    df = load_csv_checked(
        input_path,
        required_columns=["entry_date", "exit_date", "return"],
    )

    # Parse the dates so the file is checked more carefully.
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")

    # Make sure returns are numeric before running the simulation.
    df["return"] = pd.to_numeric(df["return"], errors="coerce")
    df = df.dropna(subset=["return"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No usable trade returns were found in: {input_path}")

    return df


def adjust_returns_for_cost(
    returns: np.ndarray,
    transaction_cost: float,
    agent_name: str,
) -> np.ndarray:
    """Subtract a fixed transaction cost from each trade return."""
    adjusted_returns = returns - transaction_cost

    # Log returns require 1 + return to stay positive.
    if np.any(adjusted_returns <= -1):
        smallest_value = float(adjusted_returns.min())
        raise ValueError(
            f"{agent_name} has an adjusted trade return of {smallest_value:.6f}, "
            "which is too low for log-return conversion. "
            "Try a smaller transaction cost or inspect the trade file."
        )

    return adjusted_returns


def convert_to_log_returns(adjusted_returns: np.ndarray) -> np.ndarray:
    """Convert normal trade returns into log returns."""
    return np.log1p(adjusted_returns)


def calculate_cumulative_return_from_log_returns(log_returns: np.ndarray) -> float:
    """Convert a sequence of log returns back into one cumulative return."""
    return float(np.expm1(log_returns.sum()))


def simulate_log_return_paths(
    log_returns: np.ndarray,
    number_of_simulations: int,
    rng: np.random.Generator,
) -> pd.Series:
    """Create many simulated final returns by resampling trade log returns."""
    if len(log_returns) == 0:
        raise ValueError("Cannot run simulations without at least one trade.")

    simulated_final_returns = []
    number_of_trades = len(log_returns)

    for _ in range(number_of_simulations):
        sampled_log_returns = rng.choice(log_returns, size=number_of_trades, replace=True)
        simulated_final_returns.append(
            calculate_cumulative_return_from_log_returns(sampled_log_returns)
        )

    return pd.Series(simulated_final_returns, name="simulated_return")


def save_simulation_results(
    output_path: Path,
    simulated_returns: pd.Series,
) -> None:
    """Save the full simulation results for one agent."""
    simulated_returns.to_frame().to_csv(output_path, index=False)


def run_agent_simulation(
    data_clean_dir: Path,
    agent_name: str,
    rng: np.random.Generator,
) -> dict[str, float | str]:
    """Run the realistic Monte Carlo simulation for one trading agent."""
    input_path = data_clean_dir / f"{ticker}_{agent_name}_trades.csv"
    output_path = data_clean_dir / f"{ticker}_{agent_name}_realistic_monte_carlo.csv"

    trade_df = load_trade_data(input_path)
    raw_returns = trade_df["return"].to_numpy(dtype=float)

    adjusted_returns = adjust_returns_for_cost(
        returns=raw_returns,
        transaction_cost=TRANSACTION_COST,
        agent_name=agent_name,
    )
    log_returns = convert_to_log_returns(adjusted_returns)

    # The real strategy result uses the same transaction-cost-adjusted framework
    # as the simulation, which keeps the comparison fair.
    real_return = calculate_cumulative_return_from_log_returns(log_returns)
    simulated_returns = simulate_log_return_paths(
        log_returns=log_returns,
        number_of_simulations=NUMBER_OF_SIMULATIONS,
        rng=rng,
    )

    save_simulation_results(output_path, simulated_returns)

    median_simulated_return = float(simulated_returns.median())
    mean_simulated_return = float(simulated_returns.mean())
    actual_percentile = float((simulated_returns <= real_return).mean() * 100)

    print(f"\nAgent: {agent_name}")
    print(f"Real return: {real_return:.6f}")
    print(f"Median simulated return: {median_simulated_return:.6f}")
    print(f"Mean simulated return: {mean_simulated_return:.6f}")
    print(f"Actual percentile: {actual_percentile:.2f}")
    print(f"Transaction cost used: {TRANSACTION_COST:.6f}")

    return {
        "agent": agent_name,
        "real_return": real_return,
        "median_simulated_return": median_simulated_return,
        "mean_simulated_return": mean_simulated_return,
        "actual_percentile": actual_percentile,
        "transaction_cost": TRANSACTION_COST,
    }


def main() -> None:
    """Run the realistic Monte Carlo simulation for all active agents."""
    project_root = Path(__file__).resolve().parents[1]
    data_clean_dir = resolve_data_clean_dir(project_root)
    summary_output_path = data_clean_dir / f"{ticker}_realistic_monte_carlo_summary.csv"

    rng = np.random.default_rng(RANDOM_SEED)
    summary_rows = []

    for agent_name in AGENT_ORDER:
        summary_rows.append(
            run_agent_simulation(
                data_clean_dir=data_clean_dir,
                agent_name=agent_name,
                rng=rng,
            )
        )

    summary_df = pd.DataFrame(
        summary_rows,
        columns=[
            "agent",
            "real_return",
            "median_simulated_return",
            "mean_simulated_return",
            "actual_percentile",
            "transaction_cost",
        ],
    )

    summary_df.to_csv(summary_output_path, index=False)

    print("\nSummary table:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
