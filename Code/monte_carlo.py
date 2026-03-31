"""Run a timing-based Monte Carlo null model for all active strategies."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from execution_model import EXPECTED_ROUND_TRIP_EXECUTION_COST, TRADE_RETURNS_ALREADY_NET
    from research_metrics import calculate_p_value_prominence
    from strategy_config import AGENT_ORDER
except ModuleNotFoundError:
    from Code.execution_model import EXPECTED_ROUND_TRIP_EXECUTION_COST, TRADE_RETURNS_ALREADY_NET
    from Code.research_metrics import calculate_p_value_prominence
    from Code.strategy_config import AGENT_ORDER


# Read the active ticker from the environment, or fall back to SPY.
ticker = os.environ.get("TICKER", "SPY")

# These settings are easy to change at the top of the script.
NUMBER_OF_SIMULATIONS = 5000
TRANSACTION_COST = float(
    os.environ.get(
        "SIMULATED_TRANSACTION_COST",
        str(EXPECTED_ROUND_TRIP_EXECUTION_COST),
    )
)
REPRODUCIBLE = os.environ.get("MONTE_CARLO_REPRODUCIBLE", "1") == "1"
SEED = int(os.environ.get("MONTE_CARLO_SEED", "42"))
SIMULATION_BATCH_SIZE = int(os.environ.get("MONTE_CARLO_BATCH_SIZE", "512"))
NULL_MODEL_NAME = "random_timing_matched_duration"


def resolve_data_clean_dir(project_root: Path) -> Path:
    """Return the clean-data folder, supporting either naming style."""
    uppercase_dir = project_root / "Data_Clean"
    lowercase_dir = project_root / "data_clean"

    if uppercase_dir.exists():
        return uppercase_dir
    if lowercase_dir.exists():
        return lowercase_dir

    uppercase_dir.mkdir(parents=True, exist_ok=True)
    return uppercase_dir


def load_csv_checked(
    input_path: Path,
    required_columns: list[str],
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Load a CSV file and confirm that it has the columns we need."""
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    df = pd.read_csv(input_path)

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(
            f"Missing required columns in {input_path}: {', '.join(missing_columns)}"
        )

    if df.empty and not allow_empty:
        raise ValueError(f"The input file is empty: {input_path}")

    return df


def load_trade_data(input_path: Path, allow_empty: bool = False) -> pd.DataFrame:
    """Load one trade file and validate its dates and returns."""
    df = load_csv_checked(
        input_path,
        required_columns=["entry_date", "exit_date", "return"],
        allow_empty=allow_empty,
    )

    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    df["return"] = pd.to_numeric(df["return"], errors="coerce")
    if "position_value_fraction" in df.columns:
        df["position_value_fraction"] = pd.to_numeric(
            df["position_value_fraction"],
            errors="coerce",
        )
    else:
        df["position_value_fraction"] = 1.0

    if df.empty:
        return df.sort_values("exit_date", ascending=True).reset_index(drop=True)

    invalid_rows = df[
        df["entry_date"].isna() | df["exit_date"].isna() | df["return"].isna()
    ]
    if not invalid_rows.empty:
        raise ValueError(
            f"{input_path} contains {len(invalid_rows)} row(s) with invalid dates or returns."
        )

    df = df.sort_values("exit_date", ascending=True).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No usable trade data was found in: {input_path}")

    return df


def build_no_trade_simulated_returns(simulation_count: int) -> pd.Series:
    """Create a degenerate zero-return simulation series for no-trade strategies."""
    return pd.Series(
        np.zeros(simulation_count, dtype=float),
        name="simulated_cumulative_return",
    )


def extract_position_value_fractions(trade_df: pd.DataFrame) -> np.ndarray:
    """Return one portfolio-at-risk fraction for each realized trade."""
    fractions = trade_df["position_value_fraction"].to_numpy(dtype=float)

    if np.any(np.isnan(fractions)):
        raise ValueError("Trade data contains invalid position_value_fraction values.")
    if np.any(fractions <= 0):
        raise ValueError("Trade data contains non-positive position_value_fraction values.")
    if np.any(fractions > 1.0):
        raise ValueError("Trade data contains position_value_fraction values above 1.0.")

    return fractions


def load_market_data(input_path: Path) -> pd.DataFrame:
    """Load the ticker's open-price history used for random-timing simulations."""
    df = load_csv_checked(
        input_path,
        required_columns=["Date", "Open"],
    )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df = df.dropna(subset=["Date", "Open"]).sort_values("Date").reset_index(drop=True)

    if len(df) < 2:
        raise ValueError(
            f"Need at least two usable open-price rows to simulate trades: {input_path}"
        )

    if (df["Open"] <= 0).any():
        raise ValueError(f"Open prices must stay positive for simulation: {input_path}")

    return df


def adjust_trade_returns(
    raw_returns: np.ndarray,
    transaction_cost: float,
    input_path: Path,
) -> np.ndarray:
    """Subtract a fixed transaction cost from each trade return."""
    if TRADE_RETURNS_ALREADY_NET:
        adjusted_returns = raw_returns.astype(float, copy=True)
    else:
        adjusted_returns = raw_returns - transaction_cost

    # Log returns require 1 + adjusted_return to stay positive.
    if np.any(adjusted_returns <= -1):
        smallest_value = float(adjusted_returns.min())
        raise ValueError(
            f"{input_path} contains an adjusted return of {smallest_value:.6f}, "
            "which is too low for log-return conversion. "
            "Reduce the transaction cost or inspect the trade data."
        )

    return adjusted_returns


def convert_to_log_returns(adjusted_returns: np.ndarray) -> np.ndarray:
    """Convert normal trade returns into log returns."""
    return np.log1p(adjusted_returns)


def calculate_cumulative_return_from_log_returns(log_returns: np.ndarray) -> float:
    """Convert a sequence of log returns into a final cumulative return."""
    return float(np.expm1(log_returns.sum()))


def build_random_generator(reproducible: bool, seed: int) -> np.random.Generator:
    """Create either a reproducible or stochastic random number generator."""
    if reproducible:
        return np.random.default_rng(seed)

    # No seed means a fresh stochastic sequence on each run.
    return np.random.default_rng()


def calculate_trade_durations(
    trade_df: pd.DataFrame,
    market_df: pd.DataFrame,
    input_path: Path,
) -> np.ndarray:
    """Convert realized trades into holding periods measured in open-to-open bars."""
    date_to_index = pd.Series(market_df.index.to_numpy(), index=market_df["Date"])
    entry_indices = trade_df["entry_date"].map(date_to_index)
    exit_indices = trade_df["exit_date"].map(date_to_index)

    if entry_indices.isna().any() or exit_indices.isna().any():
        raise ValueError(
            f"{input_path} contains trade dates that do not align with the market data."
        )

    durations = exit_indices.to_numpy(dtype=int) - entry_indices.to_numpy(dtype=int)
    if np.any(durations <= 0):
        raise ValueError(
            f"{input_path} contains at least one non-positive holding period, "
            "which is inconsistent with next-open execution."
        )

    return durations


def draw_gap_allocation(
    slack_bars: int,
    slot_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly distribute slack bars across schedule gaps."""
    if slack_bars < 0:
        raise ValueError("Slack bars must be non-negative.")
    if slot_count <= 0:
        raise ValueError("Slot count must be positive.")

    if slack_bars == 0:
        return np.zeros(slot_count, dtype=int)

    probabilities = np.full(slot_count, 1.0 / slot_count, dtype=float)
    return rng.multinomial(slack_bars, probabilities)


def build_random_trade_schedule(
    durations: np.ndarray,
    max_open_index: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a non-overlapping random trade schedule with the same duration profile."""
    if len(durations) == 0:
        raise ValueError("Cannot build a schedule without at least one duration.")

    shuffled_durations = rng.permutation(durations.astype(int, copy=False))
    mandatory_internal_gaps = len(shuffled_durations) - 1
    slack_bars = int(max_open_index - shuffled_durations.sum() - mandatory_internal_gaps)

    if slack_bars < 0:
        raise ValueError(
            "Trade durations cannot fit inside the available price history "
            "without overlapping."
        )

    extra_gaps = draw_gap_allocation(
        slack_bars=slack_bars,
        slot_count=len(shuffled_durations) + 1,
        rng=rng,
    )

    pre_gap = int(extra_gaps[0])
    internal_gaps = extra_gaps[1:-1].astype(int) + 1

    entry_indices = np.empty(len(shuffled_durations), dtype=int)
    exit_indices = np.empty(len(shuffled_durations), dtype=int)

    current_entry = pre_gap
    for index, duration in enumerate(shuffled_durations):
        current_exit = current_entry + int(duration)
        entry_indices[index] = current_entry
        exit_indices[index] = current_exit

        if index < len(shuffled_durations) - 1:
            current_entry = current_exit + int(internal_gaps[index])

    if exit_indices[-1] > max_open_index:
        raise ValueError("Random schedule construction exceeded the available history.")

    return entry_indices, exit_indices


def calculate_interval_returns_from_open_prices(
    open_prices: np.ndarray,
    entry_indices: np.ndarray,
    exit_indices: np.ndarray,
) -> np.ndarray:
    """Calculate simple returns for a batch of open-to-open trade intervals."""
    entry_prices = open_prices[entry_indices]
    exit_prices = open_prices[exit_indices]
    return (exit_prices / entry_prices) - 1.0


def simulate_random_timing_cumulative_returns(
    open_prices: np.ndarray,
    durations: np.ndarray,
    position_value_fractions: np.ndarray,
    simulation_count: int,
    rng: np.random.Generator,
) -> pd.Series:
    """Simulate random-timing strategies that preserve trade count and durations."""
    if len(durations) == 0:
        raise ValueError("Cannot run simulations without at least one realized trade.")
    if len(position_value_fractions) != len(durations):
        raise ValueError("Each duration must have a matching position_value_fraction.")

    durations = durations.astype(np.int32, copy=False)
    position_value_fractions = position_value_fractions.astype(float, copy=False)
    max_open_index = len(open_prices) - 1
    number_of_trades = len(durations)
    mandatory_internal_gaps = number_of_trades - 1
    slack_bars = int(max_open_index - durations.sum() - mandatory_internal_gaps)

    if slack_bars < 0:
        raise ValueError(
            "Trade durations cannot fit inside the available price history "
            "without overlapping."
        )

    gap_probabilities = np.full(number_of_trades + 1, 1.0 / (number_of_trades + 1))
    simulated_cumulative_returns = np.empty(simulation_count, dtype=float)

    for batch_start in range(0, simulation_count, SIMULATION_BATCH_SIZE):
        batch_size = min(SIMULATION_BATCH_SIZE, simulation_count - batch_start)

        # Build one random permutation of the realized duration profile per simulation.
        permutation_keys = rng.random((batch_size, number_of_trades))
        permutation_indices = np.argsort(permutation_keys, axis=1)
        shuffled_durations = durations[permutation_indices]
        shuffled_position_value_fractions = position_value_fractions[permutation_indices]

        if slack_bars == 0:
            extra_gaps = np.zeros((batch_size, number_of_trades + 1), dtype=np.int32)
        else:
            extra_gaps = rng.multinomial(
                slack_bars,
                gap_probabilities,
                size=batch_size,
            ).astype(np.int32, copy=False)

        leading_gap = extra_gaps[:, [0]]
        if number_of_trades == 1:
            entry_indices = leading_gap.astype(np.int64, copy=False)
        else:
            internal_gaps = extra_gaps[:, 1:-1] + 1
            step_sizes = shuffled_durations[:, :-1] + internal_gaps
            cumulative_steps = np.cumsum(step_sizes, axis=1, dtype=np.int64)
            entry_indices = np.concatenate(
                [
                    leading_gap.astype(np.int64, copy=False),
                    leading_gap.astype(np.int64, copy=False) + cumulative_steps,
                ],
                axis=1,
            )

        exit_indices = entry_indices + shuffled_durations.astype(np.int64, copy=False)
        if np.any(exit_indices[:, -1] > max_open_index):
            raise ValueError(
                "Random schedule construction exceeded the available history."
            )

        entry_prices = open_prices[entry_indices]
        exit_prices = open_prices[exit_indices]
        simulated_position_returns = (exit_prices / entry_prices) - 1.0 - TRANSACTION_COST
        simulated_adjusted_returns = (
            shuffled_position_value_fractions * simulated_position_returns
        )

        if np.any(simulated_adjusted_returns <= -1):
            raise ValueError(
                "A simulated trade produced an adjusted return below -100%, "
                "which cannot be converted to log-return space."
            )

        simulated_log_returns = np.log1p(simulated_adjusted_returns)
        simulated_cumulative_returns[batch_start : batch_start + batch_size] = np.expm1(
            simulated_log_returns.sum(axis=1)
        )

    return pd.Series(
        simulated_cumulative_returns,
        name="simulated_cumulative_return",
    )


def calculate_p_value(simulated_returns: np.ndarray, actual_cumulative_return: float) -> float:
    """Calculate the one-sided p-value against the simulated baseline."""
    return float((simulated_returns >= actual_cumulative_return).mean())


def interpret_p_value(p_value: float) -> str:
    """Translate a p-value into a simple evidence label."""
    if p_value < 0.05:
        return "strong evidence"
    if p_value < 0.10:
        return "weak evidence"
    return "no evidence"


def save_simulation_results(
    output_path: Path,
    simulated_returns: pd.Series,
) -> None:
    """Save the full simulation results for one strategy."""
    results_df = pd.DataFrame(
        {
            "simulation_id": range(1, len(simulated_returns) + 1),
            "simulated_cumulative_return": simulated_returns.to_numpy(dtype=float),
        }
    )
    results_df.to_csv(output_path, index=False)


def build_agent_summary(
    agent_name: str,
    actual_cumulative_return: float,
    simulated_returns: pd.Series,
    number_of_trades: int,
) -> dict[str, float | int | str | bool]:
    """Build one summary row for the Monte Carlo output table."""
    simulated_array = simulated_returns.to_numpy(dtype=float)
    p_value = calculate_p_value(simulated_array, actual_cumulative_return)

    return {
        "agent": agent_name,
        "actual_cumulative_return": actual_cumulative_return,
        "median_simulated_return": float(np.median(simulated_array)),
        "mean_simulated_return": float(np.mean(simulated_array)),
        "std_simulated_return": float(np.std(simulated_array, ddof=0)),
        "actual_percentile": float((simulated_array <= actual_cumulative_return).mean() * 100),
        "p_value": p_value,
        "p_value_prominence": calculate_p_value_prominence(p_value),
        "p_value_interpretation": interpret_p_value(p_value),
        "lower_5pct": float(np.percentile(simulated_array, 5)),
        "upper_95pct": float(np.percentile(simulated_array, 95)),
        "number_of_trades": number_of_trades,
        "transaction_cost": TRANSACTION_COST,
        "simulation_count": NUMBER_OF_SIMULATIONS,
        "reproducible": REPRODUCIBLE,
        "seed_used": SEED if REPRODUCIBLE else "",
        "null_model": NULL_MODEL_NAME,
    }


def build_no_trade_summary(agent_name: str) -> dict[str, float | int | str | bool]:
    """Build a placeholder summary row for a strategy that produced no trades."""
    return {
        "agent": agent_name,
        "actual_cumulative_return": 0.0,
        "median_simulated_return": 0.0,
        "mean_simulated_return": 0.0,
        "std_simulated_return": 0.0,
        "actual_percentile": 100.0,
        "p_value": 1.0,
        "p_value_prominence": calculate_p_value_prominence(1.0),
        "p_value_interpretation": "no trades",
        "lower_5pct": 0.0,
        "upper_95pct": 0.0,
        "number_of_trades": 0,
        "transaction_cost": TRANSACTION_COST,
        "simulation_count": NUMBER_OF_SIMULATIONS,
        "reproducible": REPRODUCIBLE,
        "seed_used": SEED if REPRODUCIBLE else "",
        "null_model": NULL_MODEL_NAME,
    }


def print_agent_summary(summary_row: dict[str, float | int | str | bool]) -> None:
    """Print a clean terminal summary for one strategy."""
    print(f"\nAgent: {summary_row['agent']}")
    if int(summary_row["number_of_trades"]) == 0:
        print("No completed trades were generated for this strategy.")
        print("Stored a degenerate zero-return baseline for pipeline compatibility.")
        print(f"Simulation count: {summary_row['simulation_count']}")
        print(f"Null model: {summary_row['null_model']}")
        return

    print(f"Actual cumulative return: {summary_row['actual_cumulative_return']:.6f}")
    print(f"Median simulated return: {summary_row['median_simulated_return']:.6f}")
    print(f"Mean simulated return: {summary_row['mean_simulated_return']:.6f}")
    print(f"Std simulated return: {summary_row['std_simulated_return']:.6f}")
    print(f"Actual percentile: {summary_row['actual_percentile']:.2f}")
    print(f"p-value: {summary_row['p_value']:.6f}")
    print(f"p-value prominence: {summary_row['p_value_prominence']:.6f}")
    print(f"Evidence label: {summary_row['p_value_interpretation']}")
    print(f"5th percentile: {summary_row['lower_5pct']:.6f}")
    print(f"95th percentile: {summary_row['upper_95pct']:.6f}")
    print(f"Number of trades: {summary_row['number_of_trades']}")
    print(f"Transaction cost: {summary_row['transaction_cost']:.6f}")
    print(f"Simulation count: {summary_row['simulation_count']}")
    print(f"Reproducible mode: {summary_row['reproducible']}")
    print(f"Seed used: {summary_row['seed_used']}")
    print(f"Null model: {summary_row['null_model']}")


def main() -> None:
    """Run the Monte Carlo engine for all active strategies."""
    project_root = Path(__file__).resolve().parents[1]
    data_clean_dir = resolve_data_clean_dir(project_root)
    market_path = data_clean_dir / f"{ticker}_regimes.csv"
    market_df = load_market_data(market_path)
    open_prices = market_df["Open"].to_numpy(dtype=float)
    seed_sequence = np.random.SeedSequence(SEED)
    child_sequences = seed_sequence.spawn(len(AGENT_ORDER))

    summary_rows = []

    for agent_name, child_sequence in zip(AGENT_ORDER, child_sequences):
        input_path = data_clean_dir / f"{ticker}_{agent_name}_trades.csv"
        results_output_path = data_clean_dir / f"{ticker}_{agent_name}_monte_carlo_results.csv"
        rng = build_random_generator(
            reproducible=REPRODUCIBLE,
            seed=int(child_sequence.generate_state(1, dtype=np.uint64)[0]),
        )

        trade_df = load_trade_data(input_path, allow_empty=True)
        if trade_df.empty:
            simulated_returns = build_no_trade_simulated_returns(NUMBER_OF_SIMULATIONS)
            save_simulation_results(results_output_path, simulated_returns)
            summary_row = build_no_trade_summary(agent_name)
            summary_rows.append(summary_row)
            print_agent_summary(summary_row)
            continue

        raw_returns = trade_df["return"].to_numpy(dtype=float)
        adjusted_returns = adjust_trade_returns(
            raw_returns=raw_returns,
            transaction_cost=TRANSACTION_COST,
            input_path=input_path,
        )
        log_returns = convert_to_log_returns(adjusted_returns)
        durations = calculate_trade_durations(
            trade_df=trade_df,
            market_df=market_df,
            input_path=input_path,
        )
        position_value_fractions = extract_position_value_fractions(trade_df)

        actual_cumulative_return = calculate_cumulative_return_from_log_returns(log_returns)
        simulated_returns = simulate_random_timing_cumulative_returns(
            open_prices=open_prices,
            durations=durations,
            position_value_fractions=position_value_fractions,
            simulation_count=NUMBER_OF_SIMULATIONS,
            rng=rng,
        )

        save_simulation_results(results_output_path, simulated_returns)

        summary_row = build_agent_summary(
            agent_name=agent_name,
            actual_cumulative_return=actual_cumulative_return,
            simulated_returns=simulated_returns,
            number_of_trades=len(trade_df),
        )
        summary_rows.append(summary_row)
        print_agent_summary(summary_row)

    summary_df = pd.DataFrame(
        summary_rows,
        columns=[
            "agent",
            "actual_cumulative_return",
            "median_simulated_return",
            "mean_simulated_return",
            "std_simulated_return",
            "actual_percentile",
            "p_value",
            "p_value_prominence",
            "p_value_interpretation",
            "lower_5pct",
            "upper_95pct",
            "number_of_trades",
            "transaction_cost",
            "simulation_count",
            "reproducible",
            "seed_used",
            "null_model",
        ],
    )

    summary_output_path = data_clean_dir / f"{ticker}_monte_carlo_summary.csv"
    summary_df.to_csv(summary_output_path, index=False)

    print("\nMonte Carlo summary table:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
