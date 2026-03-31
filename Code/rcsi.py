"""Calculate RCSI directly from the current Monte Carlo summary file."""

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


def load_csv_checked(input_path: Path, required_columns: list[str]) -> pd.DataFrame:
    """Load a CSV file and confirm that it contains the expected columns."""
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


def load_summary_data(input_path: Path) -> pd.DataFrame:
    """Load and validate the Monte Carlo summary table."""
    df = load_csv_checked(
        input_path,
        required_columns=[
            "agent",
            "actual_cumulative_return",
            "median_simulated_return",
            "mean_simulated_return",
            "std_simulated_return",
            "actual_percentile",
        ],
    )

    numeric_columns = [
        "actual_cumulative_return",
        "median_simulated_return",
        "mean_simulated_return",
        "std_simulated_return",
        "actual_percentile",
    ]

    df["agent"] = df["agent"].astype(str).str.strip()
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=["agent", *numeric_columns],
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No usable Monte Carlo summary rows were found in: {input_path}")

    duplicate_agents = df["agent"][df["agent"].duplicated()].unique().tolist()
    if duplicate_agents:
        raise ValueError(
            "Each strategy should appear only once in the Monte Carlo summary. "
            f"Duplicate rows found for: {duplicate_agents}"
        )

    available_agents = [agent for agent in AGENT_ORDER if agent in df["agent"].tolist()]
    if not available_agents:
        raise ValueError(f"No expected strategy names were found in: {input_path}")

    return df.set_index("agent").reindex(available_agents).reset_index()


def main() -> None:
    """Calculate and save the RCSI table for the active ticker."""
    project_root = Path(__file__).resolve().parents[1]
    data_clean_dir = resolve_data_clean_dir(project_root)
    input_path = data_clean_dir / f"{ticker}_monte_carlo_summary.csv"
    output_path = data_clean_dir / f"{ticker}_rcsi.csv"

    df = load_summary_data(input_path)

    # RCSI compares the actual observed cumulative return with the median
    # return from the Monte Carlo distribution.
    df["RCSI"] = df["actual_cumulative_return"] - df["median_simulated_return"]

    # RCSI_z standardizes the gap using the Monte Carlo mean and standard deviation.
    df["RCSI_z"] = np.where(
        df["std_simulated_return"] > 0,
        (df["actual_cumulative_return"] - df["mean_simulated_return"])
        / df["std_simulated_return"],
        np.where(
            np.isclose(df["actual_cumulative_return"], df["mean_simulated_return"]),
            0.0,
            np.nan,
        ),
    )

    output_df = df[
        [
            "agent",
            "actual_cumulative_return",
            "median_simulated_return",
            "mean_simulated_return",
            "std_simulated_return",
            "actual_percentile",
            "RCSI",
            "RCSI_z",
        ]
    ].copy()

    output_df = output_df.sort_values("RCSI", ascending=False).reset_index(drop=True)
    output_df.to_csv(output_path, index=False)

    print(output_df.to_string(index=False))


if __name__ == "__main__":
    main()
