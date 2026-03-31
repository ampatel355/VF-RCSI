"""Calculate simple performance metrics for the random trading agent."""

import os
from pathlib import Path

import pandas as pd

try:
    from research_metrics import calculate_trade_level_return_ratio
    from random_agent import main as create_trades
except ModuleNotFoundError:
    from Code.research_metrics import calculate_trade_level_return_ratio
    from Code.random_agent import main as create_trades


# Read the active ticker from the environment, or fall back to SPY.
ticker = os.environ.get("TICKER", "SPY")


def resolve_data_clean_dir(project_root: Path) -> Path:
    """Return the project's clean-data folder, supporting either naming style."""
    lowercase_dir = project_root / "data_clean"
    uppercase_dir = project_root / "Data_Clean"

    if lowercase_dir.exists():
        return lowercase_dir
    if uppercase_dir.exists():
        return uppercase_dir

    lowercase_dir.mkdir(parents=True, exist_ok=True)
    return lowercase_dir


def load_trade_data(input_path: Path) -> pd.DataFrame:
    """Load trade data, creating it first if the CSV does not exist yet."""
    if not input_path.exists():
        create_trades()

    df = pd.read_csv(input_path)

    # Convert the trade dates into datetime values so pandas handles them correctly.
    for column in ["entry_date", "exit_date"]:
        df[column] = pd.to_datetime(df[column], errors="coerce")

    # Make sure the return column is numeric before calculating metrics.
    df["return"] = pd.to_numeric(df["return"], errors="coerce")

    return df.dropna(subset=["return"]).reset_index(drop=True)


def safe_mean(series: pd.Series) -> float:
    """Return the mean of a series, or 0.0 if the series is empty."""
    if series.empty:
        return 0.0
    return float(series.mean())


def main() -> None:
    # Find the project root so the script works from any starting directory.
    project_root = Path(__file__).resolve().parents[1]
    data_clean_dir = resolve_data_clean_dir(project_root)

    # Build the input and output file paths using the shared ticker setting.
    input_path = data_clean_dir / f"{ticker}_random_trades.csv"
    output_path = data_clean_dir / f"{ticker}_random_metrics.csv"

    # Load the trade data created by the random agent.
    trades_df = load_trade_data(input_path)

    # Separate winning and losing trades so we can calculate win/loss averages.
    winning_returns = trades_df.loc[trades_df["return"] > 0, "return"]
    losing_returns = trades_df.loc[trades_df["return"] < 0, "return"]

    # Count the number of completed trades.
    total_trades = int(len(trades_df))

    # These are the main summary statistics requested by the user.
    average_return = safe_mean(trades_df["return"])
    median_return = float(trades_df["return"].median()) if total_trades > 0 else 0.0
    win_rate = float((trades_df["return"] > 0).mean()) if total_trades > 0 else 0.0
    std_return = float(trades_df["return"].std(ddof=0)) if total_trades > 1 else 0.0

    trade_level_return_ratio = calculate_trade_level_return_ratio(
        trades_df["return"].to_numpy(dtype=float)
    )

    # average_win uses only winning trades.
    average_win = safe_mean(winning_returns)

    # average_loss uses losing trades, and the absolute value turns it positive.
    average_loss = abs(safe_mean(losing_returns))

    # expected_value combines the chance of winning with the average win/loss size.
    expected_value = (win_rate * average_win) - ((1 - win_rate) * average_loss)

    # Store all metrics in one row so they can be saved to a CSV file.
    metrics_df = pd.DataFrame(
        [
            {
                "total_trades": total_trades,
                "average_return": average_return,
                "median_return": median_return,
                "win_rate": win_rate,
                "std_return": std_return,
                "trade_level_return_ratio": trade_level_return_ratio,
                "average_win": average_win,
                "average_loss": average_loss,
                "expected_value": expected_value,
            }
        ]
    )

    # Save the metrics as a one-row CSV file.
    metrics_df.to_csv(output_path, index=False)

    # Print the metrics clearly so they are easy to read.
    print(f"total_trades: {total_trades}")
    print(f"average_return: {average_return:.6f}")
    print(f"median_return: {median_return:.6f}")
    print(f"win_rate: {win_rate:.6f}")
    print(f"std_return: {std_return:.6f}")
    print(f"trade_level_return_ratio: {trade_level_return_ratio:.6f}")
    print(f"average_win: {average_win:.6f}")
    print(f"average_loss: {average_loss:.6f}")
    print(f"expected_value: {expected_value:.6f}")


if __name__ == "__main__":
    main()
