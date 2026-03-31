"""Calculate simple performance metrics for the breakout agent."""

import os
from pathlib import Path

import pandas as pd

try:
    from research_metrics import calculate_trade_level_return_ratio
    from breakout_agent import main as create_trades
except ModuleNotFoundError:
    from Code.research_metrics import calculate_trade_level_return_ratio
    from Code.breakout_agent import main as create_trades


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
    for column in ["entry_date", "exit_date"]:
        df[column] = pd.to_datetime(df[column], errors="coerce")
    df["return"] = pd.to_numeric(df["return"], errors="coerce")
    return df.dropna(subset=["return"]).reset_index(drop=True)


def safe_mean(series: pd.Series) -> float:
    """Return the mean of a series, or 0.0 if the series is empty."""
    if series.empty:
        return 0.0
    return float(series.mean())


def main() -> None:
    """Calculate and save the breakout strategy metrics."""
    project_root = Path(__file__).resolve().parents[1]
    data_clean_dir = resolve_data_clean_dir(project_root)
    input_path = data_clean_dir / f"{ticker}_breakout_trades.csv"
    output_path = data_clean_dir / f"{ticker}_breakout_metrics.csv"

    trades_df = load_trade_data(input_path)
    winning_returns = trades_df.loc[trades_df["return"] > 0, "return"]
    losing_returns = trades_df.loc[trades_df["return"] < 0, "return"]

    total_trades = int(len(trades_df))
    average_return = safe_mean(trades_df["return"])
    median_return = float(trades_df["return"].median()) if total_trades > 0 else 0.0
    win_rate = float((trades_df["return"] > 0).mean()) if total_trades > 0 else 0.0
    std_return = float(trades_df["return"].std(ddof=0)) if total_trades > 1 else 0.0
    trade_level_return_ratio = calculate_trade_level_return_ratio(
        trades_df["return"].to_numpy(dtype=float)
    )
    average_win = safe_mean(winning_returns)
    average_loss = abs(safe_mean(losing_returns))
    expected_value = (win_rate * average_win) - ((1 - win_rate) * average_loss)

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
    metrics_df.to_csv(output_path, index=False)

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
