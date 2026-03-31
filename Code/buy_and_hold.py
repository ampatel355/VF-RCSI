"""Create a buy-and-hold benchmark from the current ticker's feature data."""

import os
from pathlib import Path

from plot_config import data_clean_dir, load_csv_checked
import pandas as pd

try:
    from features import main as create_features
    from research_metrics import (
        build_buy_and_hold_curve as build_daily_buy_and_hold_curve,
        summarize_daily_curve,
    )
except ModuleNotFoundError:
    from Code.features import main as create_features
    from Code.research_metrics import (
        build_buy_and_hold_curve as build_daily_buy_and_hold_curve,
        summarize_daily_curve,
    )


# Read the active ticker from the environment, or fall back to SPY.
ticker = os.environ.get("TICKER", "SPY")

# Keep benchmark friction simple and easy to change.
BUY_HOLD_TRANSACTION_COST = 0.0


def load_feature_data(input_path: Path) -> pd.DataFrame:
    """Load the ticker's feature table, creating it first if needed."""
    if not input_path.exists():
        create_features()

    df = load_csv_checked(
        input_path,
        required_columns=["Date", "Close"],
    )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No usable price rows were found in: {input_path}")

    return df


def build_buy_hold_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Build the benchmark curve from the first close to the last close."""
    return build_daily_buy_and_hold_curve(
        market_df=df[["Date", "Close"]],
        transaction_cost=BUY_HOLD_TRANSACTION_COST,
    )


def build_metrics(curve_df: pd.DataFrame) -> pd.DataFrame:
    """Build a one-row metrics table for the buy-and-hold benchmark."""
    curve_summary = summarize_daily_curve(curve_df)

    metrics_df = pd.DataFrame(
        [
            {
                "start_date": curve_df["Date"].iloc[0].strftime("%Y-%m-%d"),
                "end_date": curve_df["Date"].iloc[-1].strftime("%Y-%m-%d"),
                "first_close": float(curve_df["Close"].iloc[0]),
                "last_close": float(curve_df["Close"].iloc[-1]),
                "buy_hold_return": float(curve_summary["cumulative_return"]),
                "annualized_sharpe": float(curve_summary["annualized_sharpe"]),
                "max_drawdown": float(curve_summary["max_drawdown"]),
                "number_of_periods": int(curve_summary["number_of_periods"]),
                "transaction_cost": BUY_HOLD_TRANSACTION_COST,
            }
        ]
    )

    return metrics_df


def main() -> None:
    """Create the buy-and-hold benchmark files for the active ticker."""
    input_path = data_clean_dir() / f"{ticker}_features.csv"
    metrics_output_path = data_clean_dir() / f"{ticker}_buy_hold_metrics.csv"
    curve_output_path = data_clean_dir() / f"{ticker}_buy_hold_curve.csv"

    feature_df = load_feature_data(input_path)
    curve_df = build_buy_hold_curve(feature_df)
    metrics_df = build_metrics(curve_df)

    curve_df.to_csv(curve_output_path, index=False)
    metrics_df.to_csv(metrics_output_path, index=False)

    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
