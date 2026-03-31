"""Download and clean daily historical data for one ticker."""

import os
from pathlib import Path

import yfinance as yf

# Read the active ticker from the environment, or fall back to SPY.
configured_ticker = os.environ.get("TICKER", "SPY")


def resolve_data_raw_dir(project_root: Path) -> Path:
    """Return the project's raw-data folder, supporting either naming style."""
    lowercase_dir = project_root / "data_raw"
    uppercase_dir = project_root / "Data_Raw"

    if lowercase_dir.exists():
        return lowercase_dir
    if uppercase_dir.exists():
        return uppercase_dir

    lowercase_dir.mkdir(parents=True, exist_ok=True)
    return lowercase_dir


def main(ticker: str | None = None) -> None:
    # Use the shared project ticker unless a specific ticker was passed in.
    ticker = ticker or configured_ticker

    # Find the project root so the script works from any current working directory.
    project_root = Path(__file__).resolve().parents[1]

    # Create the output directory if it does not already exist.
    output_dir = resolve_data_raw_dir(project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download the full daily price history for the chosen ticker.
    df = yf.download(
        ticker,
        period="max",
        interval="1d",
        auto_adjust=True,
        actions=False,
        progress=False,
    )
    if df.empty:
        raise RuntimeError(f"No {ticker} price data was returned by yfinance.")

    # Move the date index into a normal column so it can be saved to CSV.
    df = df.reset_index()

    # yfinance may return a two-level column index even for one ticker, so flatten it.
    df.columns = [column[0] if isinstance(column, tuple) else column for column in df.columns]

    # Keep only the requested columns in the expected order.
    columns_to_keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[columns_to_keep]

    # Remove rows with missing values and sort from oldest to newest date.
    df = df.dropna().sort_values("Date", ascending=True)

    # Save the cleaned data to the requested CSV file.
    output_path = output_dir / f"{ticker}.csv"
    df.to_csv(output_path, index=False)

    # Print the first five cleaned rows so the result is easy to verify.
    print(df.head(5))


if __name__ == "__main__":
    main()
