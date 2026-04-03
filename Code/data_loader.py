"""Download and clean historical market data for one ticker."""

import os
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

try:
    from timeframe_config import (
        DEFAULT_YAHOO_PERIOD,
        RESEARCH_INTERVAL,
        RESEARCH_TIMEFRAME_LABEL,
        interval_looks_compatible,
        normalize_timestamp_series,
    )
except ModuleNotFoundError:
    from Code.timeframe_config import (
        DEFAULT_YAHOO_PERIOD,
        RESEARCH_INTERVAL,
        RESEARCH_TIMEFRAME_LABEL,
        interval_looks_compatible,
        normalize_timestamp_series,
    )

# Read the active ticker from the environment, or fall back to SPY.
configured_ticker = os.environ.get("TICKER", "SPY")
ALLOW_STALE_RAW_FALLBACK = os.environ.get("ALLOW_STALE_RAW_FALLBACK", "0") == "1"
RAW_DATA_MAX_STALENESS_HOURS = float(os.environ.get("RAW_DATA_MAX_STALENESS_HOURS", "48"))


def resolve_data_raw_dir(project_root: Path) -> Path:
    """Return the project's raw-data folder, preferring the uppercase path."""
    lowercase_dir = project_root / "data_raw"
    uppercase_dir = project_root / "Data_Raw"

    if uppercase_dir.exists():
        return uppercase_dir
    if lowercase_dir.exists():
        return lowercase_dir

    uppercase_dir.mkdir(parents=True, exist_ok=True)
    return uppercase_dir


def _fallback_file_is_fresh_enough(existing_df: pd.DataFrame) -> bool:
    """Return whether a saved raw file is fresh enough to reuse on download failure."""
    if "downloaded_at_utc" not in existing_df.columns:
        return False

    downloaded_at = pd.to_datetime(existing_df["downloaded_at_utc"], errors="coerce", utc=True)
    downloaded_at = downloaded_at.dropna()
    if downloaded_at.empty:
        return False

    file_age_hours = (
        datetime.now(timezone.utc) - downloaded_at.iloc[-1].to_pydatetime()
    ).total_seconds() / 3600.0
    return file_age_hours <= RAW_DATA_MAX_STALENESS_HOURS


def main(ticker: str | None = None) -> None:
    # Use the shared project ticker unless a specific ticker was passed in.
    ticker = ticker or configured_ticker

    # Find the project root so the script works from any current working directory.
    project_root = Path(__file__).resolve().parents[1]

    # Create the output directory if it does not already exist.
    output_dir = resolve_data_raw_dir(project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{ticker}.csv"

    # Download the longest Yahoo-valid history for the chosen interval.
    try:
        df = yf.download(
            ticker,
            period=DEFAULT_YAHOO_PERIOD,
            interval=RESEARCH_INTERVAL,
            auto_adjust=True,
            actions=False,
            progress=False,
        )
    except Exception:
        df = None

    if df is None or df.empty:
        if output_path.exists():
            existing_df = pd.read_csv(output_path)
            if "Date" in existing_df.columns and interval_looks_compatible(
                existing_df["Date"],
                RESEARCH_INTERVAL,
            ) and (ALLOW_STALE_RAW_FALLBACK or _fallback_file_is_fresh_enough(existing_df)):
                print(
                    f"Download failed for {ticker}. Reusing the existing raw file at {output_path}."
                )
                print(output_path)
                return
            if output_path.exists() and not ALLOW_STALE_RAW_FALLBACK:
                print(
                    f"Download failed for {ticker}, and the existing raw file at {output_path} "
                    "is too stale to reuse safely."
                )
            print(
                f"Download failed for {ticker}, and the existing raw file at {output_path} "
                f"does not look like valid {RESEARCH_INTERVAL} data."
            )
            raise RuntimeError(
                "The pipeline needs interval-compatible raw data before it can continue. "
                "Run again with network access or replace the stale file."
            )
        raise RuntimeError(f"No {ticker} price data was returned by yfinance.")

    # Move the date index into a normal column so it can be saved to CSV.
    df = df.reset_index()

    # yfinance may return a two-level column index even for one ticker, so flatten it.
    df.columns = [column[0] if isinstance(column, tuple) else column for column in df.columns]

    # Keep only the requested columns in the expected order.
    if "Datetime" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Datetime": "Date"})

    columns_to_keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[columns_to_keep]

    # Normalize timestamps so cross-asset alignment later in the research stack
    # does not depend on the source exchange timezone.
    df["Date"] = normalize_timestamp_series(df["Date"])

    # Remove rows with missing values, drop duplicates, and sort from oldest to
    # newest timestamp. We do not forward-fill missing bars because that would
    # invent prices and create fake trades.
    df = (
        df.dropna()
        .drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date", ascending=True)
        .reset_index(drop=True)
    )
    df["data_interval"] = RESEARCH_INTERVAL
    df["timeframe_label"] = RESEARCH_TIMEFRAME_LABEL
    df["data_source"] = "yfinance"
    df["source_period_requested"] = DEFAULT_YAHOO_PERIOD
    df["downloaded_at_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    # Save the cleaned data to the requested CSV file.
    df.to_csv(output_path, index=False)

    # Print the first five cleaned rows so the result is easy to verify.
    print(df.head(5))


if __name__ == "__main__":
    main()
