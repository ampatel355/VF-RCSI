"""Compare the active strategies using the shared research metrics."""

import os

from plot_config import data_clean_dir, load_csv_checked
import pandas as pd

try:
    from buy_and_hold import main as create_buy_and_hold
    from monte_carlo import main as create_monte_carlo
    from monte_carlo import (
        TRANSACTION_COST,
        load_trade_data,
        load_market_data,
    )
    from multiple_testing import apply_cross_ticker_fdr
    from rcsi import main as create_rcsi
    from research_metrics import (
        build_daily_strategy_curve,
        calculate_p_value_prominence,
        calculate_trade_level_return_ratio,
        summarize_daily_curve,
    )
    from strategy_config import AGENT_ORDER, BENCHMARK_NAME, COMPARISON_ORDER
    from strategy_verdicts import load_strategy_verdicts
except ModuleNotFoundError:
    from Code.buy_and_hold import main as create_buy_and_hold
    from Code.monte_carlo import main as create_monte_carlo
    from Code.monte_carlo import (
        TRANSACTION_COST,
        load_trade_data,
        load_market_data,
    )
    from Code.multiple_testing import apply_cross_ticker_fdr
    from Code.rcsi import main as create_rcsi
    from Code.research_metrics import (
        build_daily_strategy_curve,
        calculate_p_value_prominence,
        calculate_trade_level_return_ratio,
        summarize_daily_curve,
    )
    from Code.strategy_config import AGENT_ORDER, BENCHMARK_NAME, COMPARISON_ORDER
    from Code.strategy_verdicts import load_strategy_verdicts


# Read the active ticker from the environment, or fall back to SPY.
ticker = os.environ.get("TICKER", "SPY")
STRATEGY_ORDER = COMPARISON_ORDER


def load_market_curve_data() -> pd.DataFrame:
    """Load the market data used for daily equity reconstruction."""
    market_path = data_clean_dir() / f"{ticker}_regimes.csv"
    open_df = load_market_data(market_path)[["Date", "Open"]].copy()
    close_df = load_csv_checked(market_path, required_columns=["Date", "Close"])[["Date", "Close"]].copy()
    close_df["Date"] = pd.to_datetime(close_df["Date"], errors="coerce")
    close_df["Close"] = pd.to_numeric(close_df["Close"], errors="coerce")
    close_df = close_df.dropna(subset=["Date", "Close"]).reset_index(drop=True)
    return open_df.merge(close_df, on="Date", how="inner")


def load_strategy_summary_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the Monte Carlo summary and RCSI tables used in the comparison."""
    monte_carlo_summary_path = data_clean_dir() / f"{ticker}_monte_carlo_summary.csv"
    rcsi_path = data_clean_dir() / f"{ticker}_rcsi.csv"

    if not monte_carlo_summary_path.exists():
        create_monte_carlo()
    if not rcsi_path.exists():
        create_rcsi()

    required_monte_carlo_columns = [
        "agent",
        "actual_cumulative_return",
        "actual_percentile",
        "p_value",
        "p_value_prominence",
    ]
    required_rcsi_columns = ["agent", "RCSI", "RCSI_z"]

    try:
        monte_carlo_summary_df = load_csv_checked(
            monte_carlo_summary_path,
            required_columns=required_monte_carlo_columns,
        )
    except (FileNotFoundError, KeyError, ValueError):
        create_monte_carlo()
        monte_carlo_summary_df = load_csv_checked(
            monte_carlo_summary_path,
            required_columns=required_monte_carlo_columns,
        )

    try:
        rcsi_df = load_csv_checked(
            rcsi_path,
            required_columns=required_rcsi_columns,
        )
    except (FileNotFoundError, KeyError, ValueError):
        create_rcsi()
        rcsi_df = load_csv_checked(
            rcsi_path,
            required_columns=required_rcsi_columns,
        )

    monte_carlo_summary_df["agent"] = monte_carlo_summary_df["agent"].astype(str).str.strip()
    rcsi_df["agent"] = rcsi_df["agent"].astype(str).str.strip()

    expected_agents = set(AGENT_ORDER)
    if set(monte_carlo_summary_df["agent"]) != expected_agents:
        create_monte_carlo()
        monte_carlo_summary_df = load_csv_checked(
            monte_carlo_summary_path,
            required_columns=required_monte_carlo_columns,
        )
        monte_carlo_summary_df["agent"] = monte_carlo_summary_df["agent"].astype(str).str.strip()

    if set(rcsi_df["agent"]) != expected_agents:
        create_rcsi()
        rcsi_df = load_csv_checked(
            rcsi_path,
            required_columns=required_rcsi_columns,
        )
        rcsi_df["agent"] = rcsi_df["agent"].astype(str).str.strip()

    return monte_carlo_summary_df, rcsi_df


def build_strategy_row(
    agent_name: str,
    monte_carlo_summary_df: pd.DataFrame,
    rcsi_df: pd.DataFrame,
    verdict_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> dict[str, float | int | str]:
    """Create one comparison row for one strategy."""
    trade_path = data_clean_dir() / f"{ticker}_{agent_name}_trades.csv"
    trade_df = load_trade_data(trade_path, allow_empty=True)
    strategy_curve_df = build_daily_strategy_curve(trade_df, market_df)
    curve_summary = summarize_daily_curve(strategy_curve_df)
    trade_level_return_ratio = calculate_trade_level_return_ratio(
        trade_df["return"].to_numpy(dtype=float)
    )
    has_completed_trades = len(trade_df) > 0

    monte_carlo_row = monte_carlo_summary_df.loc[monte_carlo_summary_df["agent"] == agent_name]
    rcsi_row = rcsi_df.loc[rcsi_df["agent"] == agent_name]
    verdict_row = verdict_df.loc[verdict_df["agent"] == agent_name]

    if monte_carlo_row.empty or rcsi_row.empty or verdict_row.empty:
        raise ValueError(f"Missing summary rows for strategy '{agent_name}'.")

    monte_carlo_row = monte_carlo_row.iloc[0]
    rcsi_row = rcsi_row.iloc[0]
    verdict_row = verdict_row.iloc[0]

    return {
        "agent": agent_name,
        "skill_luck_verdict": str(verdict_row["verdict_label"]),
        "evidence_label": str(verdict_row["evidence_label"]),
        "confidence_label": str(verdict_row["confidence_label"]),
        "final_classification": str(verdict_row["final_classification"]),
        "stability_classification": str(verdict_row["stability_classification"]),
        "verdict_reason": str(verdict_row["verdict_reason"]),
        "cumulative_return": float(curve_summary["cumulative_return"]),
        "annualized_sharpe": float(curve_summary["annualized_sharpe"]),
        "trade_level_return_ratio": float(trade_level_return_ratio),
        "max_drawdown": float(curve_summary["max_drawdown"]),
        "RCSI": float(rcsi_row["RCSI"]) if has_completed_trades else pd.NA,
        "RCSI_z": float(rcsi_row["RCSI_z"]) if has_completed_trades else pd.NA,
        "p_value": float(monte_carlo_row["p_value"]) if has_completed_trades else pd.NA,
        "p_value_prominence": (
            float(monte_carlo_row["p_value_prominence"]) if has_completed_trades else pd.NA
        ),
        "actual_percentile": (
            float(monte_carlo_row["actual_percentile"]) if has_completed_trades else pd.NA
        ),
        "number_of_trades": int(len(trade_df)),
        "number_of_periods": int(curve_summary["number_of_periods"]),
        "transaction_cost": TRANSACTION_COST,
    }


def build_buy_and_hold_row(market_df: pd.DataFrame) -> dict[str, float | int | str]:
    """Create one comparison row for the passive buy-and-hold benchmark."""
    metrics_path = data_clean_dir() / f"{ticker}_buy_hold_metrics.csv"
    if not metrics_path.exists():
        create_buy_and_hold()

    metrics_df = load_csv_checked(
        metrics_path,
        required_columns=[
            "buy_hold_return",
            "annualized_sharpe",
            "max_drawdown",
            "number_of_periods",
            "transaction_cost",
        ],
    )
    benchmark_row = metrics_df.iloc[0]

    return {
        "agent": BENCHMARK_NAME,
        "skill_luck_verdict": "Benchmark",
        "evidence_label": "Benchmark Reference",
        "confidence_label": "Not Applicable",
        "final_classification": "benchmark",
        "stability_classification": "not_applicable",
        "verdict_reason": "Passive benchmark included for strategy context; not evaluated against the Monte Carlo skill-vs-luck null.",
        "cumulative_return": float(benchmark_row["buy_hold_return"]),
        "annualized_sharpe": float(benchmark_row["annualized_sharpe"]),
        "trade_level_return_ratio": pd.NA,
        "max_drawdown": float(benchmark_row["max_drawdown"]),
        "RCSI": pd.NA,
        "RCSI_z": pd.NA,
        "p_value": pd.NA,
        "p_value_prominence": pd.NA,
        "actual_percentile": pd.NA,
        "number_of_trades": pd.NA,
        "number_of_periods": int(benchmark_row["number_of_periods"]),
        "transaction_cost": float(benchmark_row["transaction_cost"]),
    }


def main() -> None:
    """Build the full comparison table for the active strategies."""
    monte_carlo_summary_df, rcsi_df = load_strategy_summary_tables()
    verdict_df = load_strategy_verdicts(ticker)
    market_df = load_market_curve_data()
    comparison_rows = []

    for agent_name in AGENT_ORDER:
        comparison_rows.append(
            build_strategy_row(
                agent_name=agent_name,
                monte_carlo_summary_df=monte_carlo_summary_df,
                rcsi_df=rcsi_df,
                verdict_df=verdict_df,
                market_df=market_df,
            )
        )

    comparison_rows.append(build_buy_and_hold_row(market_df))

    comparison_df = pd.DataFrame(
        comparison_rows,
        columns=[
            "agent",
            "skill_luck_verdict",
            "evidence_label",
            "confidence_label",
            "final_classification",
            "stability_classification",
            "verdict_reason",
            "cumulative_return",
            "annualized_sharpe",
            "trade_level_return_ratio",
            "max_drawdown",
            "RCSI",
            "RCSI_z",
            "p_value",
            "p_value_prominence",
            "actual_percentile",
            "number_of_trades",
            "number_of_periods",
            "transaction_cost",
        ],
    )

    comparison_df["agent"] = pd.Categorical(
        comparison_df["agent"],
        categories=STRATEGY_ORDER,
        ordered=True,
    )
    comparison_df = comparison_df.sort_values("agent").reset_index(drop=True)
    comparison_df["agent"] = comparison_df["agent"].astype(str)

    full_output_path = data_clean_dir() / f"{ticker}_full_comparison.csv"
    legacy_output_path = data_clean_dir() / f"{ticker}_agent_comparison.csv"

    comparison_df.to_csv(full_output_path, index=False)
    comparison_df.to_csv(legacy_output_path, index=False)
    apply_cross_ticker_fdr(data_clean_dir())
    comparison_df = pd.read_csv(full_output_path)

    display_columns = [
        "agent",
        "skill_luck_verdict",
        "evidence_label",
        "confidence_label",
        "final_classification",
        "cumulative_return",
        "annualized_sharpe",
        "RCSI_z",
        "p_value",
        "fdr_q_value",
        "actual_percentile",
        "number_of_trades",
    ]
    print(comparison_df[display_columns].to_string(index=False))
    print("\nSkill vs luck verdicts:")
    for _, row in comparison_df.iterrows():
        print(
            f"- {row['agent']}: {row['evidence_label']} | "
            f"high-level={row['skill_luck_verdict']} | "
            f"confidence={row['confidence_label']} | "
            f"{row['verdict_reason']}"
        )


if __name__ == "__main__":
    main()
