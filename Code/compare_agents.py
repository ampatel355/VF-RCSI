"""Compare the active strategies using a small, research-focused summary table."""

import os

from plot_config import data_clean_dir, load_csv_checked
import pandas as pd

try:
    from artifact_provenance import artifact_run_id, write_dataframe_artifact
    from buy_and_hold import main as create_buy_and_hold
    from monte_carlo import main as create_monte_carlo
    from monte_carlo import (
        TRANSACTION_COST,
        load_trade_data,
        load_market_data,
    )
    from rcsi import main as create_rcsi
    from research_metrics import (
        build_daily_strategy_curve,
        calculate_p_value_prominence,
        calculate_trade_level_return_ratio,
        summarize_daily_curve,
    )
    from strategy_config import AGENT_ORDER, BENCHMARK_NAME, COMPARISON_ORDER
    from strategy_artifact_utils import ensure_trade_file_exists
    from strategy_curve_utils import load_saved_strategy_curve
    from strategy_verdicts import load_strategy_verdicts
    from timeframe_config import RESEARCH_TIMEFRAME_LABEL, normalize_timestamp_series
except ModuleNotFoundError:
    from Code.artifact_provenance import artifact_run_id, write_dataframe_artifact
    from Code.buy_and_hold import main as create_buy_and_hold
    from Code.monte_carlo import main as create_monte_carlo
    from Code.monte_carlo import (
        TRANSACTION_COST,
        load_trade_data,
        load_market_data,
    )
    from Code.rcsi import main as create_rcsi
    from Code.research_metrics import (
        build_daily_strategy_curve,
        calculate_p_value_prominence,
        calculate_trade_level_return_ratio,
        summarize_daily_curve,
    )
    from Code.strategy_config import AGENT_ORDER, BENCHMARK_NAME, COMPARISON_ORDER
    from Code.strategy_artifact_utils import ensure_trade_file_exists
    from Code.strategy_curve_utils import load_saved_strategy_curve
    from Code.strategy_verdicts import load_strategy_verdicts
    from Code.timeframe_config import RESEARCH_TIMEFRAME_LABEL, normalize_timestamp_series


# Read the active ticker from the environment, or fall back to SPY.
ticker = os.environ.get("TICKER", "SPY")
STRATEGY_ORDER = COMPARISON_ORDER
FAST_TEST_MODE = os.environ.get("FAST_TEST_MODE", "0") == "1"
MIN_RESEARCH_GRADE_SIMULATIONS = int(
    os.environ.get("MIN_RESEARCH_GRADE_SIMULATIONS", "5000")
)
REQUIRE_RESEARCH_GRADE_INPUTS = (
    os.environ.get("REQUIRE_RESEARCH_GRADE_INPUTS", "1") == "1"
    and not FAST_TEST_MODE
)


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    """Normalize bool-like CSV columns loaded from pandas or plain text."""
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map({"true": True, "false": False, "1": True, "0": False}).fillna(False)


def load_market_curve_data() -> pd.DataFrame:
    """Load the market data used for bar-based equity reconstruction."""
    market_path = data_clean_dir() / f"{ticker}_regimes.csv"
    open_df = load_market_data(market_path)[["Date", "Open"]].copy()
    close_df = load_csv_checked(market_path, required_columns=["Date", "Close"])[["Date", "Close"]].copy()
    close_df["Date"] = normalize_timestamp_series(close_df["Date"])
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

    monte_carlo_run_id = artifact_run_id(monte_carlo_summary_path)
    rcsi_run_id = artifact_run_id(rcsi_path)
    if monte_carlo_run_id and rcsi_run_id and monte_carlo_run_id != rcsi_run_id:
        create_monte_carlo()
        create_rcsi()

    required_monte_carlo_columns = [
        "agent",
        "actual_cumulative_return",
        "actual_percentile",
        "p_value",
        "p_value_prominence",
        "bh_adjusted_p_value",
        "simulation_count",
        "research_grade",
        "null_model",
    ]
    required_rcsi_columns = [
        "agent",
        "RCSI",
        "RCSI_z",
        "simulation_count",
        "research_grade",
    ]

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

    if REQUIRE_RESEARCH_GRADE_INPUTS:
        monte_carlo_simulations = pd.to_numeric(
            monte_carlo_summary_df["simulation_count"],
            errors="coerce",
        ).fillna(0)
        monte_carlo_research_grade = (
            _coerce_bool_series(monte_carlo_summary_df["research_grade"])
            & monte_carlo_simulations.ge(MIN_RESEARCH_GRADE_SIMULATIONS)
        )
        rcsi_simulations = pd.to_numeric(
            rcsi_df["simulation_count"],
            errors="coerce",
        ).fillna(0)
        rcsi_research_grade = (
            _coerce_bool_series(rcsi_df["research_grade"])
            & rcsi_simulations.ge(MIN_RESEARCH_GRADE_SIMULATIONS)
        )
        if not bool(monte_carlo_research_grade.all() and rcsi_research_grade.all()):
            create_monte_carlo()
            create_rcsi()
            monte_carlo_summary_df = load_csv_checked(
                monte_carlo_summary_path,
                required_columns=required_monte_carlo_columns,
            )
            rcsi_df = load_csv_checked(
                rcsi_path,
                required_columns=required_rcsi_columns,
            )
            monte_carlo_summary_df["agent"] = monte_carlo_summary_df["agent"].astype(str).str.strip()
            rcsi_df["agent"] = rcsi_df["agent"].astype(str).str.strip()

    return monte_carlo_summary_df, rcsi_df


def load_trade_activity_validation() -> pd.DataFrame:
    """Load the trade-activity validation table, creating it if needed."""
    input_path = data_clean_dir() / f"{ticker}_trade_activity_validation.csv"
    required_columns = [
        "agent",
        "signal_count",
        "trades_per_month",
        "average_holding_bars",
        "activity_status",
        "activity_warning",
    ]
    if not input_path.exists():
        try:
            from trade_activity_validation import main as create_trade_activity_validation
        except ModuleNotFoundError:
            from Code.trade_activity_validation import main as create_trade_activity_validation
        create_trade_activity_validation()

    activity_df = load_csv_checked(input_path, required_columns=required_columns)
    activity_df["agent"] = activity_df["agent"].astype(str).str.strip()
    if "average_holding_days" not in activity_df.columns:
        if "average_holding_hours" in activity_df.columns:
            activity_df["average_holding_days"] = (
                pd.to_numeric(activity_df["average_holding_hours"], errors="coerce") / 24.0
            )
        else:
            activity_df["average_holding_days"] = pd.NA
    if "average_holding_hours" not in activity_df.columns:
        if "average_holding_days" in activity_df.columns:
            activity_df["average_holding_hours"] = (
                pd.to_numeric(activity_df["average_holding_days"], errors="coerce") * 24.0
            )
        else:
            activity_df["average_holding_hours"] = pd.NA
    return activity_df


def build_strategy_row(
    agent_name: str,
    monte_carlo_summary_df: pd.DataFrame,
    rcsi_df: pd.DataFrame,
    verdict_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> dict[str, float | int | str]:
    """Create one comparison row for one strategy."""
    trade_path = ensure_trade_file_exists(ticker, agent_name)
    trade_df = load_trade_data(trade_path, allow_empty=True)
    strategy_curve_df = load_saved_strategy_curve(ticker, agent_name)
    if strategy_curve_df is None:
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
    reference_rcsi = pd.to_numeric(
        pd.Series([verdict_row.get("reference_rcsi", rcsi_row.get("RCSI"))]),
        errors="coerce",
    ).iloc[0]
    reference_rcsi_z = pd.to_numeric(
        pd.Series([verdict_row.get("reference_rcsi_z", rcsi_row.get("RCSI_z"))]),
        errors="coerce",
    ).iloc[0]
    reference_p_value = pd.to_numeric(
        pd.Series([verdict_row.get("reference_p_value", monte_carlo_row.get("p_value"))]),
        errors="coerce",
    ).iloc[0]
    adjusted_p_value = pd.to_numeric(
        pd.Series(
            [
                verdict_row.get(
                    "reference_adjusted_p_value",
                    monte_carlo_row.get("bh_adjusted_p_value"),
                )
            ]
        ),
        errors="coerce",
    ).iloc[0]
    reference_percentile = pd.to_numeric(
        pd.Series(
            [verdict_row.get("reference_percentile", monte_carlo_row.get("actual_percentile"))]
        ),
        errors="coerce",
    ).iloc[0]
    displayed_p_value_prominence = (
        calculate_p_value_prominence(float(reference_p_value))
        if has_completed_trades and pd.notna(reference_p_value)
        else pd.NA
    )

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
        "RCSI": float(reference_rcsi) if has_completed_trades and pd.notna(reference_rcsi) else pd.NA,
        "RCSI_z": float(reference_rcsi_z) if has_completed_trades and pd.notna(reference_rcsi_z) else pd.NA,
        "p_value": float(reference_p_value) if has_completed_trades and pd.notna(reference_p_value) else pd.NA,
        "adjusted_p_value": (
            float(adjusted_p_value)
            if has_completed_trades and pd.notna(adjusted_p_value)
            else pd.NA
        ),
        "p_value_prominence": displayed_p_value_prominence,
        "actual_percentile": (
            float(reference_percentile)
            if has_completed_trades and pd.notna(reference_percentile)
            else pd.NA
        ),
        "number_of_trades": int(len(trade_df)),
        "number_of_periods": int(curve_summary["number_of_periods"]),
        "transaction_cost": TRANSACTION_COST,
        "research_grade": bool(verdict_row.get("research_grade", False)),
        "artifact_run_id": str(verdict_row.get("artifact_run_id", "")),
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


def comparison_research_grade(comparison_df: pd.DataFrame) -> bool:
    """Return whether the evaluated strategy rows are all research-grade."""
    strategy_rows = comparison_df.loc[
        comparison_df["agent"].astype(str).str.strip() != BENCHMARK_NAME
    ]
    if strategy_rows.empty or "research_grade" not in strategy_rows.columns:
        return False

    research_grade_values = _coerce_bool_series(strategy_rows["research_grade"]).dropna()
    if research_grade_values.empty:
        return False

    return bool(research_grade_values.all())


def main() -> None:
    """Build the full comparison table for the active strategies."""
    monte_carlo_summary_df, rcsi_df = load_strategy_summary_tables()
    activity_df = load_trade_activity_validation()
    try:
        verdict_df = load_strategy_verdicts(ticker)
    except ValueError as error:
        message = str(error)
        if "Monte Carlo summary is stale" not in message:
            raise
        print(
            "Detected stale Monte Carlo artifacts while loading verdicts. "
            "Rebuilding Monte Carlo + RCSI once for alignment..."
        )
        create_monte_carlo()
        create_rcsi()
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
            "adjusted_p_value",
            "p_value_prominence",
            "actual_percentile",
            "number_of_trades",
            "number_of_periods",
            "transaction_cost",
            "research_grade",
            "artifact_run_id",
        ],
    )

    comparison_df["agent"] = pd.Categorical(
        comparison_df["agent"],
        categories=STRATEGY_ORDER,
        ordered=True,
    )
    comparison_df = comparison_df.sort_values("agent").reset_index(drop=True)
    comparison_df["agent"] = comparison_df["agent"].astype(str)

    comparison_df = comparison_df.merge(
        activity_df[
            [
                "agent",
                "signal_count",
                "trades_per_month",
                "average_holding_bars",
                "average_holding_days",
                "average_holding_hours",
                "activity_status",
                "activity_warning",
            ]
        ],
        on="agent",
        how="left",
    )

    full_output_path = data_clean_dir() / f"{ticker}_full_comparison.csv"
    write_dataframe_artifact(
        comparison_df,
        full_output_path,
        producer="compare_agents.main",
        current_ticker=ticker,
        dependencies=[
            data_clean_dir() / f"{ticker}_monte_carlo_summary.csv",
            data_clean_dir() / f"{ticker}_rcsi.csv",
        ],
        research_grade=comparison_research_grade(comparison_df),
        canonical_policy="always",
        parameters={
            "artifact_type": "full_comparison",
        },
    )
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
        "adjusted_p_value",
        "actual_percentile",
        "number_of_trades",
        "trades_per_month",
        "average_holding_days",
        "activity_status",
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

    random_verdict_row = verdict_df.loc[verdict_df["agent"].astype(str).str.strip() == "random"]
    if not random_verdict_row.empty:
        random_verdict_row = random_verdict_row.iloc[0]
        random_bucket = str(random_verdict_row["evidence_bucket"]).strip()
        if random_bucket in {"strong_skill", "moderate_skill", "suspicious"}:
            print(
                "\nWARNING: Random baseline calibration review required. "
                f"Random is currently classified as {random_verdict_row['verdict_label']}."
            )
        elif random_bucket == "weak_skill":
            print(
                "\nNOTE: Random baseline reached Weak Skill on this ticker. "
                "That can happen by chance, but it should be monitored."
            )


if __name__ == "__main__":
    main()
