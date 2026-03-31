"""Generate a concise research-style conclusion from a comparison table."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

try:
    from strategy_config import AGENT_DISPLAY_NAMES, BENCHMARK_NAME
except ModuleNotFoundError:
    from Code.strategy_config import AGENT_DISPLAY_NAMES, BENCHMARK_NAME


NUMERIC_COLUMNS = [
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
]

STRATEGY_CONCEPTS = {
    "trend": "Trend follows persistent price direction and tries to stay aligned with broader moves.",
    "mean_reversion": "Mean Reversion buys weakness and expects price to snap back toward a recent average.",
    "random": "Random is a control strategy with no market thesis and serves as a noise baseline.",
    "momentum": "Momentum buys assets that have already been moving upward over the selected lookback window.",
    "breakout": "Breakout enters when price pushes through recent highs and tries to capture expansion moves.",
    BENCHMARK_NAME: "Buy and Hold is the passive reference point with no tactical timing.",
}

REQUIRED_COLUMNS = {"agent", "cumulative_return"}


@dataclass(frozen=True)
class ConclusionSection:
    """One titled section of the generated conclusion."""

    title: str
    body: str


def can_build_comparison_conclusion(comparison_df: pd.DataFrame | None) -> bool:
    """Return whether the DataFrame contains the fields needed for interpretation."""
    if comparison_df is None or comparison_df.empty:
        return False
    return REQUIRED_COLUMNS.issubset(set(comparison_df.columns))


def _normalize_comparison_df(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize comparison data into a predictable working DataFrame."""
    working_df = comparison_df.copy()
    working_df["agent"] = working_df["agent"].astype(str).str.strip()

    for column in NUMERIC_COLUMNS:
        if column in working_df.columns:
            working_df[column] = pd.to_numeric(working_df[column], errors="coerce")

    return working_df


def _strategy_label(agent_name: str) -> str:
    """Return the display label for one strategy."""
    return AGENT_DISPLAY_NAMES.get(agent_name, agent_name.replace("_", " ").title())


def _concept_text(agent_name: str) -> str:
    """Return a short conceptual description for one strategy."""
    return STRATEGY_CONCEPTS.get(agent_name, "This strategy applies a rule-based entry and exit process.")


def _float_value(row: pd.Series, column: str) -> float | None:
    """Safely extract a numeric value from one row."""
    if column not in row.index:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    return float(value)


def _int_value(row: pd.Series, column: str) -> int | None:
    """Safely extract an integer-like value from one row."""
    numeric_value = _float_value(row, column)
    if numeric_value is None:
        return None
    return int(round(numeric_value))


def _format_percent(value: float | None) -> str:
    """Format a fractional value as a percent."""
    if value is None:
        return "n/a"
    return f"{value:.2%}"


def _format_decimal(value: float | None, digits: int = 2) -> str:
    """Format a scalar value with a fixed number of decimals."""
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _format_trades(value: int | None) -> str:
    """Format the trade count for display."""
    if value is None:
        return "n/a"
    return f"{value:d}"


def _ordinal_label(position: int) -> str:
    """Return a readable ordinal label."""
    labels = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
    }
    return labels.get(position, f"{position}th")


def _top_row_by_metric(
    comparison_df: pd.DataFrame,
    column: str,
    *,
    ascending: bool,
) -> pd.Series | None:
    """Return the best available row for one metric."""
    if column not in comparison_df.columns:
        return None

    metric_df = comparison_df.dropna(subset=[column]).copy()
    if metric_df.empty:
        return None

    return metric_df.sort_values(column, ascending=ascending).iloc[0]


def _market_behavior_comment(active_df: pd.DataFrame, best_row: pd.Series) -> str:
    """Infer a simple market-behavior takeaway from the relative rankings."""
    best_agent = str(best_row["agent"])
    directional_agents = {"trend", "momentum", "breakout"}

    directional_df = active_df.loc[active_df["agent"].isin(directional_agents)]
    mean_reversion_df = active_df.loc[active_df["agent"] == "mean_reversion"]

    directional_mean = (
        directional_df["cumulative_return"].mean()
        if not directional_df.empty
        else None
    )
    mean_reversion_return = (
        _float_value(mean_reversion_df.iloc[0], "cumulative_return")
        if not mean_reversion_df.empty
        else None
    )

    if best_agent in directional_agents and directional_mean is not None:
        if mean_reversion_return is not None and directional_mean > mean_reversion_return:
            return (
                "The asset appears to have rewarded directional or breakout behavior more than quick reversal trades, "
                "which is consistent with a market that spent meaningful time trending rather than cleanly oscillating around a short-term mean."
            )
        return (
            "The asset appears to have offered enough persistence in price direction for directional rules to outperform a purely reactive approach."
        )

    if best_agent == "mean_reversion":
        return (
            "The asset appears to have behaved more like a range or overshoot-and-revert market than a clean trend, "
            "which gave reversal entries better conditions than directional chasing."
        )

    if best_agent == "random":
        return (
            "A random baseline rising to the top suggests the test was noisy and that none of the structured rules established a clear edge on this asset."
        )

    return (
        "The results point to a mixed market where one rule set still came out ahead, but the edge likely depended on a narrower slice of the asset's behavior."
    )


def _build_overall_result(
    active_df: pd.DataFrame,
    best_row: pd.Series,
    benchmark_row: pd.Series | None,
) -> str:
    """Build the overall-result section."""
    best_label = _strategy_label(str(best_row["agent"]))
    best_return = _float_value(best_row, "cumulative_return")
    best_sharpe = _float_value(best_row, "annualized_sharpe")
    best_drawdown = _float_value(best_row, "max_drawdown")

    top_sharpe_row = _top_row_by_metric(active_df, "annualized_sharpe", ascending=False)
    lowest_drawdown_row = _top_row_by_metric(active_df, "max_drawdown", ascending=False)

    support_clauses: list[str] = []
    if top_sharpe_row is not None:
        top_sharpe_label = _strategy_label(str(top_sharpe_row["agent"]))
        if str(top_sharpe_row["agent"]) == str(best_row["agent"]):
            support_clauses.append(
                f"It also led the group on Sharpe ratio at {_format_decimal(_float_value(top_sharpe_row, 'annualized_sharpe'))}."
            )
        else:
            support_clauses.append(
                f"The strongest risk-adjusted result came from {top_sharpe_label} with a Sharpe ratio of {_format_decimal(_float_value(top_sharpe_row, 'annualized_sharpe'))}, "
                "so the return leader should be interpreted with that trade-off in mind."
            )

    if lowest_drawdown_row is not None:
        lowest_drawdown_label = _strategy_label(str(lowest_drawdown_row["agent"]))
        if str(lowest_drawdown_row["agent"]) == str(best_row["agent"]):
            support_clauses.append(
                f"It also had the shallowest drawdown at {_format_percent(_float_value(lowest_drawdown_row, 'max_drawdown'))}."
            )
        else:
            support_clauses.append(
                f"The most defensive profile came from {lowest_drawdown_label} with max drawdown {_format_percent(_float_value(lowest_drawdown_row, 'max_drawdown'))}."
            )

    benchmark_sentence = ""
    if benchmark_row is not None:
        benchmark_return = _float_value(benchmark_row, "cumulative_return")
        if best_return is not None and benchmark_return is not None:
            delta = best_return - benchmark_return
            benchmark_sentence = (
                f" Relative to the passive benchmark, that is a spread of {_format_percent(delta)}."
            )

    return (
        f"{best_label} was the strongest strategy in this run. "
        f"The ranking is anchored to cumulative return, and {best_label} finished first at {_format_percent(best_return)} "
        f"with annualized Sharpe {_format_decimal(best_sharpe)} and max drawdown {_format_percent(best_drawdown)}."
        f"{benchmark_sentence} "
        + " ".join(support_clauses)
    ).strip()


def _build_strategy_comparison(
    active_df: pd.DataFrame,
    benchmark_row: pd.Series | None,
) -> str:
    """Build a bullet-by-bullet comparison of every strategy."""
    ordered_df = active_df.sort_values("cumulative_return", ascending=False).reset_index(drop=True)
    lines: list[str] = []

    for position, (_, row) in enumerate(ordered_df.iterrows(), start=1):
        agent_name = str(row["agent"])
        label = _strategy_label(agent_name)
        lines.append(
            "- "
            + (
                f"{label}: {_concept_text(agent_name)} It ranked {_ordinal_label(position)} by cumulative return at {_format_percent(_float_value(row, 'cumulative_return'))}, "
                f"with Sharpe {_format_decimal(_float_value(row, 'annualized_sharpe'))}, max drawdown {_format_percent(_float_value(row, 'max_drawdown'))}, "
                f"and {_format_trades(_int_value(row, 'number_of_trades'))} trades."
            )
        )

    if benchmark_row is not None:
        lines.append(
            "- "
            + (
                f"{_strategy_label(BENCHMARK_NAME)}: {_concept_text(BENCHMARK_NAME)} "
                f"It returned {_format_percent(_float_value(benchmark_row, 'cumulative_return'))} with Sharpe {_format_decimal(_float_value(benchmark_row, 'annualized_sharpe'))} "
                f"and max drawdown {_format_percent(_float_value(benchmark_row, 'max_drawdown'))}."
            )
        )

    return "\n".join(lines)


def _build_reasoning(active_df: pd.DataFrame, best_row: pd.Series) -> str:
    """Build the reasoning section."""
    best_agent = str(best_row["agent"])
    best_label = _strategy_label(best_agent)

    return (
        f"{best_label} works by this logic: {_concept_text(best_agent)} "
        f"{_market_behavior_comment(active_df, best_row)}"
    )


def _build_reliability(active_df: pd.DataFrame, best_row: pd.Series) -> str:
    """Build the reliability section."""
    best_label = _strategy_label(str(best_row["agent"]))
    best_return = _float_value(best_row, "cumulative_return")
    runner_up_return = None
    if len(active_df) > 1:
        sorted_returns = active_df.sort_values("cumulative_return", ascending=False).reset_index(drop=True)
        runner_up_return = _float_value(sorted_returns.iloc[1], "cumulative_return")

    confidence_parts: list[str] = []

    p_value = _float_value(best_row, "p_value")
    actual_percentile = _float_value(best_row, "actual_percentile")
    rcsi_z = _float_value(best_row, "RCSI_z")
    verdict_label = str(best_row.get("skill_luck_verdict", "")).strip()
    confidence_label = str(best_row.get("confidence_label", "")).strip()

    if p_value is not None:
        if p_value <= 0.05:
            p_value_comment = "statistically strong by conventional thresholds"
        elif p_value <= 0.10:
            p_value_comment = "borderline rather than decisive"
        else:
            p_value_comment = "not especially strong statistically"
        confidence_parts.append(
            f"The best strategy's p-value was {_format_decimal(p_value, 4)}, which is {p_value_comment}."
        )

    if actual_percentile is not None:
        confidence_parts.append(
            f"Its actual return sat at the {_format_decimal(actual_percentile)}th percentile of the null distribution."
        )

    if rcsi_z is not None:
        confidence_parts.append(
            f"RCSI_z was {_format_decimal(rcsi_z)}, which helps show whether the edge looks meaningfully above noise."
        )

    if verdict_label:
        confidence_parts.append(
            f"The table classified it as {verdict_label}"
            + (f" with {confidence_label.lower()} confidence." if confidence_label else ".")
        )

    if best_return is not None and runner_up_return is not None:
        gap = best_return - runner_up_return
        if abs(gap) < 0.02:
            confidence_parts.append(
                f"The lead over the runner-up was only {_format_percent(gap)}, so the ranking is fairly tight."
            )
        elif abs(gap) >= 0.10:
            confidence_parts.append(
                f"The lead over the runner-up was {_format_percent(gap)}, which is a material performance gap."
            )
        else:
            confidence_parts.append(
                f"The lead over the runner-up was {_format_percent(gap)}, which is noticeable but not overwhelming."
            )

    if not confidence_parts:
        confidence_parts.append(
            f"Confidence is limited because {best_label} does not have enough supporting statistical fields populated in the comparison table."
        )

    return " ".join(confidence_parts)


def _build_final_takeaway(best_row: pd.Series, benchmark_row: pd.Series | None) -> str:
    """Build the final takeaway section."""
    best_label = _strategy_label(str(best_row["agent"]))
    best_return = _float_value(best_row, "cumulative_return")

    if best_return is None:
        return (
            f"The main takeaway is that {best_label} finished on top, but the missing supporting metrics mean the result should be treated as exploratory rather than conclusive."
        )

    if best_return <= 0:
        return (
            f"The main takeaway is caution: {best_label} ranked first, but even the leader failed to produce a positive cumulative return."
        )

    if benchmark_row is not None:
        benchmark_return = _float_value(benchmark_row, "cumulative_return")
        if benchmark_return is not None and best_return > benchmark_return:
            return (
                f"The clearest lesson from this test is that {best_label} was the best tactical rule on this asset and it improved on passive buy-and-hold."
            )
        if benchmark_return is not None and best_return <= benchmark_return:
            return (
                f"The clearest lesson from this test is that {best_label} was the best active rule, but passive buy-and-hold still matched or beat it."
            )

    return f"The clearest lesson from this test is that {best_label} offered the strongest active result on this asset."


def build_comparison_conclusion_sections(
    comparison_df: pd.DataFrame | None,
) -> list[ConclusionSection]:
    """Build the full interpretation as structured sections."""
    if not can_build_comparison_conclusion(comparison_df):
        return [
            ConclusionSection(
                title="Conclusion and Interpretation",
                body="The selected table does not contain the fields needed to generate a data-grounded conclusion yet.",
            )
        ]

    normalized_df = _normalize_comparison_df(comparison_df)
    benchmark_df = normalized_df.loc[normalized_df["agent"] == BENCHMARK_NAME].copy()
    active_df = normalized_df.loc[normalized_df["agent"] != BENCHMARK_NAME].copy()

    if active_df.empty:
        return [
            ConclusionSection(
                title="Conclusion and Interpretation",
                body="No active strategy rows are available in the comparison table, so there is nothing to interpret yet.",
            )
        ]

    active_df = active_df.sort_values("cumulative_return", ascending=False).reset_index(drop=True)
    best_row = active_df.iloc[0]
    benchmark_row = benchmark_df.iloc[0] if not benchmark_df.empty else None

    return [
        ConclusionSection("1. Overall Result", _build_overall_result(active_df, best_row, benchmark_row)),
        ConclusionSection("2. Strategy Comparison", _build_strategy_comparison(active_df, benchmark_row)),
        ConclusionSection("3. Reasoning", _build_reasoning(active_df, best_row)),
        ConclusionSection("4. Reliability / Confidence", _build_reliability(active_df, best_row)),
        ConclusionSection("5. Final Takeaway", _build_final_takeaway(best_row, benchmark_row)),
    ]


def build_comparison_conclusion_markdown(comparison_df: pd.DataFrame | None) -> str:
    """Render the structured conclusion as markdown."""
    sections = build_comparison_conclusion_sections(comparison_df)
    lines = ["**Conclusion and Interpretation**", ""]
    for section in sections:
        lines.append(f"**{section.title}**")
        lines.append(section.body)
        lines.append("")
    return "\n".join(lines).strip()


def build_comparison_conclusion_text(comparison_df: pd.DataFrame | None) -> str:
    """Render the structured conclusion as plain text."""
    sections = build_comparison_conclusion_sections(comparison_df)
    blocks = ["Conclusion and Interpretation"]
    for section in sections:
        blocks.append(f"{section.title}\n{section.body}")
    return "\n\n".join(blocks).strip()
