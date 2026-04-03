"""Create a clean, publication-quality RCSI bar chart for the current ticker."""

import os

from plot_config import (
    AGENT_COLORS,
    BAR_EDGE_COLOR,
    TEXT_COLOR,
    ZERO_LINE_COLOR,
    add_figure_caption,
    apply_categorical_tick_labels,
    apply_axis_number_format,
    apply_clean_style,
    create_placeholder_chart,
    emphasize_tiny_bars,
    format_agent_name,
    save_chart,
    show_chart,
    size_for_categories,
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from strategy_verdicts import load_strategy_verdicts
    from strategy_config import AGENT_ORDER
except ModuleNotFoundError:
    from Code.strategy_verdicts import load_strategy_verdicts
    from Code.strategy_config import AGENT_ORDER


# Read the active ticker from the environment, or fall back to SPY.
ticker = os.environ.get("TICKER", "SPY")


def load_rcsi_data():
    """Load the classifier-aligned RCSI table for the active ticker."""
    df = load_strategy_verdicts(ticker)[
        [
            "agent",
            "reference_rcsi",
            "compact_evidence_label",
            "confidence_label",
            "evidence_bucket",
        ]
    ].copy()

    df["agent"] = df["agent"].astype(str).str.strip()
    df["RCSI"] = pd.to_numeric(df["reference_rcsi"], errors="coerce")
    df = df.dropna(subset=["agent", "RCSI"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No usable classifier-aligned RCSI rows were found for: {ticker}")

    duplicate_agents = df["agent"][df["agent"].duplicated()].unique().tolist()
    if duplicate_agents:
        raise ValueError(
            "Each strategy should appear only once in the RCSI view. "
            f"Duplicate rows found for: {duplicate_agents}"
        )

    available_agents = [agent for agent in AGENT_ORDER if agent in df["agent"].tolist()]
    if not available_agents:
        raise ValueError(f"No expected strategy names were found for: {ticker}")

    df = df.set_index("agent").reindex(available_agents).reset_index()
    return df.loc[df["evidence_bucket"] != "no_trades"].reset_index(drop=True)


def dominant_ratio(values: pd.Series) -> float:
    """Measure whether one RCSI value dominates the others visually."""
    absolute_values = values.abs().sort_values(ascending=False).reset_index(drop=True)

    if len(absolute_values) < 2:
        return 1.0

    second_largest = float(absolute_values.iloc[1])
    if second_largest == 0:
        return np.inf if float(absolute_values.iloc[0]) > 0 else 1.0

    return float(absolute_values.iloc[0] / second_largest)


def calculate_y_limits(values: pd.Series) -> tuple[float, float]:
    """Use a symmetric range around zero so the scale stays honest and readable."""
    max_absolute_value = float(values.abs().max())

    # Give the chart a stable shape even if values are tiny or all zero.
    if max_absolute_value == 0:
        max_absolute_value = 0.1

    limit = max_absolute_value * 1.22
    return -limit, limit


def main() -> None:
    """Build and save the RCSI chart for the active ticker."""
    output_filename = f"{ticker}_rcsi_bar_chart.png"

    df = load_rcsi_data()
    if df.empty:
        create_placeholder_chart(
            title=f"{ticker}: Regime-Conditional Skill Index by Strategy",
            output_filename=output_filename,
            subtitle="No inferential RCSI values are available for this ticker.",
            message=(
                "No completed trades were generated for any active strategy.\n"
                "RCSI is therefore not meaningful for this ticker."
            ),
        )
        return

    values = df["RCSI"].astype(float)
    x_positions = np.arange(len(df))
    colors = [AGENT_COLORS.get(agent, "#355C7D") for agent in df["agent"]]

    fig, ax = plt.subplots(figsize=size_for_categories(len(df), height=6.0))
    bars = ax.bar(
        x_positions,
        values,
        color=colors,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.6,
        width=0.60,
        zorder=3,
    )

    ax.set_xticks(x_positions)
    apply_categorical_tick_labels(
        ax,
        [format_agent_name(agent) for agent in df["agent"]],
    )
    ax.axhline(0, color=ZERO_LINE_COLOR, linewidth=1.0, zorder=2)

    apply_clean_style(
        ax,
        title=f"{ticker}: Regime-Conditional Skill Index by Strategy",
        x_label="Strategy",
        y_label="RCSI (Actual Return − Median Monte Carlo Return, decimal)",
        show_y_grid=True,
        add_legend=False,
    )
    apply_axis_number_format(ax)

    y_lower, y_upper = calculate_y_limits(values)
    ax.set_ylim(y_lower, y_upper)
    ax.margins(x=0.10)
    emphasize_tiny_bars(ax, bars, values)

    add_figure_caption(
        fig,
        (
            "Positive RCSI indicates performance above the strategy-specific Monte Carlo "
            "median, while negative values indicate performance at or below the random "
            "baseline. Classification is determined jointly from RCSI_z, p-value, and "
            "percentile in the comparison tables."
        ),
    )

    save_chart(fig, output_filename)
    show_chart()


if __name__ == "__main__":
    main()
