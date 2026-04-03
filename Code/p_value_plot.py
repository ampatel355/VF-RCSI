"""Create a publication-style p-value chart from the Monte Carlo summary."""

import os

from plot_config import (
    ACTUAL_LINE_COLOR,
    AGENT_COLORS,
    BAR_EDGE_COLOR,
    add_figure_caption,
    apply_categorical_tick_labels,
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
SIGNIFICANCE_LEVEL = 0.05


def load_p_value_data() -> pd.DataFrame:
    """Load the classifier-aligned p-values used in the chart."""
    verdict_df = load_strategy_verdicts(ticker)[
        [
            "agent",
            "reference_p_value",
            "compact_evidence_label",
            "confidence_label",
            "evidence_bucket",
        ]
    ].copy()
    verdict_df["agent"] = verdict_df["agent"].astype(str).str.strip()
    verdict_df["p_value"] = pd.to_numeric(verdict_df["reference_p_value"], errors="coerce")
    verdict_df = verdict_df.dropna(subset=["agent", "p_value"]).reset_index(drop=True)

    if verdict_df.empty:
        raise ValueError(f"No usable classifier-aligned p-value rows were found for: {ticker}")

    available_agents = [agent for agent in AGENT_ORDER if agent in verdict_df["agent"].tolist()]
    if not available_agents:
        raise ValueError(f"No expected strategy names were found for: {ticker}")

    verdict_df = verdict_df.set_index("agent").reindex(available_agents).reset_index()
    return verdict_df.loc[verdict_df["evidence_bucket"] != "no_trades"].reset_index(drop=True)


def main() -> None:
    """Create the Monte Carlo p-value chart for the active ticker."""
    output_filename = f"{ticker}_p_value_chart.png"
    df = load_p_value_data()
    if df.empty:
        create_placeholder_chart(
            title=f"{ticker}: Monte Carlo p-Value by Strategy",
            output_filename=output_filename,
            subtitle="No inferential p-values are available for this ticker.",
            message=(
                "No completed trades were generated for any active strategy.\n"
                "Skill-versus-luck p-value comparisons are therefore not applicable."
            ),
        )
        return

    x_positions = np.arange(len(df))
    values = df["p_value"].to_numpy(dtype=float)
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
    ax.axhline(
        SIGNIFICANCE_LEVEL,
        color=ACTUAL_LINE_COLOR,
        linestyle="--",
        linewidth=1.2,
        zorder=2,
        label="p = 0.05 threshold",
    )

    ax.set_xticks(x_positions)
    apply_categorical_tick_labels(
        ax,
        [format_agent_name(agent) for agent in df["agent"]],
    )

    apply_clean_style(
        ax,
        title=f"{ticker}: Monte Carlo p-Value by Strategy",
        x_label="Strategy",
        y_label="p-value",
        show_y_grid=True,
        add_legend=True,
        legend_location="upper right",
    )
    ax.set_ylim(0, 1.08)
    emphasize_tiny_bars(ax, bars, values)

    add_figure_caption(
        fig,
        (
            "Lower p-values indicate that the observed strategy result was less common under "
            "the strategy-specific Monte Carlo null. The dashed reference line marks the "
            "conventional 0.05 threshold."
        ),
    )

    save_chart(fig, output_filename)
    show_chart()


if __name__ == "__main__":
    main()
