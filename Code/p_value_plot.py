"""Create a publication-style p-value chart from the Monte Carlo summary."""

import os

from plot_config import (
    ACTUAL_LINE_COLOR,
    AGENT_COLORS,
    BAR_EDGE_COLOR,
    DEFAULT_FIGSIZE,
    add_subtitle,
    apply_clean_style,
    create_placeholder_chart,
    data_clean_dir,
    emphasize_tiny_bars,
    format_agent_name,
    load_csv_checked,
    save_chart,
    show_chart,
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
    """Load the Monte Carlo summary rows needed for the p-value chart."""
    input_path = data_clean_dir() / f"{ticker}_monte_carlo_summary.csv"
    df = load_csv_checked(
        input_path,
        required_columns=["agent", "p_value"],
    )

    df["agent"] = df["agent"].astype(str).str.strip()
    df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    df = df.dropna(subset=["agent", "p_value"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No usable p-value rows were found in: {input_path}")

    available_agents = [agent for agent in AGENT_ORDER if agent in df["agent"].tolist()]
    if not available_agents:
        raise ValueError(f"No expected strategy names were found in: {input_path}")

    df = df.set_index("agent").reindex(available_agents).reset_index()
    verdict_df = load_strategy_verdicts(ticker)[
        ["agent", "compact_evidence_label", "confidence_label", "evidence_bucket"]
    ]
    df = df.merge(verdict_df, on="agent", how="left")
    return df.loc[df["evidence_bucket"] != "no_trades"].reset_index(drop=True)


def add_value_labels(
    ax,
    x_positions: np.ndarray,
    values: np.ndarray,
    evidence_labels: list[str],
) -> None:
    """Add compact p-value plus evidence labels above the bars."""
    offset = 0.02

    for x_position, value, evidence_label in zip(x_positions, values, evidence_labels):
        ax.text(
            x_position,
            min(value + offset, 1.045),
            f"{value:.3f}\n{evidence_label}",
            ha="center",
            va="bottom",
            fontsize=8.4,
            fontweight="semibold",
        )


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
    evidence_labels = df["compact_evidence_label"].fillna("Inconclusive").tolist()
    colors = [AGENT_COLORS.get(agent, "#355C7D") for agent in df["agent"]]

    figure_width = max(DEFAULT_FIGSIZE[0], 1.7 * len(df) + 2.0)
    fig, ax = plt.subplots(figsize=(figure_width, DEFAULT_FIGSIZE[1]))
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
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([format_agent_name(agent) for agent in df["agent"]])

    apply_clean_style(
        ax,
        title=f"{ticker}: Monte Carlo p-Value by Strategy",
        x_label="Strategy",
        y_label="p-value",
        show_y_grid=True,
        add_legend=False,
    )
    add_subtitle(
        ax,
        "The dashed line marks p = 0.05. Lower values indicate stronger evidence against the simulated-random baseline. Labels show the overall evidence class.",
    )
    ax.set_ylim(0, 1.08)
    emphasize_tiny_bars(ax, bars, values)

    add_value_labels(ax, x_positions, values, evidence_labels)

    save_chart(fig, output_filename)
    show_chart()


if __name__ == "__main__":
    main()
