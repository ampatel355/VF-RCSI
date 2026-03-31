"""Create publication-style robustness charts from Monte Carlo seed sweeps."""

import os

from plot_config import (
    AGENT_COLORS,
    BAR_EDGE_COLOR,
    DEFAULT_FIGSIZE,
    ZERO_LINE_COLOR,
    add_note_box,
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
    from strategy_config import AGENT_ORDER
except ModuleNotFoundError:
    from Code.strategy_config import AGENT_ORDER


# Read the active ticker from the environment, or fall back to SPY.
ticker = os.environ.get("TICKER", "SPY")


def load_robustness_data():
    """Load the robustness runs file and the aggregated summary file."""
    runs_path = data_clean_dir() / f"{ticker}_monte_carlo_robustness_runs.csv"
    summary_path = data_clean_dir() / f"{ticker}_monte_carlo_robustness_summary.csv"

    runs_df = load_csv_checked(
        runs_path,
        required_columns=[
            "agent",
            "outer_run",
            "seed_used",
            "RCSI",
            "actual_percentile",
            "p_value",
        ],
    )
    summary_df = load_csv_checked(
        summary_path,
        required_columns=[
            "agent",
            "mean_percentile",
            "std_percentile",
            "min_percentile",
            "max_percentile",
            "mean_p_value",
            "std_p_value",
            "mean_RCSI",
            "std_RCSI",
            "min_RCSI",
            "max_RCSI",
            "number_of_outer_runs",
            "simulations_per_run",
            "transaction_cost",
        ],
    )

    runs_df["agent"] = runs_df["agent"].astype(str).str.strip()
    summary_df["agent"] = summary_df["agent"].astype(str).str.strip()

    numeric_columns = [
        "mean_percentile",
        "std_percentile",
        "min_percentile",
        "max_percentile",
        "mean_p_value",
        "std_p_value",
        "mean_RCSI",
        "std_RCSI",
        "min_RCSI",
        "max_RCSI",
        "number_of_outer_runs",
        "simulations_per_run",
        "transaction_cost",
    ]
    for column in numeric_columns:
        summary_df[column] = pd.to_numeric(summary_df[column], errors="coerce")

    if "mean_number_of_trades" in summary_df.columns:
        summary_df["mean_number_of_trades"] = pd.to_numeric(
            summary_df["mean_number_of_trades"],
            errors="coerce",
        )
        summary_df = summary_df.loc[summary_df["mean_number_of_trades"] > 0].reset_index(drop=True)

    summary_df = summary_df.dropna(subset=["agent", "mean_RCSI", "mean_percentile"])

    summary_df["agent"] = pd.Categorical(
        summary_df["agent"],
        categories=AGENT_ORDER,
        ordered=True,
    )
    summary_df = summary_df.sort_values("agent").reset_index(drop=True)
    summary_df["agent"] = summary_df["agent"].astype(str)

    return runs_df, summary_df


def add_value_labels(ax, x_positions, means, stds, decimals: int) -> None:
    """Place small mean-plus-minus-SD labels above each bar."""
    span = max(float(np.max(means + stds) - np.min(means - stds)), 1.0)
    offset = span * 0.04
    y_min, y_max = ax.get_ylim()
    top_buffer = span * 0.05
    bottom_buffer = span * 0.05

    for x_position, mean_value, std_value in zip(x_positions, means, stds):
        label_text = f"{mean_value:.{decimals}f}\n±{std_value:.{decimals}f}"
        label_y = mean_value + std_value + offset if mean_value >= 0 else mean_value - std_value - offset
        vertical_alignment = "bottom" if mean_value >= 0 else "top"

        if mean_value >= 0 and label_y > (y_max - top_buffer):
            label_y = mean_value - std_value - offset
            vertical_alignment = "top"
        elif mean_value < 0 and label_y < (y_min + bottom_buffer):
            label_y = mean_value + std_value + offset
            vertical_alignment = "bottom"

        ax.text(
            x_position,
            label_y,
            label_text,
            ha="center",
            va=vertical_alignment,
            fontsize=8.8,
        )


def create_rcsi_stability_chart(summary_df: pd.DataFrame) -> None:
    """Create the RCSI mean-plus-SD chart across seeds."""
    output_filename = f"{ticker}_monte_carlo_robustness_rcsi.png"
    x_positions = np.arange(len(summary_df))
    means = summary_df["mean_RCSI"].to_numpy(dtype=float)
    stds = summary_df["std_RCSI"].fillna(0.0).to_numpy(dtype=float)
    colors = [AGENT_COLORS.get(agent, "#355C7D") for agent in summary_df["agent"]]

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    bars = ax.bar(
        x_positions,
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.6,
        width=0.60,
        zorder=3,
    )
    ax.axhline(0, color=ZERO_LINE_COLOR, linewidth=0.9, zorder=2)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([format_agent_name(agent) for agent in summary_df["agent"]])

    apply_clean_style(
        ax,
        title=f"{ticker}: Monte Carlo Robustness of RCSI Across Seeds",
        x_label="Strategy",
        y_label="RCSI (mean ± SD across seeds, decimal)",
        show_y_grid=True,
        add_legend=False,
    )

    rcsi_low = float(np.min(means - stds))
    rcsi_high = float(np.max(means + stds))
    rcsi_span = max(rcsi_high - rcsi_low, 0.5)
    rcsi_padding = rcsi_span * 0.18
    ax.set_ylim(rcsi_low - rcsi_padding, rcsi_high + rcsi_padding)
    emphasize_tiny_bars(ax, bars, means)

    add_value_labels(ax, x_positions, means, stds, decimals=3)

    add_note_box(
        ax,
        (
            f"Outer runs: {int(summary_df['number_of_outer_runs'].iloc[0])}\n"
            f"Simulations per run: {int(summary_df['simulations_per_run'].iloc[0])}\n"
            f"Transaction cost: {float(summary_df['transaction_cost'].iloc[0]):.6f}"
        ),
        x=0.98,
        y=0.97,
        ha="right",
        va="top",
    )

    save_chart(fig, output_filename)
    show_chart()


def create_percentile_stability_chart(summary_df: pd.DataFrame) -> None:
    """Create the actual-percentile mean-plus-SD chart across seeds."""
    output_filename = f"{ticker}_monte_carlo_robustness_percentile.png"
    x_positions = np.arange(len(summary_df))
    means = summary_df["mean_percentile"].to_numpy(dtype=float)
    stds = summary_df["std_percentile"].fillna(0.0).to_numpy(dtype=float)
    colors = [AGENT_COLORS.get(agent, "#355C7D") for agent in summary_df["agent"]]

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    bars = ax.bar(
        x_positions,
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.6,
        width=0.60,
        zorder=3,
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([format_agent_name(agent) for agent in summary_df["agent"]])

    apply_clean_style(
        ax,
        title=f"{ticker}: Monte Carlo Robustness of Actual Percentile Across Seeds",
        x_label="Strategy",
        y_label="Actual Percentile in Simulated Distribution (%)",
        show_y_grid=True,
        add_legend=False,
    )
    add_subtitle(
        ax,
        "Higher values place the observed outcome deeper into the right tail of the simulated distribution.",
    )

    percentile_low = max(float(np.min(means - stds)) - 5, 0)
    percentile_high = min(float(np.max(means + stds)) + 8, 104)
    ax.set_ylim(percentile_low, percentile_high)
    emphasize_tiny_bars(ax, bars, means)

    add_value_labels(ax, x_positions, means, stds, decimals=1)

    save_chart(fig, output_filename)
    show_chart()


def main() -> None:
    """Create both publication-style robustness charts."""
    _, summary_df = load_robustness_data()
    if summary_df.empty:
        create_placeholder_chart(
            title=f"{ticker}: Monte Carlo Robustness of RCSI Across Seeds",
            output_filename=f"{ticker}_monte_carlo_robustness_rcsi.png",
            subtitle="No repeated-run robustness chart is available for this ticker.",
            message=(
                "No completed trades were available for repeated robustness testing.\n"
                "The robustness RCSI chart is therefore not applicable."
            ),
        )
        create_placeholder_chart(
            title=f"{ticker}: Monte Carlo Robustness of Actual Percentile Across Seeds",
            output_filename=f"{ticker}_monte_carlo_robustness_percentile.png",
            subtitle="No repeated-run robustness chart is available for this ticker.",
            message=(
                "No completed trades were available for repeated robustness testing.\n"
                "The robustness percentile chart is therefore not applicable."
            ),
        )
        return

    create_rcsi_stability_chart(summary_df)
    create_percentile_stability_chart(summary_df)


if __name__ == "__main__":
    main()
