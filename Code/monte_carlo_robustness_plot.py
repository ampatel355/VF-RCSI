"""Create publication-style robustness charts from Monte Carlo seed sweeps."""

import os

from plot_config import (
    AGENT_COLORS,
    BAR_EDGE_COLOR,
    ZERO_LINE_COLOR,
    add_figure_caption,
    apply_categorical_tick_labels,
    apply_clean_style,
    create_placeholder_chart,
    data_clean_dir,
    emphasize_tiny_bars,
    format_agent_name,
    load_csv_checked,
    save_chart,
    show_chart,
    size_for_categories,
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


def annotate_interval_values(
    ax,
    bars,
    means: np.ndarray,
    stds: np.ndarray,
    *,
    decimals: int,
) -> None:
    """Annotate bars with mean ± SD so interval values are visible on the figure."""
    y_min, y_max = ax.get_ylim()
    axis_span = max(abs(y_max - y_min), 1e-9)
    offset = axis_span * 0.018

    for bar, mean_value, std_value in zip(bars, means, stds):
        if np.isnan(mean_value) or np.isnan(std_value):
            continue

        upper_bound = mean_value + std_value
        label_y = upper_bound + offset
        vertical_align = "bottom"

        # Keep text inside bounds when bars are already near the top edge.
        if label_y > (y_max - offset * 0.5):
            label_y = y_max - offset * 0.4
            vertical_align = "top"

        ax.text(
            bar.get_x() + (bar.get_width() / 2),
            label_y,
            f"{mean_value:.{decimals}f}\n±{std_value:.{decimals}f}",
            ha="center",
            va=vertical_align,
            fontsize=7.4,
            color="#1F2937",
            linespacing=1.0,
            zorder=4,
        )


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


def create_rcsi_stability_chart(summary_df: pd.DataFrame) -> None:
    """Create the RCSI mean-plus-SD chart across seeds."""
    output_filename = f"{ticker}_monte_carlo_robustness_rcsi.png"
    x_positions = np.arange(len(summary_df))
    means = summary_df["mean_RCSI"].to_numpy(dtype=float)
    stds = summary_df["std_RCSI"].fillna(0.0).to_numpy(dtype=float)
    colors = [AGENT_COLORS.get(agent, "#355C7D") for agent in summary_df["agent"]]

    fig, ax = plt.subplots(figsize=size_for_categories(len(summary_df), height=6.0))
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
    apply_categorical_tick_labels(
        ax,
        [format_agent_name(agent) for agent in summary_df["agent"]],
    )

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
    rcsi_padding = rcsi_span * 0.24
    ax.set_ylim(rcsi_low - rcsi_padding, rcsi_high + rcsi_padding)
    emphasize_tiny_bars(ax, bars, means)
    annotate_interval_values(ax, bars, means, stds, decimals=3)

    add_figure_caption(
        fig,
        (
            f"Bars show mean RCSI across robustness seeds with one standard deviation. "
            f"Outer runs = {int(summary_df['number_of_outer_runs'].iloc[0])}; "
            f"simulations per run = {int(summary_df['simulations_per_run'].iloc[0])}; "
            f"transaction cost = {float(summary_df['transaction_cost'].iloc[0]):.6f}."
        ),
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

    fig, ax = plt.subplots(figsize=size_for_categories(len(summary_df), height=6.0))
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
    apply_categorical_tick_labels(
        ax,
        [format_agent_name(agent) for agent in summary_df["agent"]],
    )

    apply_clean_style(
        ax,
        title=f"{ticker}: Monte Carlo Robustness of Actual Percentile Across Seeds",
        x_label="Strategy",
        y_label="Actual Percentile in Simulated Distribution (%)",
        show_y_grid=True,
        add_legend=False,
    )

    percentile_low = max(float(np.min(means - stds)) - 5, 0)
    percentile_high = min(float(np.max(means + stds)) + 12, 106)
    ax.set_ylim(percentile_low, percentile_high)
    emphasize_tiny_bars(ax, bars, means)
    annotate_interval_values(ax, bars, means, stds, decimals=1)

    add_figure_caption(
        fig,
        "Bars show mean actual percentile across robustness seeds with one standard deviation. "
        "Higher values place the observed outcome deeper into the right tail of the simulated distribution.",
    )

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
