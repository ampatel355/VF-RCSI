"""Create a publication-style grouped trade-level return-ratio chart across market regimes."""

import os
from pathlib import Path

from plot_config import (
    AGENT_COLORS,
    BAR_EDGE_COLOR,
    DEFAULT_FIGSIZE,
    REGIME_DISPLAY_NAMES,
    REGIME_ORDER,
    ZERO_LINE_COLOR,
    add_subtitle,
    apply_axis_number_format,
    apply_clean_style,
    create_placeholder_chart,
    data_clean_dir,
    emphasize_tiny_bars,
    format_agent_name,
    format_precise_value,
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


def load_regime_analysis(input_path: Path) -> pd.DataFrame:
    """Load, validate, and clean the regime analysis table."""
    df = load_csv_checked(
        input_path,
        required_columns=[
            "agent",
            "regime_at_entry",
            "total_trades",
            "meets_min_trade_threshold",
            "plot_trade_level_return_ratio",
        ],
    )

    df["agent"] = df["agent"].astype(str).str.strip()
    df["regime_at_entry"] = df["regime_at_entry"].astype(str).str.strip().str.lower()
    df["total_trades"] = pd.to_numeric(df["total_trades"], errors="coerce")
    df["meets_min_trade_threshold"] = (
        df["meets_min_trade_threshold"].astype(str).str.lower().map({"true": True, "false": False})
    )
    df["plot_trade_level_return_ratio"] = pd.to_numeric(
        df["plot_trade_level_return_ratio"], errors="coerce"
    )
    df = df.dropna(subset=["agent", "regime_at_entry", "total_trades"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No usable regime analysis data was found in: {input_path}")

    return df


def main() -> None:
    """Build the grouped trade-level return-ratio chart for the active ticker."""
    input_path = data_clean_dir() / f"{ticker}_regime_analysis.csv"
    output_filename = f"{ticker}_regime_sharpe_chart.png"
    df = load_regime_analysis(input_path)

    metric_pivot_df = (
        df.pivot_table(
            index="regime_at_entry",
            columns="agent",
            values="plot_trade_level_return_ratio",
            aggfunc="mean",
        )
        .reindex(REGIME_ORDER)
        .reindex(columns=AGENT_ORDER)
    )
    trade_count_pivot_df = (
        df.pivot_table(
            index="regime_at_entry",
            columns="agent",
            values="total_trades",
            aggfunc="mean",
        )
        .reindex(REGIME_ORDER)
        .reindex(columns=AGENT_ORDER)
    )
    threshold_pivot_df = (
        df.pivot_table(
            index="regime_at_entry",
            columns="agent",
            values="meets_min_trade_threshold",
            aggfunc="first",
        )
        .reindex(REGIME_ORDER)
        .reindex(columns=AGENT_ORDER)
    )

    if metric_pivot_df.dropna(how="all").empty:
        create_placeholder_chart(
            title=f"{ticker}: Trade-Level Return Ratio by Strategy and Market Regime",
            output_filename=output_filename,
            subtitle="No regime-level strategy cells met the minimum trade threshold.",
            message=(
                "All regime cells were suppressed because the strategy logs contain too few\n"
                "completed trades for reliable regime-level interpretation."
            ),
        )
        return

    x_positions = np.arange(len(REGIME_ORDER))
    bar_width = min(0.78 / max(len(AGENT_ORDER), 1), 0.23)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    bar_groups = []
    group_center = (len(AGENT_ORDER) - 1) / 2

    for index, agent_name in enumerate(AGENT_ORDER):
        heights = metric_pivot_df[agent_name].to_numpy(dtype=float)
        trade_counts = trade_count_pivot_df[agent_name].to_numpy(dtype=float)
        meets_threshold = threshold_pivot_df[agent_name].fillna(False).to_numpy(dtype=bool)
        offset = (index - group_center) * bar_width

        bars = ax.bar(
            x_positions + offset,
            np.nan_to_num(heights, nan=0.0),
            width=bar_width,
            color=AGENT_COLORS[agent_name],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=0.55,
            label=format_agent_name(agent_name),
        )
        for bar, is_reliable in zip(bars, meets_threshold):
            if not is_reliable:
                bar.set_facecolor("#D9DDE4")
                bar.set_edgecolor("#8B95A5")
                bar.set_hatch("//")
        bar_groups.append((bars, heights, trade_counts, meets_threshold))

    ax.set_xticks(x_positions)
    ax.set_xticklabels(REGIME_DISPLAY_NAMES)
    ax.axhline(0, color=ZERO_LINE_COLOR, linewidth=1.0)

    apply_clean_style(
        ax,
        title=f"{ticker}: Trade-Level Return Ratio by Strategy and Market Regime",
        x_label="Market Regime",
        y_label="Trade-Level Return Ratio (unitless)",
        show_y_grid=True,
        add_legend=False,
    )
    add_subtitle(
        ax,
        "Gray hatched bars are suppressed because the regime cell has fewer than 20 trades.",
    )
    apply_axis_number_format(ax)

    ax.legend(
        frameon=False,
        fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=min(len(AGENT_ORDER), 3),
        handlelength=1.8,
        borderaxespad=0.0,
    )

    finite_values = metric_pivot_df.to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    value_span = max(float(finite_values.max() - finite_values.min()), 0.18)
    label_offset = value_span * 0.05
    y_padding = value_span * 0.18

    ax.set_ylim(
        min(float(finite_values.min()) - y_padding, -y_padding * 0.25),
        max(float(finite_values.max()) + y_padding, y_padding * 0.25),
    )
    ax.margins(x=0.08)

    for bars, heights, _, _ in bar_groups:
        emphasize_tiny_bars(ax, bars, heights, min_fraction_of_axis=0.014)

    for bars, heights, trade_counts, meets_threshold in bar_groups:
        for bar, height, trade_count, is_reliable in zip(bars, heights, trade_counts, meets_threshold):
            if not is_reliable:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.02 if ax.get_ylim()[1] > 0 else ax.get_ylim()[0] * 0.02,
                    f"n={int(trade_count)}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color="#5B6471",
                    rotation=90,
                )
                continue

            y_position = height + label_offset if height >= 0 else height - label_offset
            vertical_alignment = "bottom" if height >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_position,
                format_precise_value(float(height)),
                ha="center",
                va=vertical_alignment,
                fontsize=9,
            )

    save_chart(fig, output_filename)
    show_chart()


if __name__ == "__main__":
    main()
