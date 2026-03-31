"""Create a publication-style heatmap of regime performance with low-sample suppression."""

import os
from pathlib import Path

from plot_config import (
    DEFAULT_FIGSIZE,
    HEATMAP_COLORS,
    REGIME_DISPLAY_NAMES,
    REGIME_ORDER,
    add_subtitle,
    apply_categorical_tick_labels,
    apply_clean_style,
    create_placeholder_chart,
    data_clean_dir,
    format_agent_name,
    load_csv_checked,
    save_chart,
    show_chart,
)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

try:
    from strategy_config import AGENT_ORDER
except ModuleNotFoundError:
    from Code.strategy_config import AGENT_ORDER


# Read the active ticker from the environment, or fall back to SPY.
ticker = os.environ.get("TICKER", "SPY")

VALUE_COLUMN = "plot_trade_level_return_ratio"
VALUE_LABEL = "Trade-Level Return Ratio (unitless)"


def load_regime_analysis(input_path: Path) -> pd.DataFrame:
    """Load, validate, and clean the regime analysis table."""
    df = load_csv_checked(
        input_path,
        required_columns=[
            "agent",
            "regime_at_entry",
            "total_trades",
            "meets_min_trade_threshold",
            VALUE_COLUMN,
        ],
    )

    df["agent"] = df["agent"].astype(str).str.strip()
    df["regime_at_entry"] = df["regime_at_entry"].astype(str).str.strip().str.lower()
    df["total_trades"] = pd.to_numeric(df["total_trades"], errors="coerce")
    df["meets_min_trade_threshold"] = (
        df["meets_min_trade_threshold"].astype(str).str.lower().map({"true": True, "false": False})
    )
    df[VALUE_COLUMN] = pd.to_numeric(df[VALUE_COLUMN], errors="coerce")
    df = df.dropna(subset=["agent", "regime_at_entry", "total_trades"]).reset_index(
        drop=True
    )

    if df.empty:
        raise ValueError(f"No usable regime analysis data was found in: {input_path}")

    return df


def build_heatmap_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot the analysis table into value and trade-count strategy-by-regime grids."""
    heatmap_df = (
        df.pivot_table(
            index="agent",
            columns="regime_at_entry",
            values=VALUE_COLUMN,
            aggfunc="mean",
        )
        .reindex(index=AGENT_ORDER)
        .reindex(columns=REGIME_ORDER)
    )

    trade_count_df = (
        df.pivot_table(
            index="agent",
            columns="regime_at_entry",
            values="total_trades",
            aggfunc="mean",
        )
        .reindex(index=AGENT_ORDER)
        .reindex(columns=REGIME_ORDER)
    )

    return heatmap_df, trade_count_df


def build_color_norm(heatmap_values: np.ndarray):
    """Center the color scale at zero when the data spans both signs."""
    finite_values = heatmap_values[np.isfinite(heatmap_values)]

    if len(finite_values) == 0:
        return None

    min_value = float(finite_values.min())
    max_value = float(finite_values.max())

    if min_value < 0 < max_value:
        limit = max(abs(min_value), abs(max_value))
        return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    return None


def annotate_heatmap(ax, heatmap_values: np.ndarray, trade_counts: np.ndarray) -> None:
    """Write each cell value into the heatmap."""
    finite_values = heatmap_values[np.isfinite(heatmap_values)]
    if len(finite_values) == 0:
        return

    midpoint = float(np.nanmean(finite_values))

    for row_index in range(heatmap_values.shape[0]):
        for column_index in range(heatmap_values.shape[1]):
            value = heatmap_values[row_index, column_index]

            trade_count = trade_counts[row_index, column_index]
            if np.isnan(value):
                ax.text(
                    column_index,
                    row_index,
                    f"Suppressed\nn={int(trade_count)}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#6B7280",
                )
                continue

            text_color = "white" if value < midpoint else "#1F2937"
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
            )


def main() -> None:
    """Build the regime heatmap for the active ticker."""
    input_path = data_clean_dir() / f"{ticker}_regime_analysis.csv"
    output_filename = f"{ticker}_regime_heatmap.png"

    df = load_regime_analysis(input_path)
    heatmap_df, trade_count_df = build_heatmap_table(df)
    if heatmap_df.dropna(how="all").empty:
        create_placeholder_chart(
            title=f"{ticker}: Strategy Performance Across Market Regimes",
            output_filename=output_filename,
            subtitle="No regime heatmap is available for the current ticker.",
            message=(
                "All regime cells were suppressed because the strategy logs contain too few\n"
                "completed trades for reliable regime-level interpretation."
            ),
            figsize=(9.4, max(6.1, 0.75 * len(AGENT_ORDER) + 2.5)),
        )
        return

    heatmap_values = heatmap_df.to_numpy(dtype=float)
    trade_counts = trade_count_df.to_numpy(dtype=float)

    academic_cmap = LinearSegmentedColormap.from_list(
        "academic_diverging",
        HEATMAP_COLORS,
    )
    academic_cmap.set_bad(color="#E5E7EB")

    figure_height = max(6.1, 0.75 * len(AGENT_ORDER) + 2.5)
    fig, ax = plt.subplots(figsize=(9.4, figure_height))
    image = ax.imshow(
        heatmap_values,
        cmap=academic_cmap,
        aspect="auto",
        norm=build_color_norm(heatmap_values),
    )

    apply_clean_style(
        ax,
        title=f"{ticker}: Strategy Performance Across Market Regimes",
        x_label="Market Regime",
        y_label="Strategy",
        show_y_grid=False,
        add_legend=False,
    )
    add_subtitle(
        ax,
        "Gray cells are suppressed because the regime cell has fewer than 20 trades.",
    )

    ax.set_xticks(np.arange(len(REGIME_ORDER)))
    apply_categorical_tick_labels(ax, REGIME_DISPLAY_NAMES)
    ax.set_yticks(np.arange(len(AGENT_ORDER)))
    apply_categorical_tick_labels(
        ax,
        [format_agent_name(agent_name) for agent_name in AGENT_ORDER],
        axis="y",
        fontsize=9.8,
    )

    # Draw light cell borders so each block looks crisp on the page.
    ax.set_xticks(np.arange(-0.5, len(REGIME_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(AGENT_ORDER), 1), minor=True)
    ax.grid(which="minor", color="#F3F4F6", linestyle="-", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)

    annotate_heatmap(ax, heatmap_values, trade_counts)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.04)
    colorbar.ax.tick_params(labelsize=10)
    colorbar.set_label(VALUE_LABEL, fontsize=11)

    save_chart(fig, output_filename)
    show_chart()


if __name__ == "__main__":
    main()
