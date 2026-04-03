"""Create a publication-style heatmap of regime performance with low-sample suppression."""

import os
from pathlib import Path

from plot_config import (
    HEATMAP_COLORS,
    REGIME_DISPLAY_NAMES,
    REGIME_ORDER,
    add_figure_caption,
    apply_categorical_tick_labels,
    apply_clean_style,
    create_placeholder_chart,
    data_clean_dir,
    format_agent_name,
    load_csv_checked,
    save_chart,
    show_chart,
    size_for_heatmap,
)
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
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
VALUE_LABEL = "Trade-Level Return Ratio (mean/std, Sharpe-like per trade)"


def load_regime_analysis(input_path: Path) -> pd.DataFrame:
    """Load, validate, and clean the regime analysis table."""
    df = load_csv_checked(
        input_path,
        required_columns=[
            "agent",
            "regime_at_entry",
            "total_trades",
            "min_trades_required",
            "meets_min_trade_threshold",
            VALUE_COLUMN,
        ],
    )

    df["agent"] = df["agent"].astype(str).str.strip()
    df["regime_at_entry"] = df["regime_at_entry"].astype(str).str.strip().str.lower()
    df["total_trades"] = pd.to_numeric(df["total_trades"], errors="coerce")
    df["min_trades_required"] = pd.to_numeric(df["min_trades_required"], errors="coerce")
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


def _relative_luminance(rgb_triplet: tuple[float, float, float]) -> float:
    """Measure perceived brightness so annotation text color stays readable."""

    def to_linear(channel: float) -> float:
        if channel <= 0.04045:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    red, green, blue = rgb_triplet
    return (
        0.2126 * to_linear(red)
        + 0.7152 * to_linear(green)
        + 0.0722 * to_linear(blue)
    )


def annotate_heatmap(
    ax,
    image,
    heatmap_values: np.ndarray,
    trade_counts: np.ndarray,
) -> None:
    """Write readable cell values into the heatmap while leaving suppressed cells blank."""
    for row_index in range(heatmap_values.shape[0]):
        for column_index in range(heatmap_values.shape[1]):
            value = heatmap_values[row_index, column_index]
            if np.isnan(value):
                continue

            rgba = image.cmap(image.norm(value))
            luminance = _relative_luminance((rgba[0], rgba[1], rgba[2]))
            text_color = "#111827" if luminance > 0.46 else "#FFFFFF"

            trade_count = trade_counts[row_index, column_index]
            count_suffix = ""
            if np.isfinite(trade_count):
                count_suffix = f"\n(n={int(round(trade_count))})"

            text_artist = ax.text(
                column_index,
                row_index,
                f"{value:.2f}{count_suffix}",
                ha="center",
                va="center",
                fontsize=8.2,
                color=text_color,
                linespacing=1.1,
            )
            stroke_color = "#FFFFFF" if text_color != "#FFFFFF" else "#111827"
            text_artist.set_path_effects(
                [path_effects.withStroke(linewidth=1.0, foreground=stroke_color, alpha=0.38)]
            )


def main() -> None:
    """Build the regime heatmap for the active ticker."""
    input_path = data_clean_dir() / f"{ticker}_regime_analysis.csv"
    output_filename = f"{ticker}_regime_heatmap.png"

    df = load_regime_analysis(input_path)
    min_trades_required = int(df["min_trades_required"].dropna().iloc[0]) if df["min_trades_required"].notna().any() else 10
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
            figsize=size_for_heatmap(len(AGENT_ORDER), len(REGIME_ORDER)),
        )
        return

    heatmap_values = heatmap_df.to_numpy(dtype=float)
    trade_counts = trade_count_df.to_numpy(dtype=float)

    academic_cmap = LinearSegmentedColormap.from_list(
        "academic_diverging",
        HEATMAP_COLORS,
    )
    academic_cmap.set_bad(color="#D6DCE6")

    fig, ax = plt.subplots(figsize=size_for_heatmap(len(AGENT_ORDER), len(REGIME_ORDER)))
    image = ax.imshow(
        heatmap_values,
        cmap=academic_cmap,
        aspect="auto",
        interpolation="nearest",
        norm=build_color_norm(heatmap_values),
    )
    # Keep the visual footprint square inside the square chart canvas.
    ax.set_box_aspect(1.0)

    apply_clean_style(
        ax,
        title=f"{ticker}: Strategy Performance Across Market Regimes",
        x_label="Market Regime",
        y_label="Strategy",
        show_y_grid=False,
        add_legend=False,
    )

    ax.set_xticks(np.arange(len(REGIME_ORDER)))
    apply_categorical_tick_labels(ax, REGIME_DISPLAY_NAMES)
    ax.set_yticks(np.arange(len(AGENT_ORDER)))
    y_labels = [format_agent_name(agent_name, short=True) for agent_name in AGENT_ORDER]
    ax.set_yticklabels(y_labels, fontsize=8.9)
    for label in ax.get_yticklabels():
        label.set_rotation(0)
        label.set_ha("right")
        label.set_va("center")

    # Draw very light cell borders so each block remains readable in print.
    ax.set_xticks(np.arange(-0.5, len(REGIME_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(AGENT_ORDER), 1), minor=True)
    ax.grid(which="minor", color="#D5DEE8", linestyle="-", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    annotate_heatmap(
        ax=ax,
        image=image,
        heatmap_values=heatmap_values,
        trade_counts=trade_counts,
    )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.043, pad=0.022)
    colorbar.ax.tick_params(labelsize=8.8)
    colorbar.set_label(VALUE_LABEL, fontsize=9.4)

    suppressed_cells = int(np.isnan(heatmap_values).sum())
    add_figure_caption(
        fig,
        "Cell colors summarize regime-level trade performance. Blank light-gray cells are "
        f"suppressed for low sample size (fewer than {min_trades_required} trades). "
        f"Suppressed cells: {suppressed_cells}.",
    )

    save_chart(fig, output_filename)
    show_chart()


if __name__ == "__main__":
    main()
