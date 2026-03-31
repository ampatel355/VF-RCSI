"""Create a clean, publication-quality RCSI bar chart for the current ticker."""

import os

from plot_config import (
    AGENT_COLORS,
    BAR_EDGE_COLOR,
    DEFAULT_FIGSIZE,
    TEXT_COLOR,
    ZERO_LINE_COLOR,
    add_note_box,
    apply_axis_number_format,
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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


def load_rcsi_data(input_path):
    """Load and validate the RCSI table for the active ticker."""
    df = load_csv_checked(
        input_path,
        required_columns=["agent", "RCSI"],
    )

    df["agent"] = df["agent"].astype(str).str.strip()
    df["RCSI"] = pd.to_numeric(df["RCSI"], errors="coerce")
    df = df.dropna(subset=["agent", "RCSI"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No usable RCSI rows were found in: {input_path}")

    duplicate_agents = df["agent"][df["agent"].duplicated()].unique().tolist()
    if duplicate_agents:
        raise ValueError(
            "Each strategy should appear only once in the RCSI file. "
            f"Duplicate rows found for: {duplicate_agents}"
        )

    available_agents = [agent for agent in AGENT_ORDER if agent in df["agent"].tolist()]
    if not available_agents:
        raise ValueError(f"No expected strategy names were found in: {input_path}")

    # Keep the strategy order consistent, but still allow missing strategies.
    df = df.set_index("agent").reindex(available_agents).reset_index()
    verdict_df = load_strategy_verdicts(ticker)[
        ["agent", "compact_evidence_label", "confidence_label", "evidence_bucket"]
    ]
    df = df.merge(verdict_df, on="agent", how="left")
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


def label_offset(values: pd.Series) -> float:
    """Place labels near bars without pushing them outside the chart."""
    max_absolute_value = float(values.abs().max())
    return max(max_absolute_value * 0.02, 0.008)


def add_bar_labels(
    ax,
    bars,
    values: pd.Series,
    evidence_labels: list[str],
    offset: float,
) -> None:
    """Add clean value plus evidence labels above positive bars and below negative bars."""
    for bar, value, evidence_label in zip(bars, values, evidence_labels):
        y_position = value + offset if value >= 0 else value - offset
        vertical_alignment = "bottom" if value >= 0 else "top"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_position,
            f"{float(value):.3f}\n{evidence_label}",
            ha="center",
            va=vertical_alignment,
            fontsize=8.4,
            color=TEXT_COLOR,
            fontweight="semibold",
        )


def add_zoom_inset(ax, df: pd.DataFrame, x_positions: np.ndarray) -> None:
    """Add a small inset to help read the non-dominant bars when needed."""
    values = df["RCSI"].astype(float)
    largest_index = int(values.abs().idxmax())
    smaller_values = values.drop(index=largest_index)

    if smaller_values.empty:
        return

    zoom_min = float(smaller_values.min())
    zoom_max = float(smaller_values.max())
    zoom_span = max(zoom_max - zoom_min, 0.02)
    zoom_padding = zoom_span * 0.28

    inset = inset_axes(
        ax,
        width="42%",
        height="40%",
        loc="upper right",
        borderpad=1.2,
    )

    inset_values = values.copy()
    inset_values.iloc[largest_index] = np.nan

    inset_bars = inset.bar(
        x_positions,
        inset_values,
        color=[AGENT_COLORS.get(agent, "#355C7D") for agent in df["agent"]],
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.55,
        width=0.58,
    )

    inset.set_ylim(zoom_min - zoom_padding, zoom_max + zoom_padding)
    inset.set_xticks(x_positions)
    inset.set_xticklabels([format_agent_name(agent) for agent in df["agent"]], fontsize=8)
    inset.axhline(0, color=ZERO_LINE_COLOR, linewidth=0.8)
    inset.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)
    inset.spines["left"].set_linewidth(0.7)
    inset.spines["bottom"].set_linewidth(0.7)
    inset.tick_params(axis="y", labelsize=8)
    inset.tick_params(axis="x", pad=1)
    inset.set_title("Zoom on smaller values", fontsize=8.5, pad=4)
    apply_axis_number_format(inset)

    inset_offset = max(zoom_span * 0.08, 0.006)
    for bar, value in zip(inset_bars, inset_values):
        if np.isnan(value):
            continue

        y_position = value + inset_offset if value >= 0 else value - inset_offset
        vertical_alignment = "bottom" if value >= 0 else "top"
        inset.text(
            bar.get_x() + bar.get_width() / 2,
            y_position,
            f"{float(value):.3f}",
            ha="center",
            va=vertical_alignment,
            fontsize=7.5,
            color=TEXT_COLOR,
        )


def main() -> None:
    """Build and save the RCSI chart for the active ticker."""
    input_path = data_clean_dir() / f"{ticker}_rcsi.csv"
    output_filename = f"{ticker}_rcsi_bar_chart.png"

    df = load_rcsi_data(input_path)
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
    evidence_labels = df["compact_evidence_label"].fillna("Inconclusive").tolist()
    x_positions = np.arange(len(df))
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

    ax.set_xticks(x_positions)
    ax.set_xticklabels([format_agent_name(agent) for agent in df["agent"]])
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

    add_bar_labels(ax, bars, values, evidence_labels, offset=label_offset(values))

    add_note_box(
        ax,
        (
            "Positive RCSI indicates performance above the simulated baseline (skill).\n"
            "Negative RCSI suggests performance consistent with or below randomness.\n"
            "Labels show the overall evidence class, not RCSI alone."
        ),
        x=0.98,
        y=0.05,
        ha="right",
        va="bottom",
    )

    # Keep the main figure on a normal linear scale. When one bar is much larger
    # than the rest, add a small inset so the smaller values remain interpretable.
    if dominant_ratio(values) >= 8 and len(values) > 1:
        add_zoom_inset(ax, df, x_positions)

    save_chart(fig, output_filename)
    show_chart()


if __name__ == "__main__":
    main()
