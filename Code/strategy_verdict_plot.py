"""Create a one-page academic skill-vs-luck summary chart for the active ticker."""

from __future__ import annotations

import math
import os

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import pandas as pd

from plot_config import (
    AGENT_COLORS,
    ANNOTATION_SIZE,
    BACKGROUND_COLOR,
    GRID_COLOR,
    SPINE_COLOR,
    SUBTITLE_SIZE,
    TABLE_BODY_SIZE,
    TABLE_HEADER_SIZE,
    TEXT_COLOR,
    TITLE_SIZE,
    add_note_box,
    format_agent_name,
    save_chart,
    show_chart,
)

try:
    from strategy_verdicts import EVIDENCE_COLORS, VERDICT_COLORS, load_strategy_verdicts
except ModuleNotFoundError:
    from Code.strategy_verdicts import EVIDENCE_COLORS, VERDICT_COLORS, load_strategy_verdicts


ticker = os.environ.get("TICKER", "SPY")


def format_numeric_or_na(value, decimals: int) -> str:
    """Format numeric table values while keeping missing entries explicit."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "n/a"

    if math.isnan(numeric_value):
        return "n/a"

    return f"{numeric_value:.{decimals}f}"


def add_rect(ax, x0: float, y0: float, width: float, height: float, **kwargs) -> None:
    """Add a square-cornered rectangle in axis coordinates."""
    ax.add_patch(
        Rectangle(
            (x0, y0),
            width,
            height,
            transform=ax.transAxes,
            **kwargs,
        )
    )


def cell_center(x0: float, width: float, y0: float, height: float) -> tuple[float, float]:
    """Return the visual center of one table cell."""
    return x0 + width / 2, y0 + height / 2


def draw_table(ax, verdict_df) -> None:
    """Draw a clean academic summary table."""
    left = 0.03
    bottom = 0.23
    width = 0.94
    height = 0.55

    col_widths = [0.16, 0.20, 0.12, 0.11, 0.10, 0.10, 0.10, 0.11]
    x_edges = [left]
    for col_width in col_widths:
        x_edges.append(x_edges[-1] + width * col_width)

    header_height = 0.11
    row_height = (height - header_height) / max(len(verdict_df), 1)

    add_rect(
        ax,
        x0=left,
        y0=bottom,
        width=width,
        height=height,
        facecolor="#FFFFFF",
        edgecolor=SPINE_COLOR,
        linewidth=0.9,
    )
    add_rect(
        ax,
        x0=left,
        y0=bottom + height - header_height,
        width=width,
        height=header_height,
        facecolor="#F3F5F8",
        edgecolor=SPINE_COLOR,
        linewidth=0.9,
    )

    headers = ["Strategy", "Evidence", "Verdict", "Confidence", "p-value", "Prom.", "Pctile", "RCSI_z"]
    for index, header in enumerate(headers):
        x0 = x_edges[index]
        x1 = x_edges[index + 1]
        x_center, y_center = cell_center(
            x0=x0,
            width=x1 - x0,
            y0=bottom + height - header_height,
            height=header_height,
        )
        ax.text(
            x_center,
            y_center,
            header,
            transform=ax.transAxes,
            fontsize=TABLE_HEADER_SIZE,
            fontweight="semibold",
            color=TEXT_COLOR,
            ha="center",
            va="center",
        )

    for x_edge in x_edges[1:-1]:
        ax.plot(
            [x_edge, x_edge],
            [bottom, bottom + height],
            transform=ax.transAxes,
            color=GRID_COLOR,
            linewidth=0.8,
        )

    ax.plot(
        [left, left + width],
        [bottom + height - header_height, bottom + height - header_height],
        transform=ax.transAxes,
        color=SPINE_COLOR,
        linewidth=0.9,
    )

    for row_index, (_, row) in enumerate(verdict_df.iterrows()):
        y0 = bottom + height - header_height - ((row_index + 1) * row_height)
        row_fill = "#FFFFFF" if row_index % 2 == 0 else "#F8FAFC"
        add_rect(
            ax,
            x0=left,
            y0=y0,
            width=width,
            height=row_height,
            facecolor=row_fill,
            edgecolor="none",
        )
        if row_index < len(verdict_df) - 1:
            ax.plot(
                [left, left + width],
                [y0, y0],
                transform=ax.transAxes,
                color=GRID_COLOR,
                linewidth=0.8,
            )

        strategy_text = format_agent_name(str(row["agent"]))
        evidence_text = str(row["evidence_label"])
        verdict_text = str(row["verdict_label"])
        confidence_text = str(row["confidence_label"])
        p_value_value = pd.to_numeric(pd.Series([row["reference_p_value"]]), errors="coerce").iloc[0]
        p_value_text = format_numeric_or_na(p_value_value, 3)
        prominence_text = "n/a"
        if pd.notna(p_value_value):
            prominence_text = f"{(-math.log10(max(float(p_value_value), 1e-12))):.2f}"
        percentile_text = format_numeric_or_na(row["reference_percentile"], 1)
        rcsi_z_text = format_numeric_or_na(row["reference_rcsi_z"], 2)

        row_values = [
            strategy_text,
            evidence_text,
            verdict_text,
            confidence_text,
            p_value_text,
            prominence_text,
            percentile_text,
            rcsi_z_text,
        ]
        row_colors = [
            AGENT_COLORS.get(str(row["agent"]), TEXT_COLOR),
            EVIDENCE_COLORS.get(str(row["evidence_bucket"]), TEXT_COLOR),
            VERDICT_COLORS.get(str(row["skill_luck_verdict"]), TEXT_COLOR),
            TEXT_COLOR,
            TEXT_COLOR,
            TEXT_COLOR,
            TEXT_COLOR,
            TEXT_COLOR,
        ]
        row_weights = ["semibold", "semibold", "bold", "normal", "normal", "normal", "normal", "normal"]

        for cell_index, (value, color, weight) in enumerate(
            zip(row_values, row_colors, row_weights)
        ):
            x0 = x_edges[cell_index]
            x1 = x_edges[cell_index + 1]
            x_center, y_center = cell_center(
                x0=x0,
                width=x1 - x0,
                y0=y0,
                height=row_height,
            )
            ax.text(
                x_center,
                y_center,
                value,
                transform=ax.transAxes,
                fontsize=TABLE_BODY_SIZE,
                fontweight=weight,
                color=color,
                ha="center",
                va="center",
            )


def main() -> None:
    """Create and save the skill-vs-luck summary chart."""
    output_filename = f"{ticker}_skill_luck_summary.png"
    verdict_df = load_strategy_verdicts(ticker)

    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.axis("off")

    ax.text(
        0.03,
        0.955,
        f"{ticker}: Strategy Evidence and Confidence Overview",
        transform=ax.transAxes,
        fontsize=TITLE_SIZE + 0.7,
        fontweight="semibold",
        color=TEXT_COLOR,
        ha="left",
        va="top",
    )
    ax.text(
        0.03,
        0.905,
        "Evidence labels use scale-free statistics: p-value, percentile, normalized RCSI, robustness, and stability.",
        transform=ax.transAxes,
        fontsize=SUBTITLE_SIZE,
        color="#4B5563",
        ha="left",
        va="top",
    )

    ax.plot([0.03, 0.97], [0.875, 0.875], transform=ax.transAxes, color=SPINE_COLOR, linewidth=0.9)

    draw_table(ax, verdict_df)

    add_note_box(
        ax,
        (
            "Interpretation:\n"
            "Prom. = -log10(p), so larger values indicate rarer outcomes under the null.\n"
            "Pctile and RCSI_z provide scale-free evidence and are more comparable across tickers.\n"
            "Benchmark context appears in the comparison table and equity curve rather than this verdict summary."
        ),
        x=0.03,
        y=0.05,
        ha="left",
        va="bottom",
    )
    save_chart(fig, output_filename)
    show_chart()


if __name__ == "__main__":
    main()
