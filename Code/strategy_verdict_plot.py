"""Create a one-page Skill vs Luck classification summary chart for the active ticker."""

from __future__ import annotations

import math
import os

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import pandas as pd

from plot_config import (
    AGENT_COLORS,
    BACKGROUND_COLOR,
    GRID_COLOR,
    SPINE_COLOR,
    SUBTITLE_SIZE,
    TABLE_BODY_SIZE,
    TABLE_HEADER_SIZE,
    TEXT_COLOR,
    TITLE_SIZE,
    add_figure_caption,
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


def compact_cell_text(text: str, max_chars: int) -> str:
    """Compact long cell labels to avoid spillover in narrow table columns."""
    value = str(text).strip()
    if len(value) <= max_chars:
        return value
    return f"{value[: max_chars - 1]}…"


def draw_table(ax, verdict_df) -> None:
    """Draw a clean academic summary table."""
    left = 0.03
    bottom = 0.23
    width = 0.94
    height = 0.55

    col_widths = [0.18, 0.20, 0.12, 0.10, 0.10, 0.09, 0.09, 0.12]
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

    headers = ["Strategy", "Classification", "Verdict", "Confidence", "p-value", "Prom.", "Pctile", "RCSI_z"]
    header_fontsize = min(TABLE_HEADER_SIZE, 8.5)
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
            fontsize=header_fontsize,
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

    row_count = len(verdict_df)
    body_fontsize = 7.8 if row_count >= 8 else 8.2
    compact_fontsize = max(7.0, body_fontsize - 0.5)

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

        strategy_text = compact_cell_text(format_agent_name(str(row["agent"]), short=True), max_chars=15)
        evidence_text = compact_cell_text(str(row["evidence_label"]), max_chars=13)
        verdict_text = compact_cell_text(str(row["verdict_label"]), max_chars=13)
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
            is_text_heavy_column = cell_index in {0, 1, 2}
            cell_fontsize = compact_fontsize if is_text_heavy_column else body_fontsize
            if is_text_heavy_column:
                x_center = x0 + (x1 - x0) * 0.03
            alignment = "left" if is_text_heavy_column else "center"
            ax.text(
                x_center,
                y_center,
                value,
                transform=ax.transAxes,
                fontsize=cell_fontsize,
                fontweight=weight,
                color=color,
                ha=alignment,
                va="center",
                linespacing=0.98,
            )


def main() -> None:
    """Create and save the skill-vs-luck summary chart."""
    output_filename = f"{ticker}_skill_luck_summary.png"
    verdict_df = load_strategy_verdicts(ticker)

    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.axis("off")

    ax.text(
        0.03,
        0.955,
        f"{ticker}: Skill vs Luck Classification",
        transform=ax.transAxes,
        fontsize=TITLE_SIZE,
        fontweight="normal",
        color=TEXT_COLOR,
        ha="left",
        va="top",
    )
    ax.text(
        0.03,
        0.905,
        "Classification uses all three metrics together: p-value, percentile, and RCSI_z (distance from randomness).",
        transform=ax.transAxes,
        fontsize=SUBTITLE_SIZE,
        color="#4B5563",
        ha="left",
        va="top",
    )

    ax.plot([0.03, 0.97], [0.875, 0.875], transform=ax.transAxes, color=SPINE_COLOR, linewidth=0.9)

    draw_table(ax, verdict_df)

    add_figure_caption(
        fig,
        (
            "Green rows indicate stronger evidence of skill, gray indicates Random / Luck, "
            "red indicates Negative Skill, and orange indicates Suspicious metric disagreement. "
            "Prom. reports -log10(p), so larger values indicate rarer outcomes under the null. "
            "Classification uses RCSI_z, p-value, and percentile together."
        ),
    )
    save_chart(fig, output_filename)
    show_chart()


if __name__ == "__main__":
    main()
