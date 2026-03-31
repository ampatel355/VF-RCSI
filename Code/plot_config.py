"""Shared academic styling helpers for all project charts."""

import os
from pathlib import Path
import warnings


# Every script reads the active ticker from the same environment variable.
ticker = os.environ.get("TICKER", "SPY")

# Matplotlib writes small config files when it starts.
# This keeps those files inside the project instead of a system folder.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib"),
)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.ticker import FuncFormatter, MaxNLocator
import pandas as pd

try:
    from strategy_config import AGENT_COLORS, AGENT_DISPLAY_NAMES, AGENT_ORDER
except ModuleNotFoundError:
    from Code.strategy_config import AGENT_COLORS, AGENT_DISPLAY_NAMES, AGENT_ORDER


# The master pipeline can turn off interactive popups while charts are being built.
show_plots = os.environ.get("SHOW_PLOTS", "1") == "1"

# Use an academic serif theme inspired by paper figures.
ACADEMIC_SERIF_FONTS = [
    "Times New Roman",
    "Times",
    "Nimbus Roman",
    "Liberation Serif",
    "DejaVu Serif",
    "serif",
]

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ACADEMIC_SERIF_FONTS,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        "text.color": "#222222",
        "axes.labelcolor": "#222222",
        "axes.edgecolor": "#4B5563",
        "axes.linewidth": 0.8,
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "grid.color": "#D4D9E1",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.65,
        "axes.titlepad": 10,
    }
)


# Shared figure sizing and text scale.
DEFAULT_FIGSIZE = (10, 6)
TITLE_SIZE = 14.5
SUBTITLE_SIZE = 9.25
LABEL_SIZE = 12
TICK_SIZE = 10.5
LEGEND_SIZE = 10
ANNOTATION_SIZE = 9.5
TABLE_HEADER_SIZE = 10.5
TABLE_BODY_SIZE = 10

# Paper-style muted color theme.
REGIME_ORDER = ["calm", "neutral", "stressed"]
REGIME_DISPLAY_NAMES = ["Calm", "Neutral", "Stressed"]

ACTUAL_LINE_COLOR = "#A23B49"
MEDIAN_LINE_COLOR = "#2F2F2F"
ZERO_LINE_COLOR = "#6B7280"
GRID_COLOR = "#D7DBE2"
TEXT_COLOR = "#222222"
SPINE_COLOR = "#4B5563"
BACKGROUND_COLOR = "#FFFFFF"
NOTE_BOX_FACE_COLOR = "#FFFFFF"
BAR_EDGE_COLOR = "#454C56"

# Diverging palette for heatmaps.
HEATMAP_COLORS = [
    "#3F5D7D",
    "#7193B0",
    "#DADADA",
    "#D6A36B",
    "#A55A4A",
]


def project_root() -> Path:
    """Return the root folder of the project."""
    return Path(__file__).resolve().parents[1]


def resolve_named_dir(lowercase_name: str, uppercase_name: str) -> Path:
    """Return a project folder, supporting either lower- or upper-case names."""
    root = project_root()
    uppercase_dir = root / uppercase_name
    lowercase_dir = root / lowercase_name

    if uppercase_dir.exists():
        return uppercase_dir
    if lowercase_dir.exists():
        return lowercase_dir

    uppercase_dir.mkdir(parents=True, exist_ok=True)
    return uppercase_dir


def data_clean_dir() -> Path:
    """Return the clean-data folder."""
    return resolve_named_dir("data_clean", "Data_Clean")


def charts_dir() -> Path:
    """Return the charts folder."""
    return resolve_named_dir("charts", "Charts")


def format_agent_name(agent_name: str) -> str:
    """Convert an internal agent name into a cleaner label for charts."""
    return AGENT_DISPLAY_NAMES.get(
        agent_name,
        agent_name.replace("_", " ").title(),
    )


def load_csv_checked(input_path: Path, required_columns: list[str]) -> pd.DataFrame:
    """Load a CSV file and confirm that it contains the needed columns."""
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError(f"The input file is empty: {input_path}")

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(
            f"Missing required columns in {input_path}: {', '.join(missing_columns)}"
        )

    return df


def lighten_color(color: str, amount: float = 0.35) -> str:
    """Create a lighter shade of a color for fills and histogram bars."""
    red, green, blue = mcolors.to_rgb(color)
    new_red = 1 - (1 - red) * (1 - amount)
    new_green = 1 - (1 - green) * (1 - amount)
    new_blue = 1 - (1 - blue) * (1 - amount)
    return mcolors.to_hex((new_red, new_green, new_blue))


def histogram_color(agent_name: str) -> str:
    """Return a lighter histogram color for one strategy."""
    return lighten_color(AGENT_COLORS.get(agent_name, "#355C7D"), amount=0.48)


def fill_color(agent_name: str) -> str:
    """Return a lighter fill color for equity-curve shading."""
    return lighten_color(AGENT_COLORS.get(agent_name, "#355C7D"), amount=0.68)


def apply_bar_style(bars, linewidth: float = 0.6) -> None:
    """Give bar charts one consistent academic edge treatment."""
    for bar in bars:
        bar.set_edgecolor(BAR_EDGE_COLOR)
        bar.set_linewidth(linewidth)


def emphasize_tiny_bars(
    ax,
    bars,
    values,
    min_fraction_of_axis: float = 0.018,
) -> None:
    """Draw a thin cap line on near-zero bars so they remain visible."""
    y_min, y_max = ax.get_ylim()
    axis_span = abs(y_max - y_min)
    visibility_threshold = axis_span * min_fraction_of_axis

    for bar, value in zip(bars, values):
        if pd.isna(value):
            continue

        numeric_value = float(value)
        if abs(numeric_value) >= visibility_threshold:
            continue

        ax.hlines(
            y=numeric_value,
            xmin=bar.get_x(),
            xmax=bar.get_x() + bar.get_width(),
            colors=[bar.get_facecolor()],
            linewidth=2.4,
            zorder=max(bar.get_zorder() + 0.5, 4),
        )


def format_large_number(value: float) -> str:
    """Format a numeric value in a readable way for axes and notes."""
    absolute_value = abs(value)

    if absolute_value < 1e-12:
        return "0"
    if absolute_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if absolute_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    if absolute_value >= 10:
        return f"{value:,.1f}"
    if absolute_value >= 1:
        return f"{value:,.2f}"
    if absolute_value >= 0.01:
        return f"{value:.3f}"
    return f"{value:.4f}"


def format_precise_value(value: float) -> str:
    """Format value labels with a little more precision than axis ticks."""
    absolute_value = abs(value)

    if absolute_value >= 100:
        return f"{value:,.2f}"
    if absolute_value >= 1:
        return f"{value:,.2f}"
    if absolute_value >= 0.01:
        return f"{value:.3f}"
    return f"{value:.4f}"


def axis_number_formatter():
    """Return a compact axis formatter."""
    return FuncFormatter(lambda value, _: format_large_number(value))


def apply_axis_number_format(ax, axis: str = "y") -> None:
    """Apply the shared numeric formatter to one axis."""
    formatter = axis_number_formatter()

    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
    else:
        ax.yaxis.set_major_formatter(formatter)


def apply_clean_style(
    ax,
    title: str,
    x_label: str,
    y_label: str,
    show_y_grid: bool = True,
    add_legend: bool = False,
    legend_location: str = "best",
    legend_ncol: int = 1,
) -> None:
    """Apply one consistent academic style to a matplotlib axis."""
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.figure.set_facecolor(BACKGROUND_COLOR)
    ax.set_title(
        title,
        fontsize=TITLE_SIZE,
        fontweight="semibold",
        color=TEXT_COLOR,
        y=1.065,
        pad=0,
    )
    ax.set_xlabel(x_label, fontsize=LABEL_SIZE, color=TEXT_COLOR)
    ax.set_ylabel(y_label, fontsize=LABEL_SIZE, color=TEXT_COLOR)
    ax.tick_params(axis="both", labelsize=TICK_SIZE, colors=TEXT_COLOR)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))

    if show_y_grid:
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.65, color=GRID_COLOR)
        ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    if add_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                labels,
                frameon=False,
                fontsize=LEGEND_SIZE,
                loc=legend_location,
                ncol=legend_ncol,
                handlelength=1.8,
                borderaxespad=0.4,
            )


def add_note_box(
    ax,
    text: str,
    x: float = 0.02,
    y: float = 0.98,
    ha: str = "left",
    va: str = "top",
) -> None:
    """Add a small paper-style annotation box inside a chart."""
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=ANNOTATION_SIZE,
        ha=ha,
        va=va,
        color=TEXT_COLOR,
        zorder=6,
        bbox={
            "facecolor": NOTE_BOX_FACE_COLOR,
            "alpha": 0.96,
            "edgecolor": GRID_COLOR,
            "boxstyle": "square,pad=0.30",
        },
    )


def add_subtitle(ax, text: str) -> None:
    """Add a small subtitle-like note just below the title."""
    ax.text(
        0.5,
        1.02,
        text,
        transform=ax.transAxes,
        fontsize=SUBTITLE_SIZE,
        ha="center",
        va="bottom",
        color="#4B5563",
    )


def create_placeholder_chart(
    title: str,
    output_filename: str,
    message: str,
    subtitle: str | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> Path:
    """Create a clean placeholder chart when a figure has no valid data to plot."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.axis("off")

    ax.text(
        0.5,
        0.72,
        title,
        transform=ax.transAxes,
        fontsize=TITLE_SIZE,
        fontweight="semibold",
        color=TEXT_COLOR,
        ha="center",
        va="center",
    )

    if subtitle:
        ax.text(
            0.5,
            0.62,
            subtitle,
            transform=ax.transAxes,
            fontsize=SUBTITLE_SIZE,
            color="#4B5563",
            ha="center",
            va="center",
        )

    ax.text(
        0.5,
        0.42,
        message,
        transform=ax.transAxes,
        fontsize=LABEL_SIZE,
        color=TEXT_COLOR,
        ha="center",
        va="center",
        linespacing=1.5,
    )

    output_path = save_chart(fig, output_filename)
    show_chart()
    return output_path


def save_chart(fig, filename: str) -> Path:
    """Save a chart with consistent publication-ready export settings."""
    output_path = charts_dir() / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    return output_path


def show_chart() -> None:
    """Display the figure without blocking the rest of the pipeline."""
    if not show_plots:
        plt.close("all")
        return

    plt.show(block=False)
    plt.pause(0.1)
