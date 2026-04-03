"""Create publication-quality Monte Carlo histograms for each strategy."""

import os

from plot_config import (
    ACTUAL_LINE_COLOR,
    DEFAULT_FIGSIZE,
    add_figure_caption,
    apply_axis_number_format,
    apply_clean_style,
    create_placeholder_chart,
    data_clean_dir,
    format_agent_name,
    format_large_number,
    histogram_color,
    load_csv_checked,
    save_chart,
    show_chart,
)
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

try:
    from artifact_provenance import artifact_run_id
    from monte_carlo import (
        TRANSACTION_COST,
        adjust_trade_returns,
        calculate_actual_percentile,
        calculate_cumulative_return_from_log_returns,
        calculate_p_value,
        convert_to_log_returns,
        load_trade_data,
    )
    from strategy_config import AGENT_ORDER
except ModuleNotFoundError:
    from Code.artifact_provenance import artifact_run_id
    from Code.monte_carlo import (
        TRANSACTION_COST,
        adjust_trade_returns,
        calculate_actual_percentile,
        calculate_cumulative_return_from_log_returns,
        calculate_p_value,
        convert_to_log_returns,
        load_trade_data,
    )
    from Code.strategy_config import AGENT_ORDER


# Read the active ticker from the environment, or fall back to SPY.
ticker = os.environ.get("TICKER", "SPY")
DEFAULT_CLIP_RANGE = (0.005, 0.995)
TIGHT_CLIP_RANGE = (0.01, 0.99)
MONTE_CARLO_MEDIAN_LINE_COLOR = "#111111"
INTERVAL_LINE_COLOR = "#70757D"


def load_simulation_results(input_path):
    """Load the full Monte Carlo results for one strategy."""
    df = load_csv_checked(
        input_path,
        required_columns=["simulation_id", "simulated_cumulative_return"],
    )

    df["simulation_id"] = pd.to_numeric(df["simulation_id"], errors="coerce")
    df["simulated_cumulative_return"] = pd.to_numeric(
        df["simulated_cumulative_return"],
        errors="coerce",
    )
    df = df.dropna(subset=["simulation_id", "simulated_cumulative_return"]).reset_index(
        drop=True
    )

    if df.empty:
        raise ValueError(f"No usable simulation rows were found in: {input_path}")

    return df


def load_summary_row(summary_path, agent_name: str) -> pd.Series:
    """Load the Monte Carlo summary row for one strategy."""
    summary_df = load_csv_checked(
        summary_path,
        required_columns=[
            "agent",
            "actual_cumulative_return",
            "median_simulated_return",
            "mean_simulated_return",
            "std_simulated_return",
            "actual_percentile",
            "p_value",
            "lower_5pct",
            "upper_95pct",
            "number_of_trades",
            "transaction_cost",
            "simulation_count",
            "reproducible",
            "seed_used",
        ],
    )

    summary_df["agent"] = summary_df["agent"].astype(str).str.strip()
    numeric_columns = [
        "actual_cumulative_return",
        "median_simulated_return",
        "mean_simulated_return",
        "std_simulated_return",
        "actual_percentile",
        "p_value",
        "lower_5pct",
        "upper_95pct",
        "number_of_trades",
        "transaction_cost",
        "simulation_count",
    ]
    for column in numeric_columns:
        summary_df[column] = pd.to_numeric(summary_df[column], errors="coerce")

    matching_rows = summary_df.loc[summary_df["agent"] == agent_name]
    if matching_rows.empty:
        raise ValueError(
            f"No Monte Carlo summary row was found for strategy '{agent_name}' in {summary_path}"
        )

    if len(matching_rows) > 1:
        raise ValueError(
            f"Multiple Monte Carlo summary rows were found for strategy '{agent_name}' in {summary_path}"
        )

    return matching_rows.iloc[0]


def calculate_actual_return_from_trade_file(trade_path):
    """Recalculate the actual cumulative return from the trade file for validation."""
    trade_df = load_trade_data(trade_path, allow_empty=True)
    if trade_df.empty:
        return 0.0, 0

    raw_returns = trade_df["return"].to_numpy(dtype=float)
    adjusted_returns = adjust_trade_returns(
        raw_returns=raw_returns,
        transaction_cost=TRANSACTION_COST,
        input_path=trade_path,
    )
    log_returns = convert_to_log_returns(adjusted_returns)
    actual_cumulative_return = calculate_cumulative_return_from_log_returns(log_returns)
    return actual_cumulative_return, len(trade_df)


def validate_summary_against_results(
    summary_row: pd.Series,
    simulation_df: pd.DataFrame,
    actual_cumulative_return: float,
    number_of_trades: int,
) -> pd.Series:
    """Catch stale or mismatched files before plotting."""
    simulated_returns = simulation_df["simulated_cumulative_return"].to_numpy(dtype=float)
    expected_values = {
        "simulation_count": int(len(simulated_returns)),
        "median_simulated_return": float(np.median(simulated_returns)),
        "mean_simulated_return": float(np.mean(simulated_returns)),
        "std_simulated_return": float(np.std(simulated_returns, ddof=0)),
        "lower_5pct": float(np.percentile(simulated_returns, 5)),
        "upper_95pct": float(np.percentile(simulated_returns, 95)),
        "actual_percentile": float(
            calculate_actual_percentile(
                simulated_returns,
                actual_cumulative_return,
            )
        ),
        "p_value": float(
            calculate_p_value(
                simulated_returns,
                actual_cumulative_return,
            )
        ),
        "actual_cumulative_return": float(actual_cumulative_return),
        "number_of_trades": int(number_of_trades),
    }

    corrected_summary = summary_row.copy()
    critical_mismatches: list[str] = []
    recomputed_mismatches: list[str] = []

    for field_name, expected_value in expected_values.items():
        if field_name in {"simulation_count", "number_of_trades"}:
            matches = int(summary_row[field_name]) == int(expected_value)
        else:
            matches = np.isclose(
                float(summary_row[field_name]),
                float(expected_value),
                rtol=1e-9,
                atol=1e-12,
            )

        if matches:
            continue

        if field_name in {"actual_cumulative_return", "number_of_trades"}:
            critical_mismatches.append(field_name)
            continue

        corrected_summary[field_name] = expected_value
        recomputed_mismatches.append(field_name)

    if critical_mismatches:
        raise ValueError(
            "Monte Carlo summary and result files do not match for "
            f"{summary_row['agent']}. Mismatched fields: {', '.join(critical_mismatches)}"
        )

    if recomputed_mismatches:
        print(
            "Warning: recomputed stale Monte Carlo summary fields for "
            f"{summary_row['agent']}: {', '.join(recomputed_mismatches)}"
        )

    return corrected_summary


def choose_display_clip_range(simulated_returns: pd.Series) -> tuple[float, float]:
    """Choose a display clipping range for extremely skewed results."""
    q99 = float(simulated_returns.quantile(0.99))
    q995 = float(simulated_returns.quantile(0.995))
    median_value = float(simulated_returns.median())
    tail_jump = abs(q995 - q99)
    comparison_scale = max(abs(q99), abs(median_value), 1.0)

    if tail_jump / comparison_scale > 0.75:
        return TIGHT_CLIP_RANGE

    return DEFAULT_CLIP_RANGE


def build_display_window(
    simulated_returns: pd.Series,
    actual_value: float,
    median_value: float,
) -> tuple[pd.Series, float, float, float, float, bool]:
    """Create a display-only window while keeping the statistics unchanged."""
    clip_lower_q, clip_upper_q = choose_display_clip_range(simulated_returns)
    lower_bound = float(simulated_returns.quantile(clip_lower_q))
    upper_bound = float(simulated_returns.quantile(clip_upper_q))

    display_series = simulated_returns[
        simulated_returns.between(lower_bound, upper_bound, inclusive="both")
    ].copy()

    if display_series.empty:
        raise ValueError("All simulation values were removed by display clipping.")

    display_min = min(lower_bound, actual_value, median_value)
    display_max = max(upper_bound, actual_value, median_value)
    clipping_used = len(display_series) < len(simulated_returns)

    return (
        display_series,
        clip_lower_q,
        clip_upper_q,
        display_min,
        display_max,
        clipping_used,
    )


def should_use_log1p_display(
    display_series: pd.Series,
    display_min: float,
    display_max: float,
) -> bool:
    """Switch to log1p display only when a linear axis still looks compressed."""
    if float(display_series.min()) <= -0.999999:
        return False

    q25 = float(display_series.quantile(0.25))
    q75 = float(display_series.quantile(0.75))
    interquartile_range = max(q75 - q25, 1e-9)
    display_span = display_max - display_min
    return (display_span / interquartile_range) > 18


def transform_for_display(values, use_log1p_display: bool) -> np.ndarray:
    """Transform values for display only."""
    values = np.asarray(values, dtype=float)

    if use_log1p_display:
        return np.log1p(values)

    return values


def get_histogram_bins(display_values: np.ndarray) -> int:
    """Choose a sensible number of histogram bins."""
    suggested_edges = np.histogram_bin_edges(display_values, bins="fd")
    suggested_bins = max(len(suggested_edges) - 1, 1)
    return int(min(45, max(24, suggested_bins)))


def round_to_nice_value(value: float) -> float:
    """Round a positive number to a cleaner axis tick."""
    if value <= 0:
        return value

    magnitude = 10 ** np.floor(np.log10(value))
    fraction = value / magnitude

    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 2.5:
        nice_fraction = 2.5
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10

    return nice_fraction * magnitude


def build_actual_scale_ticks(display_min: float, display_max: float) -> list[float]:
    """Choose readable actual-value ticks for log1p display."""
    if display_max <= display_min:
        return [display_min, display_max]

    candidate_ticks = set()

    if display_min < 0:
        candidate_ticks.add(round(display_min, 2))
        candidate_ticks.add(0.0)

    positive_min = max(display_min, 1e-6)
    if display_max > 0:
        min_exponent = int(np.floor(np.log10(positive_min))) - 1
        max_exponent = int(np.ceil(np.log10(display_max))) + 1

        for exponent in range(min_exponent, max_exponent + 1):
            scale = 10 ** exponent
            for multiplier in [1, 2, 5]:
                candidate = round_to_nice_value(multiplier * scale)
                if positive_min <= candidate <= display_max:
                    candidate_ticks.add(float(candidate))

    candidate_ticks = sorted(candidate_ticks)
    if not candidate_ticks:
        return [display_min, display_max]

    transformed_candidates = np.log1p(candidate_ticks)
    target_positions = np.linspace(
        transformed_candidates[0],
        transformed_candidates[-1],
        num=min(7, len(candidate_ticks)),
    )

    selected_ticks = []
    used_indices = set()

    for target in target_positions:
        nearest_index = int(np.argmin(np.abs(transformed_candidates - target)))
        if nearest_index in used_indices:
            continue
        used_indices.add(nearest_index)
        selected_ticks.append(candidate_ticks[nearest_index])

    return sorted(set(selected_ticks))


def set_log1p_actual_value_ticks(ax, display_min: float, display_max: float) -> None:
    """Show real cumulative-return values on a log1p-spaced axis."""
    actual_ticks = build_actual_scale_ticks(display_min, display_max)
    transformed_ticks = np.log1p(actual_ticks)
    ax.set_xticks(transformed_ticks)
    ax.set_xticklabels([format_large_number(value) for value in actual_ticks])


def create_caption_text(
    summary_row: pd.Series,
    clipping_used: bool,
    clip_lower_q: float,
    clip_upper_q: float,
    use_log1p_display: bool,
) -> str:
    """Create a compact caption for the Monte Carlo histogram."""
    clipping_text = "No"
    if clipping_used:
        clipping_text = (
            f"Yes ({clip_lower_q * 100:.1f}th to {clip_upper_q * 100:.1f}th percentiles)"
        )

    display_mode = "Linear axis"
    if use_log1p_display:
        display_mode = "Actual values shown on log1p spacing"

    return (
        f"Actual return = {summary_row['actual_cumulative_return']:.4f}; "
        f"median simulated return = {summary_row['median_simulated_return']:.4f}; "
        f"simulated 5th/95th = ({summary_row['lower_5pct']:.4f}, {summary_row['upper_95pct']:.4f}); "
        f"p-value = {summary_row['p_value']:.4f}; "
        f"actual percentile = {summary_row['actual_percentile']:.1f}; "
        f"trades = {int(summary_row['number_of_trades'])}; "
        f"simulations = {int(summary_row['simulation_count'])}; "
        f"transaction cost = {summary_row['transaction_cost']:.6f}; "
        f"display clipping = {clipping_text}; "
        f"Display mode: {display_mode}"
    )


def create_monte_carlo_chart(agent_name: str) -> None:
    """Create one Monte Carlo histogram for a single strategy."""
    summary_path = data_clean_dir() / f"{ticker}_monte_carlo_summary.csv"
    results_path = data_clean_dir() / f"{ticker}_{agent_name}_monte_carlo_results.csv"
    trade_path = data_clean_dir() / f"{ticker}_{agent_name}_trades.csv"
    output_filename = f"{ticker}_{agent_name}_monte_carlo.png"

    simulation_df = load_simulation_results(results_path)
    summary_row = load_summary_row(summary_path, agent_name)
    summary_run_id = artifact_run_id(summary_path)
    results_run_id = artifact_run_id(results_path)
    if summary_run_id and results_run_id and summary_run_id != results_run_id:
        raise ValueError(
            "Monte Carlo summary and result files come from different runs for "
            f"{agent_name}. summary={summary_run_id}, results={results_run_id}"
        )
    actual_cumulative_return, number_of_trades = calculate_actual_return_from_trade_file(
        trade_path
    )
    if int(summary_row["number_of_trades"]) == 0 or number_of_trades == 0:
        create_placeholder_chart(
            title=(
                f"{ticker}: Monte Carlo Distribution of Final Returns, "
                f"{format_agent_name(agent_name)} Strategy"
            ),
            output_filename=output_filename,
            subtitle="No Monte Carlo histogram is available for a no-trade strategy.",
            message=(
                "This strategy produced no completed trades on the current ticker.\n"
                "A return-distribution histogram would be non-informative, so it was omitted."
            ),
        )
        return

    summary_row = validate_summary_against_results(
        summary_row=summary_row,
        simulation_df=simulation_df,
        actual_cumulative_return=actual_cumulative_return,
        number_of_trades=number_of_trades,
    )

    simulated_returns = simulation_df["simulated_cumulative_return"]
    actual_value = float(summary_row["actual_cumulative_return"])
    median_value = float(summary_row["median_simulated_return"])
    lower_5pct = float(summary_row["lower_5pct"])
    upper_95pct = float(summary_row["upper_95pct"])

    (
        display_series,
        clip_lower_q,
        clip_upper_q,
        display_min,
        display_max,
        clipping_used,
    ) = build_display_window(
        simulated_returns=simulated_returns,
        actual_value=actual_value,
        median_value=median_value,
    )

    use_log1p_display = should_use_log1p_display(
        display_series=display_series,
        display_min=display_min,
        display_max=display_max,
    )

    display_values = transform_for_display(display_series, use_log1p_display)
    actual_display_value = float(transform_for_display([actual_value], use_log1p_display)[0])
    median_display_value = float(transform_for_display([median_value], use_log1p_display)[0])
    lower_display_value = float(transform_for_display([lower_5pct], use_log1p_display)[0])
    upper_display_value = float(transform_for_display([upper_95pct], use_log1p_display)[0])
    display_left = float(transform_for_display([display_min], use_log1p_display)[0])
    display_right = float(transform_for_display([display_max], use_log1p_display)[0])

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.hist(
        display_values,
        bins=get_histogram_bins(display_values),
        color=histogram_color(agent_name),
        edgecolor="white",
        linewidth=0.8,
        alpha=0.95,
        label="Simulated outcomes",
    )
    ax.axvline(
        actual_display_value,
        color=ACTUAL_LINE_COLOR,
        linestyle="--",
        linewidth=1.9,
        label="Actual result",
    )
    ax.axvline(
        median_display_value,
        color=MONTE_CARLO_MEDIAN_LINE_COLOR,
        linestyle=":",
        linewidth=1.8,
        label="Median simulation",
    )
    ax.axvline(
        lower_display_value,
        color=INTERVAL_LINE_COLOR,
        linestyle="-.",
        linewidth=1.3,
        label="5th percentile simulation",
    )
    ax.axvline(
        upper_display_value,
        color=INTERVAL_LINE_COLOR,
        linestyle=(0, (4, 2)),
        linewidth=1.3,
        label="95th percentile simulation",
    )

    x_axis_label = "Final Cumulative Return (decimal, where 1.0 = 100%)"
    if use_log1p_display:
        x_axis_label = (
            "Final Cumulative Return (decimal, where 1.0 = 100%; displayed on log1p spacing)"
        )

    apply_clean_style(
        ax,
        title=f"{ticker}: Monte Carlo Distribution of Final Returns, {format_agent_name(agent_name)} Strategy",
        x_label=x_axis_label,
        y_label="Frequency",
        show_y_grid=True,
        add_legend=True,
        legend_location="upper right",
    )

    if use_log1p_display:
        set_log1p_actual_value_ticks(ax, display_min=display_min, display_max=display_max)
    else:
        apply_axis_number_format(ax, axis="x")

    display_span = max(display_right - display_left, 1e-9)
    ax.set_xlim(
        display_left - display_span * 0.04,
        display_right + display_span * 0.04,
    )

    ax.text(
        0.02,
        0.98,
        (
            f"5th pct: {lower_5pct:.3f}\n"
            f"95th pct: {upper_95pct:.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        color="#1F2937",
        bbox={
            "facecolor": "white",
            "edgecolor": "#9CA3AF",
            "linewidth": 0.8,
            "boxstyle": "square,pad=0.25",
            "alpha": 0.92,
        },
        zorder=5,
    )

    add_figure_caption(
        fig,
        create_caption_text(
            summary_row=summary_row,
            clipping_used=clipping_used,
            clip_lower_q=clip_lower_q,
            clip_upper_q=clip_upper_q,
            use_log1p_display=use_log1p_display,
        ),
    )

    save_chart(fig, output_filename)
    show_chart()


def main() -> None:
    """Create one Monte Carlo histogram per strategy."""
    for agent_name in AGENT_ORDER:
        create_monte_carlo_chart(agent_name)


if __name__ == "__main__":
    main()
