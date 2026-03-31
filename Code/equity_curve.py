"""Create one combined equity curve chart for the active strategies."""

import os

from plot_config import (
    AGENT_COLORS,
    DEFAULT_FIGSIZE,
    ZERO_LINE_COLOR,
    add_note_box,
    apply_axis_number_format,
    apply_clean_style,
    charts_dir,
    data_clean_dir,
    format_agent_name,
    load_csv_checked,
    save_chart,
    show_chart,
)
import matplotlib.pyplot as plt
import pandas as pd

try:
    from buy_and_hold import BUY_HOLD_TRANSACTION_COST
    from monte_carlo import load_market_data, load_trade_data as load_trade_log
    from research_metrics import (
        build_buy_and_hold_curve,
        build_daily_strategy_curve,
        summarize_daily_curve,
    )
    from strategy_config import AGENT_ORDER, BENCHMARK_NAME
    from strategy_curve_utils import load_saved_strategy_curve
except ModuleNotFoundError:
    from Code.buy_and_hold import BUY_HOLD_TRANSACTION_COST
    from Code.monte_carlo import load_market_data, load_trade_data as load_trade_log
    from Code.research_metrics import (
        build_buy_and_hold_curve,
        build_daily_strategy_curve,
        summarize_daily_curve,
    )
    from Code.strategy_config import AGENT_ORDER, BENCHMARK_NAME
    from Code.strategy_curve_utils import load_saved_strategy_curve


# Read the active ticker from the environment, or fall back to SPY.
ticker = os.environ.get("TICKER", "SPY")
STRATEGY_ORDER = AGENT_ORDER + [BENCHMARK_NAME]


def build_strategy_curve(df: pd.DataFrame, trade_path) -> pd.DataFrame:
    """Compatibility wrapper for the shared daily-curve builder."""
    market_df = load_market_curve_data()
    return build_daily_strategy_curve(df, market_df)


def load_market_curve_data() -> pd.DataFrame:
    """Load the market close series used to mark strategies to market daily."""
    market_path = data_clean_dir() / f"{ticker}_regimes.csv"
    market_df = load_market_data(market_path)
    close_df = load_csv_checked(market_path, required_columns=["Date", "Close"])
    close_df["Date"] = pd.to_datetime(close_df["Date"], errors="coerce")
    close_df["Close"] = pd.to_numeric(close_df["Close"], errors="coerce")
    close_df = close_df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    return market_df[["Date", "Open"]].merge(close_df, on="Date", how="inner")


def load_all_curves() -> dict[str, pd.DataFrame]:
    """Load and compute one equity curve per strategy."""
    curves = {}
    market_df = load_market_curve_data()

    for agent_name in AGENT_ORDER:
        input_path = data_clean_dir() / f"{ticker}_{agent_name}_trades.csv"
        trade_df = load_trade_log(input_path, allow_empty=True)
        saved_curve_df = load_saved_strategy_curve(ticker, agent_name)
        if saved_curve_df is not None:
            curves[agent_name] = saved_curve_df
        else:
            curves[agent_name] = build_daily_strategy_curve(trade_df, market_df)

    curves[BENCHMARK_NAME] = build_buy_and_hold_curve(
        market_df=market_df[["Date", "Close"]].copy(),
        transaction_cost=BUY_HOLD_TRANSACTION_COST,
    )

    return curves


def build_summary_text(curves: dict[str, pd.DataFrame]) -> str:
    """Build a compact note showing the final result for each strategy."""
    summary_lines = ["Daily curves mark open positions to market using close prices. Buy and Hold is a passive benchmark."]

    for agent_name in STRATEGY_ORDER:
        curve_df = curves[agent_name]
        curve_summary = summarize_daily_curve(curve_df)
        final_return = float(curve_summary["cumulative_return"])
        summary_lines.append(
            f"{format_agent_name(agent_name)}: {final_return:.6f}"
        )

    start_date = min(curve_df["Date"].iloc[0] for curve_df in curves.values())
    end_date = max(curve_df["Date"].iloc[-1] for curve_df in curves.values())
    summary_lines.append(
        f"Date span: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    )

    return "\n".join(summary_lines)


def remove_stale_equity_charts() -> None:
    """Remove legacy per-strategy equity charts for the active ticker."""
    for agent_name in STRATEGY_ORDER:
        stale_chart_path = charts_dir() / f"{ticker}_{agent_name}_equity_curve.png"
        if stale_chart_path.exists():
            stale_chart_path.unlink()


def main() -> None:
    """Create one combined equity curve chart for the active strategies."""
    output_filename = f"{ticker}_equity_curve.png"
    remove_stale_equity_charts()
    curves = load_all_curves()

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    y_min = 0.0
    y_max = 0.0

    for agent_name in STRATEGY_ORDER:
        curve_df = curves[agent_name]
        x_values = curve_df["Date"]
        y_values = curve_df["cumulative_return"]

        ax.plot(
            x_values,
            y_values,
            color=AGENT_COLORS[agent_name],
            linewidth=2.0,
            label=format_agent_name(agent_name),
        )

        ax.scatter(
            x_values.iloc[-1],
            y_values.iloc[-1],
            color=AGENT_COLORS[agent_name],
            s=26,
            zorder=3,
        )

        y_min = min(y_min, float(y_values.min()))
        y_max = max(y_max, float(y_values.max()))

    ax.axhline(0, color=ZERO_LINE_COLOR, linewidth=0.9)

    apply_clean_style(
        ax,
        title=f"{ticker}: Daily Equity Curves by Strategy and Benchmark",
        x_label="Date",
        y_label="Cumulative Return (decimal, where 1.0 = 100%)",
        show_y_grid=True,
        add_legend=True,
        legend_location="upper left",
        legend_ncol=2,
    )
    apply_axis_number_format(ax)

    y_span = max(y_max - y_min, 0.5)
    y_padding = y_span * 0.14
    ax.set_ylim(
        min(y_min - y_padding, -y_padding * 0.25),
        max(y_max + y_padding, y_padding * 0.25),
    )

    add_note_box(
        ax,
        build_summary_text(curves),
        x=0.98,
        y=0.05,
        ha="right",
        va="bottom",
    )

    save_chart(fig, output_filename)
    show_chart()


if __name__ == "__main__":
    main()
