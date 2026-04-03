"""Compare trading performance by agent and entry regime."""

import os
from pathlib import Path
from collections.abc import Callable

import pandas as pd
import numpy as np

try:
    from connors_rsi2_pullback_agent import main as create_connors_rsi2_trades
    from donchian_trend_reentry_agent import main as create_donchian_reentry_trades
    from adx_trend_following_agent import main as create_adx_trend_trades
    from breakout_volume_momentum_agent import main as create_breakout_trades
    from mean_reversion_vol_filter_agent import main as create_mean_reversion_trades
    from momentum_relative_strength_agent import main as create_momentum_trades
    from random_agent import main as create_random_trades
    from strategy_config import AGENT_ORDER
    from trend_momentum_verification_agent import main as create_trend_momentum_verification_trades
    from trend_pullback_agent import main as create_trend_trades
    from turn_of_month_seasonality_agent import main as create_turn_of_month_trades
    from uptrend_oversold_reversion_agent import main as create_oversold_reversion_trades
    from volatility_squeeze_breakout_agent import main as create_squeeze_breakout_trades
    from research_metrics import calculate_trade_level_return_ratio
except ModuleNotFoundError:
    from Code.connors_rsi2_pullback_agent import main as create_connors_rsi2_trades
    from Code.donchian_trend_reentry_agent import main as create_donchian_reentry_trades
    from Code.adx_trend_following_agent import main as create_adx_trend_trades
    from Code.breakout_volume_momentum_agent import main as create_breakout_trades
    from Code.mean_reversion_vol_filter_agent import main as create_mean_reversion_trades
    from Code.momentum_relative_strength_agent import main as create_momentum_trades
    from Code.random_agent import main as create_random_trades
    from Code.strategy_config import AGENT_ORDER
    from Code.trend_momentum_verification_agent import main as create_trend_momentum_verification_trades
    from Code.trend_pullback_agent import main as create_trend_trades
    from Code.turn_of_month_seasonality_agent import main as create_turn_of_month_trades
    from Code.uptrend_oversold_reversion_agent import main as create_oversold_reversion_trades
    from Code.volatility_squeeze_breakout_agent import main as create_squeeze_breakout_trades
    from Code.research_metrics import calculate_trade_level_return_ratio

try:
    from timeframe_config import timeframe_output_suffix
except ModuleNotFoundError:
    from Code.timeframe_config import timeframe_output_suffix


# Read the active ticker from the environment, or fall back to SPY.
ticker = os.environ.get("TICKER", "SPY")
REGIME_ORDER = ["calm", "neutral", "stressed"]
REGIME_MIN_TRADES = int(os.environ.get("REGIME_MIN_TRADES", "5"))


def resolve_data_clean_dir(project_root: Path) -> Path:
    """Return the project's clean-data folder, preferring the uppercase path."""
    suffix = timeframe_output_suffix()
    lowercase_dir = project_root / f"data_clean{suffix}"
    uppercase_dir = project_root / f"Data_Clean{suffix}"

    if uppercase_dir.exists():
        return uppercase_dir
    if lowercase_dir.exists():
        return lowercase_dir

    uppercase_dir.mkdir(parents=True, exist_ok=True)
    return uppercase_dir


def load_csv_checked(
    input_path: Path,
    required_columns: list[str],
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Load a CSV file and confirm that it contains the needed columns."""
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    df = pd.read_csv(input_path)

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(
            f"Missing required columns in {input_path}: {', '.join(missing_columns)}"
        )

    if df.empty and not allow_empty:
        raise ValueError(f"The input file is empty: {input_path}")

    return df


def load_trade_file(input_path: Path, agent_name: str, create_trades) -> pd.DataFrame:
    """Load one trade file and label it with the matching agent name."""
    if not input_path.exists():
        create_trades()

    df = load_csv_checked(
        input_path,
        required_columns=["entry_date", "exit_date", "return", "regime_at_entry"],
        allow_empty=True,
    )

    for column in ["entry_date", "exit_date"]:
        df[column] = pd.to_datetime(df[column], errors="coerce")

    df["return"] = pd.to_numeric(df["return"], errors="coerce")
    df["regime_at_entry"] = df["regime_at_entry"].astype(str).str.strip().str.lower()
    df["agent"] = agent_name

    df = df.dropna(subset=["return", "regime_at_entry"]).reset_index(drop=True)

    return df


def empty_group_summary() -> pd.Series:
    """Return the default summary row for an agent-regime cell with no trades."""
    return pd.Series(
        {
            "total_trades": 0,
            "min_trades_required": REGIME_MIN_TRADES,
            "meets_min_trade_threshold": False,
            "average_return": 0.0,
            "median_return": 0.0,
            "win_rate": 0.0,
            "std_return": 0.0,
            "trade_level_return_ratio": 0.0,
            "plot_trade_level_return_ratio": pd.NA,
        }
    )


def summarize_group(group: pd.DataFrame) -> pd.Series:
    """Calculate the requested performance metrics for one agent/regime group."""
    total_trades = int(len(group))
    average_return = float(group["return"].mean()) if total_trades > 0 else 0.0
    median_return = float(group["return"].median()) if total_trades > 0 else 0.0
    win_rate = float((group["return"] > 0).mean()) if total_trades > 0 else 0.0
    std_return = float(group["return"].std(ddof=0)) if total_trades > 1 else 0.0
    trade_level_return_ratio = calculate_trade_level_return_ratio(
        group["return"].to_numpy(dtype=float)
    )
    meets_min_trade_threshold = total_trades >= REGIME_MIN_TRADES
    plot_trade_level_return_ratio = trade_level_return_ratio if meets_min_trade_threshold else pd.NA

    return pd.Series(
        {
            "total_trades": total_trades,
            "min_trades_required": REGIME_MIN_TRADES,
            "meets_min_trade_threshold": meets_min_trade_threshold,
            "average_return": average_return,
            "median_return": median_return,
            "win_rate": win_rate,
            "std_return": std_return,
            "trade_level_return_ratio": trade_level_return_ratio,
            "plot_trade_level_return_ratio": plot_trade_level_return_ratio,
        }
    )


def build_trade_file_map(
    data_clean_dir: Path,
    *,
    current_ticker: str,
) -> dict[str, tuple[Path, Callable[[], None]]]:
    """Return the trade-file and generator mapping for each supported agent."""
    return {
        "trend_pullback": (
            data_clean_dir / f"{current_ticker}_trend_pullback_trades.csv",
            create_trend_trades,
        ),
        "breakout_volume_momentum": (
            data_clean_dir / f"{current_ticker}_breakout_volume_momentum_trades.csv",
            create_breakout_trades,
        ),
        "mean_reversion_vol_filter": (
            data_clean_dir / f"{current_ticker}_mean_reversion_vol_filter_trades.csv",
            create_mean_reversion_trades,
        ),
        "momentum_relative_strength": (
            data_clean_dir / f"{current_ticker}_momentum_relative_strength_trades.csv",
            create_momentum_trades,
        ),
        "trend_momentum_verification": (
            data_clean_dir / f"{current_ticker}_trend_momentum_verification_trades.csv",
            create_trend_momentum_verification_trades,
        ),
        "adx_trend_following": (
            data_clean_dir / f"{current_ticker}_adx_trend_following_trades.csv",
            create_adx_trend_trades,
        ),
        "uptrend_oversold_reversion": (
            data_clean_dir / f"{current_ticker}_uptrend_oversold_reversion_trades.csv",
            create_oversold_reversion_trades,
        ),
        "volatility_squeeze_breakout": (
            data_clean_dir / f"{current_ticker}_volatility_squeeze_breakout_trades.csv",
            create_squeeze_breakout_trades,
        ),
        "connors_rsi2_pullback": (
            data_clean_dir / f"{current_ticker}_connors_rsi2_pullback_trades.csv",
            create_connors_rsi2_trades,
        ),
        "donchian_trend_reentry": (
            data_clean_dir / f"{current_ticker}_donchian_trend_reentry_trades.csv",
            create_donchian_reentry_trades,
        ),
        "turn_of_month_seasonality": (
            data_clean_dir / f"{current_ticker}_turn_of_month_seasonality_trades.csv",
            create_turn_of_month_trades,
        ),
        "random": (
            data_clean_dir / f"{current_ticker}_random_trades.csv",
            create_random_trades,
        ),
    }


def main() -> None:
    """Create the regime analysis table for the active ticker."""
    project_root = Path(__file__).resolve().parents[1]
    data_clean_dir = resolve_data_clean_dir(project_root)

    trade_file_map = build_trade_file_map(data_clean_dir, current_ticker=ticker)
    missing_agents = [agent_name for agent_name in AGENT_ORDER if agent_name not in trade_file_map]
    if missing_agents:
        raise KeyError(
            "regime_analysis trade-file map is missing active agents: "
            + ", ".join(missing_agents)
        )
    output_path = data_clean_dir / f"{ticker}_regime_analysis.csv"

    all_trade_frames = []

    for agent_name in AGENT_ORDER:
        input_path, create_trades = trade_file_map[agent_name]
        all_trade_frames.append(
            load_trade_file(input_path, agent_name, create_trades)
        )

    all_trades_df = pd.concat(all_trade_frames, ignore_index=True)

    if all_trades_df.empty:
        analysis_df = pd.DataFrame(columns=["agent", "regime_at_entry"])
    else:
        grouped = all_trades_df.groupby(["agent", "regime_at_entry"], dropna=False)
        try:
            analysis_df = grouped.apply(summarize_group, include_groups=False).reset_index()
        except TypeError:
            analysis_df = grouped.apply(summarize_group).reset_index()

    full_index = pd.MultiIndex.from_product(
        [AGENT_ORDER, REGIME_ORDER],
        names=["agent", "regime_at_entry"],
    )
    if analysis_df.empty:
        analysis_df = pd.DataFrame(index=full_index).reset_index()
    else:
        analysis_df = analysis_df.set_index(["agent", "regime_at_entry"]).reindex(full_index).reset_index()

    default_values = empty_group_summary()
    for column_name, default_value in default_values.items():
        if column_name not in analysis_df.columns:
            analysis_df[column_name] = default_value
            continue

        if pd.isna(default_value):
            analysis_df[column_name] = analysis_df[column_name].where(
                analysis_df[column_name].notna(),
                pd.NA,
            )
        elif isinstance(default_value, (bool, np.bool_)):
            normalized = analysis_df[column_name].astype(str).str.strip().str.lower()
            mapped = normalized.map({"true": True, "false": False})
            mapped = mapped.where(mapped.notna(), bool(default_value))
            analysis_df[column_name] = mapped.astype(bool)
        elif isinstance(default_value, (int, np.integer)):
            analysis_df[column_name] = pd.to_numeric(
                analysis_df[column_name],
                errors="coerce",
            ).fillna(default_value).astype(int)
        else:
            analysis_df[column_name] = pd.to_numeric(
                analysis_df[column_name],
                errors="coerce",
            ).fillna(default_value)

    analysis_df["agent"] = pd.Categorical(
        analysis_df["agent"],
        categories=AGENT_ORDER,
        ordered=True,
    )
    analysis_df["regime_at_entry"] = pd.Categorical(
        analysis_df["regime_at_entry"],
        categories=REGIME_ORDER,
        ordered=True,
    )
    analysis_df = analysis_df.sort_values(["agent", "regime_at_entry"]).reset_index(
        drop=True
    )

    analysis_df.to_csv(output_path, index=False)
    print(analysis_df.to_string(index=False))


if __name__ == "__main__":
    main()
