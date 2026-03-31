"""Shared strategy names, colors, and parameter defaults for the research pipeline."""

from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a beginner-friendly helper."""
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable with a beginner-friendly helper."""
    return float(os.environ.get(name, str(default)))


def _env_text(name: str, default: str) -> str:
    """Read a text environment variable while keeping a readable default path."""
    return os.environ.get(name, default).strip()


# ---------------------------------------------------------------------------
# Strategy ordering
# ---------------------------------------------------------------------------
#
# The order below drives:
# - the comparison table
# - the chart legends
# - the Monte Carlo / RCSI tables
# - the UI summaries
#
# Keep these names stable unless the entire pipeline is updated with them.
AGENT_ORDER = [
    "trend_pullback",
    "breakout_volume_momentum",
    "mean_reversion_vol_filter",
    "volatility_managed_tsmom",
    "momentum_relative_strength",
    "random",
]
BENCHMARK_NAME = "buy_and_hold"
COMPARISON_ORDER = AGENT_ORDER + [BENCHMARK_NAME]

AGENT_DISPLAY_NAMES = {
    "trend_pullback": "Trend + Pullback",
    "breakout_volume_momentum": "Breakout + Volume + Momentum",
    "mean_reversion_vol_filter": "Mean Reversion + Volatility Filter",
    "volatility_managed_tsmom": "Volatility-Managed TSMOM",
    "momentum_relative_strength": "Momentum + Relative Strength",
    "random": "Random Baseline",
    BENCHMARK_NAME: "Buy and Hold",
}

AGENT_SHORT_DISPLAY_NAMES = {
    "trend_pullback": "Trend Pullback",
    "breakout_volume_momentum": "Breakout Vol+Mom",
    "mean_reversion_vol_filter": "Mean Rev Vol Filter",
    "volatility_managed_tsmom": "Vol Managed TSMOM",
    "momentum_relative_strength": "Rel Strength Mom",
    "random": "Random",
    BENCHMARK_NAME: "Buy & Hold",
}

AGENT_COLORS = {
    "trend_pullback": "#355C7D",
    "breakout_volume_momentum": "#C06C2B",
    "mean_reversion_vol_filter": "#4F772D",
    "volatility_managed_tsmom": "#2A9D8F",
    "momentum_relative_strength": "#8C564B",
    "random": "#7A6F9B",
    BENCHMARK_NAME: "#2F2F2F",
}


# ---------------------------------------------------------------------------
# Feature-engineering defaults
# ---------------------------------------------------------------------------
RSI_LOOKBACK_DAYS = _env_int("RSI_LOOKBACK_DAYS", 14)
ATR_LOOKBACK_DAYS = _env_int("ATR_LOOKBACK_DAYS", 14)
BOLLINGER_LOOKBACK_DAYS = _env_int("BOLLINGER_LOOKBACK_DAYS", 20)
BOLLINGER_STD_MULTIPLIER = _env_float("BOLLINGER_STD_MULTIPLIER", 2.0)
VOLUME_LOOKBACK_DAYS = _env_int("VOLUME_LOOKBACK_DAYS", 20)
MACD_FAST_DAYS = _env_int("MACD_FAST_DAYS", 12)
MACD_SLOW_DAYS = _env_int("MACD_SLOW_DAYS", 26)
MACD_SIGNAL_DAYS = _env_int("MACD_SIGNAL_DAYS", 9)


# ---------------------------------------------------------------------------
# Trend + Pullback parameters
# ---------------------------------------------------------------------------
TREND_PULLBACK_PULLBACK_TOLERANCE = _env_float("TREND_PULLBACK_PULLBACK_TOLERANCE", 0.01)
TREND_PULLBACK_RSI_MIN = _env_float("TREND_PULLBACK_RSI_MIN", 40.0)
TREND_PULLBACK_RSI_MAX = _env_float("TREND_PULLBACK_RSI_MAX", 51.0)
TREND_PULLBACK_STOP_LOOKBACK_DAYS = _env_int("TREND_PULLBACK_STOP_LOOKBACK_DAYS", 10)
TREND_PULLBACK_TARGET_LOOKBACK_DAYS = _env_int("TREND_PULLBACK_TARGET_LOOKBACK_DAYS", 20)
TREND_PULLBACK_STOP_ATR_BUFFER = _env_float("TREND_PULLBACK_STOP_ATR_BUFFER", 0.25)


# ---------------------------------------------------------------------------
# Breakout + Volume + Momentum parameters
# ---------------------------------------------------------------------------
BREAKOUT_LOOKBACK_DAYS = _env_int("BREAKOUT_LOOKBACK_DAYS", 20)
BREAKOUT_STOP_LOOKBACK_DAYS = _env_int("BREAKOUT_STOP_LOOKBACK_DAYS", 10)
BREAKOUT_VOLUME_MULTIPLIER = _env_float("BREAKOUT_VOLUME_MULTIPLIER", 1.3)
BREAKOUT_MOMENTUM_LOOKBACK_DAYS = _env_int("BREAKOUT_MOMENTUM_LOOKBACK_DAYS", 20)
BREAKOUT_MOMENTUM_THRESHOLD = _env_float("BREAKOUT_MOMENTUM_THRESHOLD", 0.02)
BREAKOUT_REWARD_TO_RISK = _env_float("BREAKOUT_REWARD_TO_RISK", 1.8)
BREAKOUT_TIME_STOP_BARS = _env_int("BREAKOUT_TIME_STOP_BARS", 18)
BREAKOUT_CLOSE_BUFFER = _env_float("BREAKOUT_CLOSE_BUFFER", 0.0025)
BREAKOUT_STOP_ATR_BUFFER = _env_float("BREAKOUT_STOP_ATR_BUFFER", 0.15)
BREAKOUT_FAILED_BREAKOUT_BUFFER = _env_float("BREAKOUT_FAILED_BREAKOUT_BUFFER", 0.005)
BREAKOUT_NO_VOLUME_RANGE_EXPANSION_RATIO = _env_float(
    "BREAKOUT_NO_VOLUME_RANGE_EXPANSION_RATIO",
    1.2,
)


# ---------------------------------------------------------------------------
# Mean Reversion + Volatility Filter parameters
# ---------------------------------------------------------------------------
MEAN_REVERSION_ATR_FILTER_LOOKBACK_DAYS = _env_int(
    "MEAN_REVERSION_ATR_FILTER_LOOKBACK_DAYS",
    60,
)
MEAN_REVERSION_BB_FILTER_LOOKBACK_DAYS = _env_int(
    "MEAN_REVERSION_BB_FILTER_LOOKBACK_DAYS",
    60,
)
MEAN_REVERSION_STOP_LOOKBACK_DAYS = _env_int("MEAN_REVERSION_STOP_LOOKBACK_DAYS", 5)
MEAN_REVERSION_STOP_ATR_BUFFER = _env_float("MEAN_REVERSION_STOP_ATR_BUFFER", 0.20)
MEAN_REVERSION_RSI_ENTRY_MAX = _env_float("MEAN_REVERSION_RSI_ENTRY_MAX", 37.0)
MEAN_REVERSION_RSI_EXIT_MIN = _env_float("MEAN_REVERSION_RSI_EXIT_MIN", 50.0)
MEAN_REVERSION_ATR_RATIO_MAX = _env_float("MEAN_REVERSION_ATR_RATIO_MAX", 1.25)
MEAN_REVERSION_BB_WIDTH_RATIO_MAX = _env_float("MEAN_REVERSION_BB_WIDTH_RATIO_MAX", 1.25)
MEAN_REVERSION_TREND_DISTANCE_THRESHOLD = _env_float(
    "MEAN_REVERSION_TREND_DISTANCE_THRESHOLD",
    0.09,
)
MEAN_REVERSION_MA_ALIGNMENT_THRESHOLD = _env_float(
    "MEAN_REVERSION_MA_ALIGNMENT_THRESHOLD",
    0.08,
)
MEAN_REVERSION_LOWER_BAND_BUFFER = _env_float("MEAN_REVERSION_LOWER_BAND_BUFFER", 0.015)


# ---------------------------------------------------------------------------
# Volatility-Managed Time-Series Momentum parameters
# ---------------------------------------------------------------------------
VOL_MANAGED_TSMOM_ENTRY_LOOKBACK_DAYS = _env_int(
    "VOL_MANAGED_TSMOM_ENTRY_LOOKBACK_DAYS",
    126,
)
VOL_MANAGED_TSMOM_EXIT_LOOKBACK_DAYS = _env_int(
    "VOL_MANAGED_TSMOM_EXIT_LOOKBACK_DAYS",
    63,
)
VOL_MANAGED_TSMOM_VOL_LOOKBACK_DAYS = _env_int(
    "VOL_MANAGED_TSMOM_VOL_LOOKBACK_DAYS",
    20,
)
VOL_MANAGED_TSMOM_ENTRY_THRESHOLD = _env_float(
    "VOL_MANAGED_TSMOM_ENTRY_THRESHOLD",
    0.0,
)
VOL_MANAGED_TSMOM_EXIT_THRESHOLD = _env_float(
    "VOL_MANAGED_TSMOM_EXIT_THRESHOLD",
    -0.02,
)
VOL_MANAGED_TSMOM_TARGET_ANNUALIZED_VOL = _env_float(
    "VOL_MANAGED_TSMOM_TARGET_ANNUALIZED_VOL",
    0.18,
)
VOL_MANAGED_TSMOM_MIN_CAPITAL_FRACTION = _env_float(
    "VOL_MANAGED_TSMOM_MIN_CAPITAL_FRACTION",
    0.35,
)
VOL_MANAGED_TSMOM_MAX_CAPITAL_FRACTION = _env_float(
    "VOL_MANAGED_TSMOM_MAX_CAPITAL_FRACTION",
    1.0,
)
VOL_MANAGED_TSMOM_STOP_ATR_MULTIPLIER = _env_float(
    "VOL_MANAGED_TSMOM_STOP_ATR_MULTIPLIER",
    3.0,
)


# ---------------------------------------------------------------------------
# Momentum + Relative Strength parameters
# ---------------------------------------------------------------------------
#
# We keep the legacy MOMENTUM_LOOKBACK_DAYS environment variable as an alias so
# the current UI and any old automation do not break. The newer, clearer name is
# RELATIVE_STRENGTH_LOOKBACK_DAYS.
RELATIVE_STRENGTH_LOOKBACK_DAYS = int(
    os.environ.get(
        "RELATIVE_STRENGTH_LOOKBACK_DAYS",
        os.environ.get("MOMENTUM_LOOKBACK_DAYS", "60"),
    )
)
MOMENTUM_LOOKBACK_DAYS = RELATIVE_STRENGTH_LOOKBACK_DAYS
RELATIVE_STRENGTH_TOP_N = _env_int("RELATIVE_STRENGTH_TOP_N", 1)
RELATIVE_STRENGTH_REBALANCE_FREQUENCY = _env_text(
    "RELATIVE_STRENGTH_REBALANCE_FREQUENCY",
    "monthly",
).lower()
RELATIVE_STRENGTH_ABSOLUTE_MOMENTUM_THRESHOLD = _env_float(
    "RELATIVE_STRENGTH_ABSOLUTE_MOMENTUM_THRESHOLD",
    0.0,
)
RELATIVE_STRENGTH_TRAILING_STOP_ATR_MULTIPLIER = _env_float(
    "RELATIVE_STRENGTH_TRAILING_STOP_ATR_MULTIPLIER",
    2.0,
)
# When this environment variable is populated, it acts as an explicit manual
# override of the automatically selected same-asset-class universe.
RELATIVE_STRENGTH_MANUAL_UNIVERSE_TEXT = _env_text(
    "RELATIVE_STRENGTH_UNIVERSE",
    "",
)
RELATIVE_STRENGTH_FORCE_ASSET_CLASS = _env_text(
    "RELATIVE_STRENGTH_FORCE_ASSET_CLASS",
    "",
).lower()


# ---------------------------------------------------------------------------
# Random baseline parameters
# ---------------------------------------------------------------------------
RANDOM_HOLDING_PERIOD_MIN_BARS = _env_int("RANDOM_HOLDING_PERIOD_MIN_BARS", 3)
RANDOM_HOLDING_PERIOD_MAX_BARS = _env_int("RANDOM_HOLDING_PERIOD_MAX_BARS", 15)
RANDOM_ENTRY_PROBABILITY = _env_float("RANDOM_ENTRY_PROBABILITY", 0.04)


# ---------------------------------------------------------------------------
# Shared feature column names
# ---------------------------------------------------------------------------
RELATIVE_STRENGTH_RETURN_COLUMN = f"trailing_return_{RELATIVE_STRENGTH_LOOKBACK_DAYS}"
BREAKOUT_MOMENTUM_RETURN_COLUMN = f"trailing_return_{BREAKOUT_MOMENTUM_LOOKBACK_DAYS}"
VOL_MANAGED_TSMOM_ENTRY_RETURN_COLUMN = f"trailing_return_{VOL_MANAGED_TSMOM_ENTRY_LOOKBACK_DAYS}"
VOL_MANAGED_TSMOM_EXIT_RETURN_COLUMN = f"trailing_return_{VOL_MANAGED_TSMOM_EXIT_LOOKBACK_DAYS}"
VOL_MANAGED_TSMOM_VOL_COLUMN = f"realized_vol_{VOL_MANAGED_TSMOM_VOL_LOOKBACK_DAYS}_ann"
BREAKOUT_HIGH_COLUMN = f"rolling_high_{BREAKOUT_LOOKBACK_DAYS}_prev"
BREAKOUT_LOW_COLUMN = f"rolling_low_{BREAKOUT_STOP_LOOKBACK_DAYS}_prev"
TREND_PULLBACK_STOP_LOW_COLUMN = f"rolling_low_{TREND_PULLBACK_STOP_LOOKBACK_DAYS}_prev"
TREND_PULLBACK_TARGET_HIGH_COLUMN = f"rolling_high_{TREND_PULLBACK_TARGET_LOOKBACK_DAYS}_prev"
MEAN_REVERSION_STOP_LOW_COLUMN = f"rolling_low_{MEAN_REVERSION_STOP_LOOKBACK_DAYS}_prev"


def format_strategy_name(agent_name: str, *, short: bool = False) -> str:
    """Return either the full or short display label for one strategy."""
    mapping = AGENT_SHORT_DISPLAY_NAMES if short else AGENT_DISPLAY_NAMES
    return mapping.get(agent_name, agent_name.replace("_", " ").title())
