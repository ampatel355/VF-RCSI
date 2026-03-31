"""Shared strategy names, ordering, colors, and parameter defaults."""

from __future__ import annotations

import os


MOMENTUM_LOOKBACK_DAYS = int(os.environ.get("MOMENTUM_LOOKBACK_DAYS", "120"))
BREAKOUT_LOOKBACK_DAYS = int(os.environ.get("BREAKOUT_LOOKBACK_DAYS", "20"))

AGENT_ORDER = [
    "trend",
    "mean_reversion",
    "random",
    "momentum",
    "breakout",
]
BENCHMARK_NAME = "buy_and_hold"
COMPARISON_ORDER = AGENT_ORDER + [BENCHMARK_NAME]

AGENT_DISPLAY_NAMES = {
    "trend": "Trend",
    "mean_reversion": "Mean Reversion",
    "random": "Random",
    "momentum": "Momentum",
    "breakout": "Breakout",
    BENCHMARK_NAME: "Buy and Hold",
}

AGENT_COLORS = {
    "trend": "#4E79A7",
    "mean_reversion": "#F28E2B",
    "random": "#8065A9",
    "momentum": "#59A14F",
    "breakout": "#9C6B4E",
    BENCHMARK_NAME: "#2F2F2F",
}

MOMENTUM_RETURN_COLUMN = f"momentum_return_{MOMENTUM_LOOKBACK_DAYS}"
BREAKOUT_HIGH_COLUMN = f"breakout_high_{BREAKOUT_LOOKBACK_DAYS}"
BREAKOUT_LOW_COLUMN = f"breakout_low_{BREAKOUT_LOOKBACK_DAYS}"
