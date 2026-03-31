"""Shared execution model used by the strategy agents."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from asset_class_universe import is_crypto_ticker
except ModuleNotFoundError:
    from Code.asset_class_universe import is_crypto_ticker


STARTING_CAPITAL = float(os.environ.get("STARTING_CAPITAL", "100000"))
MAX_CAPITAL_FRACTION = float(os.environ.get("MAX_CAPITAL_FRACTION", "1.0"))
MAX_AVG_DAILY_VOLUME_FRACTION = float(
    os.environ.get("MAX_AVG_DAILY_VOLUME_FRACTION", "0.05")
)
HALF_SPREAD_BPS = float(os.environ.get("HALF_SPREAD_BPS", "0.5"))
MIN_SLIPPAGE_BPS = float(os.environ.get("MIN_SLIPPAGE_BPS", "0.5"))
MAX_SLIPPAGE_BPS = float(os.environ.get("MAX_SLIPPAGE_BPS", "3.0"))
COMMISSION_PER_SHARE = float(os.environ.get("COMMISSION_PER_SHARE", "0.005"))
MIN_COMMISSION_PER_ORDER = float(os.environ.get("MIN_COMMISSION_PER_ORDER", "1.0"))
EXPECTED_COMMISSION_RATE = float(os.environ.get("EXPECTED_COMMISSION_RATE", "0.00002"))
EXECUTION_MODEL_REPRODUCIBLE = os.environ.get("EXECUTION_MODEL_REPRODUCIBLE", "1") == "1"
EXECUTION_MODEL_SEED = int(os.environ.get("EXECUTION_MODEL_SEED", "20260330"))
TRADE_RETURNS_ALREADY_NET = True
EXPECTED_ROUND_TRIP_EXECUTION_COST = (
    (
        2.0 * HALF_SPREAD_BPS
        + MIN_SLIPPAGE_BPS
        + MAX_SLIPPAGE_BPS
    )
    / 10000.0
) + EXPECTED_COMMISSION_RATE
ACTIVE_TICKER = os.environ.get("TICKER", "SPY").strip().upper()

TRADE_LOG_COLUMNS = [
    "signal_date",
    "signal_close",
    "asset_ticker",
    "strategy_name",
    "entry_date",
    "exit_date",
    "entry_reference_open",
    "exit_reference_open",
    "entry_price",
    "exit_price",
    "shares",
    "entry_commission",
    "exit_commission",
    "entry_slippage_bps",
    "exit_slippage_bps",
    "capital_before",
    "capital_after",
    "capital_deployed",
    "capital_deployed_fraction",
    "position_value",
    "position_value_fraction",
    "liquidity_cap_shares",
    "gross_pnl",
    "net_pnl",
    "gross_return",
    "net_position_return",
    "return",
    "holding_bars",
    "holding_period_days",
    "regime_at_entry",
    "stop_loss_used",
    "take_profit_used",
    "exit_reason",
]


@dataclass
class OpenPosition:
    """Container for an open trade entered through the shared execution model."""

    signal_date: pd.Timestamp
    signal_close: float
    entry_date: pd.Timestamp
    entry_reference_open: float
    entry_price: float
    shares: int
    entry_commission: float
    entry_slippage_bps: float
    capital_before: float
    capital_deployed: float
    liquidity_cap_shares: int
    regime_at_entry: str
    entry_index: int
    strategy_name: str
    asset_ticker: str
    stop_loss_used: float | None
    take_profit_used: float | None


def is_forex_ticker(ticker: str | None = None) -> bool:
    """Return whether the active symbol is a Yahoo Finance forex pair."""
    normalized_ticker = (ticker or ACTIVE_TICKER).strip().upper()
    return normalized_ticker.endswith("=X")


def _strategy_seed_component(strategy_name: str) -> int:
    """Build a stable integer component from a strategy name."""
    return sum((index + 1) * ord(character) for index, character in enumerate(strategy_name))


def build_execution_rng(strategy_name: str) -> np.random.Generator:
    """Create a dedicated RNG for one strategy's execution effects."""
    if EXECUTION_MODEL_REPRODUCIBLE:
        seed_sequence = np.random.SeedSequence(
            [EXECUTION_MODEL_SEED, _strategy_seed_component(strategy_name)]
        )
        return np.random.default_rng(seed_sequence)

    return np.random.default_rng()


def commission_for_position(
    shares: int,
    execution_price: float,
    *,
    ticker: str | None = None,
) -> float:
    """Estimate a per-order commission bill for the current asset type."""
    if shares <= 0 or execution_price <= 0:
        return 0.0

    if is_forex_ticker(ticker) or is_crypto_ticker(ticker or ACTIVE_TICKER):
        # Forex and crypto are quoted in units rather than exchange-listed
        # shares. Charging a stock-style per-share commission on low-priced
        # tokens would massively overstate costs and can incorrectly push a
        # long-only account below zero.
        notional_value = float(shares * execution_price)
        return float(max(MIN_COMMISSION_PER_ORDER, notional_value * EXPECTED_COMMISSION_RATE))

    return float(max(MIN_COMMISSION_PER_ORDER, COMMISSION_PER_SHARE * shares))


def draw_execution_slippage_bps(rng: np.random.Generator) -> float:
    """Draw a small adverse slippage shock in basis points."""
    return float(rng.uniform(MIN_SLIPPAGE_BPS, MAX_SLIPPAGE_BPS))


def apply_adverse_fill_price(
    reference_open: float,
    side: str,
    slippage_bps: float,
) -> float:
    """Turn the reference open into a conservative executed fill price."""
    total_bps = HALF_SPREAD_BPS + slippage_bps
    multiplier = 1.0 + (total_bps / 10000.0)

    if side == "buy":
        return float(reference_open * multiplier)
    if side == "sell":
        return float(reference_open / multiplier)

    raise ValueError(f"Unknown execution side: {side}")


def calculate_liquidity_cap_shares(
    avg_volume_20: float,
    *,
    cash_limit: float | None = None,
    entry_price: float | None = None,
    ticker: str | None = None,
) -> int:
    """Convert historical volume into a conservative position cap."""
    if not pd.isna(avg_volume_20) and avg_volume_20 > 0:
        return max(int(np.floor(avg_volume_20 * MAX_AVG_DAILY_VOLUME_FRACTION)), 0)

    if (
        is_forex_ticker(ticker)
        and cash_limit is not None
        and entry_price is not None
        and cash_limit > 0
        and entry_price > 0
    ):
        # Yahoo forex downloads expose price history but usually not exchange volume.
        # Fall back to a cash-funded cap so the strategies can still be evaluated.
        return max(int(np.floor(cash_limit / entry_price)), 0)

    return 0


def calculate_affordable_share_count(
    cash_limit: float,
    entry_price: float,
    liquidity_cap_shares: int,
    *,
    ticker: str | None = None,
) -> int:
    """Find the largest integer share count the account can actually fund."""
    if cash_limit <= 0 or entry_price <= 0 or liquidity_cap_shares <= 0:
        return 0

    shares = min(int(np.floor(cash_limit / entry_price)), liquidity_cap_shares)
    while shares > 0:
        total_entry_cost = (shares * entry_price) + commission_for_position(
            shares,
            entry_price,
            ticker=ticker,
        )
        if total_entry_cost <= cash_limit:
            return shares
        shares -= 1

    return 0


def open_position_from_signal(
    signal_row: pd.Series,
    next_row: pd.Series,
    capital_before: float,
    regime_at_entry: str,
    entry_index: int,
    rng: np.random.Generator,
    strategy_name: str,
    ticker: str | None = None,
    stop_loss_used: float | None = None,
    take_profit_used: float | None = None,
    capital_fraction_override: float | None = None,
) -> OpenPosition | None:
    """Create an open position if the trade is affordable and liquid enough."""
    asset_ticker = (ticker or ACTIVE_TICKER).strip().upper()
    entry_slippage_bps = draw_execution_slippage_bps(rng)
    entry_reference_open = float(next_row.Open)
    entry_price = apply_adverse_fill_price(
        reference_open=entry_reference_open,
        side="buy",
        slippage_bps=entry_slippage_bps,
    )

    effective_capital_fraction = MAX_CAPITAL_FRACTION
    if capital_fraction_override is not None:
        effective_capital_fraction = min(
            MAX_CAPITAL_FRACTION,
            max(float(capital_fraction_override), 0.0),
        )

    cash_limit = float(capital_before * effective_capital_fraction)
    liquidity_cap_shares = calculate_liquidity_cap_shares(
        float(signal_row.avg_volume_20),
        cash_limit=cash_limit,
        entry_price=entry_price,
        ticker=asset_ticker,
    )
    shares = calculate_affordable_share_count(
        cash_limit=cash_limit,
        entry_price=entry_price,
        liquidity_cap_shares=liquidity_cap_shares,
        ticker=asset_ticker,
    )

    if shares <= 0:
        return None

    entry_commission = commission_for_position(shares, entry_price, ticker=asset_ticker)
    position_value = float(shares * entry_price)
    capital_deployed = float(position_value + entry_commission)

    return OpenPosition(
        signal_date=pd.Timestamp(signal_row.Date),
        signal_close=float(signal_row.Close),
        entry_date=pd.Timestamp(next_row.Date),
        entry_reference_open=entry_reference_open,
        entry_price=entry_price,
        shares=shares,
        entry_commission=entry_commission,
        entry_slippage_bps=entry_slippage_bps,
        capital_before=float(capital_before),
        capital_deployed=capital_deployed,
        liquidity_cap_shares=liquidity_cap_shares,
        regime_at_entry=str(regime_at_entry),
        entry_index=entry_index,
        strategy_name=str(strategy_name),
        asset_ticker=asset_ticker,
        stop_loss_used=float(stop_loss_used) if stop_loss_used is not None else None,
        take_profit_used=float(take_profit_used) if take_profit_used is not None else None,
    )


def close_position_from_signal(
    position: OpenPosition,
    next_row: pd.Series,
    exit_index: int,
    rng: np.random.Generator,
    ticker: str | None = None,
    exit_reason: str = "signal_exit",
) -> dict[str, float | int | str | pd.Timestamp]:
    """Close an open position and return a complete trade-log record."""
    asset_ticker = (ticker or position.asset_ticker or ACTIVE_TICKER).strip().upper()
    exit_slippage_bps = draw_execution_slippage_bps(rng)
    exit_reference_open = float(next_row.Open)
    exit_price = apply_adverse_fill_price(
        reference_open=exit_reference_open,
        side="sell",
        slippage_bps=exit_slippage_bps,
    )
    exit_commission = commission_for_position(
        position.shares,
        exit_price,
        ticker=asset_ticker,
    )

    gross_pnl = float(position.shares * (exit_price - position.entry_price))
    net_pnl = float(gross_pnl - position.entry_commission - exit_commission)
    gross_return = float((exit_price - position.entry_price) / position.entry_price)
    position_value = float(position.shares * position.entry_price)
    net_position_return = float(net_pnl / position_value)
    capital_after = float(position.capital_before + net_pnl)
    capital_deployed_fraction = 0.0
    position_value_fraction = 0.0
    if position.capital_before > 0:
        capital_deployed_fraction = float(position.capital_deployed / position.capital_before)
        position_value_fraction = float(position_value / position.capital_before)
    portfolio_return = 0.0
    if position.capital_before > 0:
        portfolio_return = float(net_pnl / position.capital_before)

    return {
        "signal_date": position.signal_date,
        "signal_close": position.signal_close,
        "asset_ticker": asset_ticker,
        "strategy_name": position.strategy_name,
        "entry_date": position.entry_date,
        "exit_date": pd.Timestamp(next_row.Date),
        "entry_reference_open": position.entry_reference_open,
        "exit_reference_open": exit_reference_open,
        "entry_price": position.entry_price,
        "exit_price": exit_price,
        "shares": position.shares,
        "entry_commission": position.entry_commission,
        "exit_commission": exit_commission,
        "entry_slippage_bps": position.entry_slippage_bps,
        "exit_slippage_bps": exit_slippage_bps,
        "capital_before": position.capital_before,
        "capital_after": capital_after,
        "capital_deployed": position.capital_deployed,
        "capital_deployed_fraction": capital_deployed_fraction,
        "position_value": position_value,
        "position_value_fraction": position_value_fraction,
        "liquidity_cap_shares": position.liquidity_cap_shares,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "gross_return": gross_return,
        "net_position_return": net_position_return,
        "return": portfolio_return,
        "holding_bars": int(exit_index - position.entry_index),
        "holding_period_days": int(exit_index - position.entry_index),
        "regime_at_entry": position.regime_at_entry,
        "stop_loss_used": position.stop_loss_used,
        "take_profit_used": position.take_profit_used,
        "exit_reason": str(exit_reason),
    }
