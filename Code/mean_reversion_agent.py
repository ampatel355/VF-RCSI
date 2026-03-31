"""Compatibility wrapper that now runs Mean Reversion + Volatility Filter."""

from __future__ import annotations

try:
    from mean_reversion_vol_filter_agent import main
except ModuleNotFoundError:
    from Code.mean_reversion_vol_filter_agent import main


if __name__ == "__main__":
    main()
