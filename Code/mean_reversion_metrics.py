"""Compatibility wrapper for Mean Reversion + Volatility Filter metrics."""

from __future__ import annotations

try:
    from mean_reversion_vol_filter_metrics import main
except ModuleNotFoundError:
    from Code.mean_reversion_vol_filter_metrics import main


if __name__ == "__main__":
    main()
