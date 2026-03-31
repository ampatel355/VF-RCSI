"""Compatibility wrapper for Trend + Pullback metrics."""

from __future__ import annotations

try:
    from trend_pullback_metrics import main
except ModuleNotFoundError:
    from Code.trend_pullback_metrics import main


if __name__ == "__main__":
    main()
