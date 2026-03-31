"""Compatibility wrapper that now runs the Trend + Pullback strategy."""

from __future__ import annotations

try:
    from trend_pullback_agent import main
except ModuleNotFoundError:
    from Code.trend_pullback_agent import main


if __name__ == "__main__":
    main()
