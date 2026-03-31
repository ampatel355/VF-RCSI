"""Compatibility wrapper that now runs Breakout + Volume + Momentum."""

from __future__ import annotations

try:
    from breakout_volume_momentum_agent import main
except ModuleNotFoundError:
    from Code.breakout_volume_momentum_agent import main


if __name__ == "__main__":
    main()
