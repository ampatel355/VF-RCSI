"""Compatibility wrapper for Breakout + Volume + Momentum metrics."""

from __future__ import annotations

try:
    from breakout_volume_momentum_metrics import main
except ModuleNotFoundError:
    from Code.breakout_volume_momentum_metrics import main


if __name__ == "__main__":
    main()
