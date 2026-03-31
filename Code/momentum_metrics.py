"""Compatibility wrapper for Momentum + Relative Strength metrics."""

from __future__ import annotations

try:
    from momentum_relative_strength_metrics import main
except ModuleNotFoundError:
    from Code.momentum_relative_strength_metrics import main


if __name__ == "__main__":
    main()
