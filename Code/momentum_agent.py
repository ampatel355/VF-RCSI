"""Compatibility wrapper that now runs Momentum + Relative Strength."""

from __future__ import annotations

try:
    from momentum_relative_strength_agent import main
except ModuleNotFoundError:
    from Code.momentum_relative_strength_agent import main


if __name__ == "__main__":
    main()
