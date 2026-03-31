"""Compatibility wrapper for the Trend + Pullback Monte Carlo plot."""

from __future__ import annotations

try:
    from monte_carlo_plot import create_monte_carlo_plot
except ModuleNotFoundError:
    from Code.monte_carlo_plot import create_monte_carlo_plot


if __name__ == "__main__":
    create_monte_carlo_plot("trend_pullback")
