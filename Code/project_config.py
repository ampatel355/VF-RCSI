"""Shared project settings used across the full analysis pipeline."""

import os


# Use the ticker from the environment if the master pipeline provides one.
# Otherwise, fall back to SPY as the default.
ticker = os.environ.get("TICKER", "SPY")
