"""Unit tests for the strategy discovery helper module."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

try:
    import strategy_discovery
except ModuleNotFoundError:
    from Code import strategy_discovery


class StrategyDiscoveryTests(unittest.TestCase):
    """Keep discovery helper behavior stable and easy to refactor safely."""

    def test_split_market_dataframe_produces_three_nonempty_windows(self) -> None:
        row_count = 1200
        df = pd.DataFrame(
            {
                "Date": pd.date_range("2010-01-01", periods=row_count, freq="B"),
                "Close": np.linspace(100.0, 200.0, row_count),
            }
        )
        splits = strategy_discovery.split_market_dataframe(df)
        self.assertEqual(set(splits.keys()), {"train", "validation", "holdout"})
        self.assertGreater(len(splits["train"]), 0)
        self.assertGreater(len(splits["validation"]), 0)
        self.assertGreater(len(splits["holdout"]), 0)
        self.assertLess(splits["train"]["Date"].max(), splits["validation"]["Date"].min())
        self.assertLess(splits["validation"]["Date"].max(), splits["holdout"]["Date"].min())

    def test_apply_bh_by_group_keeps_group_shape(self) -> None:
        df = pd.DataFrame(
            {
                "ticker": ["SPY", "SPY", "QQQ", "QQQ"],
                "split": ["train", "train", "train", "train"],
                "p_value": [0.01, 0.10, 0.20, 0.90],
                "classification_raw": ["weak_skill", "random_luck", "random_luck", "random_luck"],
            }
        )
        adjusted = strategy_discovery.apply_bh_by_group(
            df,
            group_columns=["ticker", "split"],
            adjusted_column_name="adj_p",
        )
        self.assertEqual(len(adjusted), len(df))
        self.assertTrue("adj_p" in adjusted.columns)
        self.assertTrue(adjusted["adj_p"].notna().all())

    def test_classify_row_from_metrics_no_trades_short_circuits(self) -> None:
        bucket = strategy_discovery.classify_row_from_metrics(
            trade_count=0,
            rcsi_z=4.0,
            p_value=0.0001,
            percentile=99.0,
        )
        self.assertEqual(bucket, "no_trades")


if __name__ == "__main__":
    unittest.main()
