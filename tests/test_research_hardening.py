from __future__ import annotations

import sys
from pathlib import Path
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import execution_model
import compare_agents
import monte_carlo
import monte_carlo_plot
import regime_analysis
import strategy_simulator
import strategy_verdicts
from asset_class_universe import resolve_relative_strength_setup


class ResearchHardeningTests(unittest.TestCase):
    def test_empirical_p_value_uses_plus_one_smoothing(self) -> None:
        simulated_returns = pd.Series([0.01, 0.02, 0.03]).to_numpy(dtype=float)
        p_value = monte_carlo.calculate_p_value(simulated_returns, actual_cumulative_return=0.05)
        self.assertAlmostEqual(p_value, 0.25)

    def test_execution_rng_differs_by_ticker(self) -> None:
        spy_rng = execution_model.build_execution_rng("trend_pullback", "SPY")
        qqq_rng = execution_model.build_execution_rng("trend_pullback", "QQQ")
        self.assertNotEqual(
            float(spy_rng.uniform()),
            float(qqq_rng.uniform()),
        )

    def test_random_strategy_terminal_close_is_recorded(self) -> None:
        original_probability = strategy_simulator.RANDOM_ENTRY_PROBABILITY
        strategy_simulator.RANDOM_ENTRY_PROBABILITY = 1.0
        try:
            market_df = pd.DataFrame(
                {
                    "Date": pd.date_range("2025-01-01", periods=6, freq="1D"),
                    "Open": [100, 101, 102, 103, 104, 105],
                    "High": [101, 102, 103, 104, 105, 106],
                    "Low": [99, 100, 101, 102, 103, 104],
                    "Close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
                    "avg_volume_20": [100000] * 6,
                    "regime": ["neutral"] * 6,
                }
            )
            trades_df = strategy_simulator.run_random_strategy(
                market_df,
                decision_seed=7,
                ticker="SPY",
            )
        finally:
            strategy_simulator.RANDOM_ENTRY_PROBABILITY = original_probability

        self.assertEqual(len(trades_df), 1)
        self.assertEqual(str(trades_df.loc[0, "exit_reason"]), "end_of_sample")
        self.assertGreater(int(trades_df.loc[0, "holding_bars"]), 0)

    def test_multiple_testing_guard_downgrades_positive_skill(self) -> None:
        original_guard = strategy_verdicts.MULTIPLE_TESTING_GUARD_ENABLED
        strategy_verdicts.MULTIPLE_TESTING_GUARD_ENABLED = True
        try:
            downgraded = strategy_verdicts.apply_multiple_testing_guard("weak_skill", 0.26)
            self.assertEqual(downgraded, "random_luck")
        finally:
            strategy_verdicts.MULTIPLE_TESTING_GUARD_ENABLED = original_guard

    def test_classifier_uses_requested_threshold_grid(self) -> None:
        self.assertEqual(
            strategy_verdicts.classify_metrics(
                rcsi_z=1.10,
                p_value=0.04,
                percentile=86.0,
            ),
            "moderate_skill",
        )
        self.assertEqual(
            strategy_verdicts.classify_metrics(
                rcsi_z=-0.60,
                p_value=0.70,
                percentile=20.0,
            ),
            "negative_skill",
        )

    def test_classifier_requires_all_three_weak_metrics(self) -> None:
        self.assertEqual(
            strategy_verdicts.classify_metrics(
                rcsi_z=0.60,
                p_value=0.30,
                percentile=80.0,
            ),
            "random_luck",
        )

    def test_non_research_grade_rows_keep_directional_label(self) -> None:
        row = pd.Series(
            {
                "number_of_trades": 25,
                "reference_rcsi_z": -0.75,
                "reference_p_value": 0.80,
                "reference_percentile": 18.0,
                "reference_adjusted_p_value": 0.90,
                "research_grade": False,
            }
        )
        self.assertEqual(
            strategy_verdicts._classify_inference_row(row),
            "negative_skill",
        )

    def test_structure_preserving_schedule_keeps_gap_multiset(self) -> None:
        trade_structure = monte_carlo.TradeStructure(
            entry_indices=pd.Series([2, 7, 11]).to_numpy(dtype="int64"),
            exit_indices=pd.Series([4, 9, 14]).to_numpy(dtype="int64"),
            durations=pd.Series([2, 2, 3]).to_numpy(dtype="int64"),
            position_value_fractions=pd.Series([1.0, 1.0, 1.0]).to_numpy(dtype=float),
            direction_signs=pd.Series([1, 1, 1]).to_numpy(dtype="int8"),
            asset_indices=pd.Series([0, 0, 0]).to_numpy(dtype="int64"),
            transition_gap_floors=pd.Series([1, 1]).to_numpy(dtype="int64"),
            internal_gap_sizes=pd.Series([3, 2]).to_numpy(dtype="int64"),
            external_slack=5,
        )
        rng = monte_carlo.build_random_generator(reproducible=True, seed=17)
        entry_indices, exit_indices = monte_carlo.build_structure_preserving_schedule_batch(
            trade_structure=trade_structure,
            max_open_index=17,
            batch_size=12,
            rng=rng,
        )

        self.assertTrue((exit_indices - entry_indices == trade_structure.durations).all())
        realized_gaps = trade_structure.internal_gap_sizes.tolist()
        for row in range(entry_indices.shape[0]):
            simulated_gaps = (entry_indices[row, 1:] - exit_indices[row, :-1]).tolist()
            self.assertEqual(sorted(simulated_gaps), sorted(realized_gaps))
            leading_gap = int(entry_indices[row, 0])
            trailing_gap = int(17 - exit_indices[row, -1])
            self.assertEqual(leading_gap + trailing_gap, trade_structure.external_slack)

    def test_comparison_research_grade_ignores_benchmark_row(self) -> None:
        comparison_df = pd.DataFrame(
            {
                "agent": ["trend_pullback", "random", "buy_and_hold"],
                "research_grade": [True, True, pd.NA],
            }
        )
        self.assertTrue(compare_agents.comparison_research_grade(comparison_df))

    def test_plot_validation_accepts_smoothed_summary_metrics(self) -> None:
        simulation_df = pd.DataFrame(
            {
                "simulation_id": [1, 2, 3],
                "simulated_cumulative_return": [0.10, 0.20, 0.30],
            }
        )
        actual_cumulative_return = 0.20
        summary_row = pd.Series(
            {
                "agent": "trend_pullback",
                "simulation_count": 3,
                "median_simulated_return": 0.20,
                "mean_simulated_return": 0.20,
                "std_simulated_return": 0.0816496580927726,
                "lower_5pct": 0.11,
                "upper_95pct": 0.29,
                "actual_percentile": monte_carlo.calculate_actual_percentile(
                    simulation_df["simulated_cumulative_return"].to_numpy(dtype=float),
                    actual_cumulative_return,
                ),
                "p_value": monte_carlo.calculate_p_value(
                    simulation_df["simulated_cumulative_return"].to_numpy(dtype=float),
                    actual_cumulative_return,
                ),
                "actual_cumulative_return": actual_cumulative_return,
                "number_of_trades": 4,
            }
        )
        summary_row["lower_5pct"] = float(
            pd.Series(simulation_df["simulated_cumulative_return"]).quantile(0.05)
        )
        summary_row["upper_95pct"] = float(
            pd.Series(simulation_df["simulated_cumulative_return"]).quantile(0.95)
        )

        monte_carlo_plot.validate_summary_against_results(
            summary_row=summary_row,
            simulation_df=simulation_df,
            actual_cumulative_return=actual_cumulative_return,
            number_of_trades=4,
        )

    def test_plot_validation_recomputes_noncritical_stale_fields(self) -> None:
        simulation_df = pd.DataFrame(
            {
                "simulation_id": [1, 2, 3],
                "simulated_cumulative_return": [0.10, 0.20, 0.30],
            }
        )
        actual_cumulative_return = 0.20
        corrected_row = monte_carlo_plot.validate_summary_against_results(
            summary_row=pd.Series(
                {
                    "agent": "trend_pullback",
                    "simulation_count": 3,
                    "median_simulated_return": 0.20,
                    "mean_simulated_return": 0.20,
                    "std_simulated_return": 0.0816496580927726,
                    "lower_5pct": 0.11,
                    "upper_95pct": 0.29,
                    "actual_percentile": 99.0,
                    "p_value": 0.001,
                    "actual_cumulative_return": actual_cumulative_return,
                    "number_of_trades": 4,
                }
            ),
            simulation_df=simulation_df,
            actual_cumulative_return=actual_cumulative_return,
            number_of_trades=4,
        )

        self.assertAlmostEqual(float(corrected_row["actual_percentile"]), 75.0)
        self.assertAlmostEqual(float(corrected_row["p_value"]), 0.75)

    def test_relative_strength_default_excludes_anchor(self) -> None:
        setup = resolve_relative_strength_setup("SPY")
        self.assertNotIn("SPY", setup["universe"])
        self.assertIn("QQQ", setup["universe"])

    def test_context_matching_is_opt_in_by_default(self) -> None:
        self.assertFalse(monte_carlo.CONTEXT_MATCHING_ENABLED)

    def test_regime_trade_map_covers_active_agents(self) -> None:
        trade_file_map = regime_analysis.build_trade_file_map(
            PROJECT_ROOT / "Data_Clean",
            current_ticker="SPY",
        )
        missing = [agent for agent in strategy_verdicts.AGENT_ORDER if agent not in trade_file_map]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
