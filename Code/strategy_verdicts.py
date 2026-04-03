"""Clean skill-vs-luck classification using the current ticker's fresh outputs.

The classifier now does one thing only:
it combines the three requested inferential metrics from the current run

    - RCSI_z
    - p-value
    - percentile

and maps them into the final labels:

    Strong Skill
    Moderate Skill
    Weak Skill
    Random / Luck
    Negative Skill
    Suspicious

Repeated-run robustness is still loaded when present, but it is no longer used
to override or secretly demote the main classification. That makes the result
much easier to audit and keeps the user-facing verdict tied directly to the
current run's actual inferential evidence.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_utils import data_clean_dir

try:
    from monte_carlo import (
        TRANSACTION_COST,
        adjust_trade_returns,
        calculate_actual_percentile,
        calculate_cumulative_return_from_log_returns,
        calculate_p_value,
        convert_to_log_returns,
        load_trade_data,
    )
except ModuleNotFoundError:
    from Code.monte_carlo import (
        TRANSACTION_COST,
        adjust_trade_returns,
        calculate_actual_percentile,
        calculate_cumulative_return_from_log_returns,
        calculate_p_value,
        convert_to_log_returns,
        load_trade_data,
    )

try:
    from artifact_provenance import artifact_run_id
    from strategy_config import AGENT_ORDER
except ModuleNotFoundError:
    from Code.artifact_provenance import artifact_run_id
    from Code.strategy_config import AGENT_ORDER


ticker = os.environ.get("TICKER", "SPY").strip().upper()
MULTIPLE_TESTING_GUARD_ENABLED = os.environ.get("MULTIPLE_TESTING_GUARD", "1") == "1"
MIN_RESEARCH_GRADE_SIMULATION_COUNT = int(
    os.environ.get("MIN_RESEARCH_GRADE_SIMULATIONS", "5000")
)
MIN_RELIABLE_OUTER_RUNS = int(os.environ.get("MIN_RELIABLE_OUTER_RUNS", "30"))
MIN_RELIABLE_SIMULATIONS_PER_RUN = int(
    os.environ.get("MIN_RELIABLE_SIMULATIONS_PER_RUN", "1000")
)
STRICT_MONTE_CARLO_ALIGNMENT = os.environ.get("STRICT_MONTE_CARLO_ALIGNMENT", "1") == "1"
METRIC_ALIGNMENT_RTOL = 1e-9
METRIC_ALIGNMENT_ATOL = 1e-12
STRONG_SKILL_MIN_RCSI_Z = 2.0
STRONG_SKILL_MAX_P_VALUE = 0.05
STRONG_SKILL_MIN_PERCENTILE = 95.0
MODERATE_SKILL_MIN_RCSI_Z = 1.0
MODERATE_SKILL_MAX_P_VALUE = 0.05
MODERATE_SKILL_MIN_PERCENTILE = 85.0
WEAK_SKILL_MIN_RCSI_Z = 0.5
WEAK_SKILL_MAX_P_VALUE = 0.20
WEAK_SKILL_MIN_PERCENTILE = 70.0
NEGATIVE_SKILL_MAX_RCSI_Z = -0.5
NEGATIVE_SKILL_MIN_P_VALUE = 0.50
NEGATIVE_SKILL_MAX_PERCENTILE = 30.0
RANDOM_LUCK_MIN_RCSI_Z = -0.5
RANDOM_LUCK_MAX_RCSI_Z = 0.5
RANDOM_LUCK_MIN_P_VALUE = 0.30
RANDOM_LUCK_MAX_P_VALUE = 0.70
RANDOM_LUCK_MIN_PERCENTILE = 30.0
RANDOM_LUCK_MAX_PERCENTILE = 70.0


EVIDENCE_DISPLAY_NAMES = {
    "no_trades": "No Trades",
    "strong_skill": "Strong Skill",
    "skill": "Skill",
    "positive_inconclusive": "Positive but Inconclusive",
    "neutral_inconclusive": "Null-Like / Inconclusive",
    "mixed_inconclusive": "Mixed Evidence",
    "likely_random": "Likely Random",
    "strongly_random": "Strongly Random",
    "moderate_skill": "Moderate Skill",
    "weak_skill": "Weak Skill",
    "random_luck": "Random / Luck",
    "negative_skill": "Negative Skill",
    "suspicious": "Suspicious",
}

EVIDENCE_COLORS = {
    "no_trades": "#6B7280",
    "strong_skill": "#1B5E20",
    "skill": "#2E7D32",
    "positive_inconclusive": "#8A6D1D",
    "neutral_inconclusive": "#7A6F5A",
    "mixed_inconclusive": "#9A6B39",
    "likely_random": "#B23A48",
    "strongly_random": "#8E2A37",
    "moderate_skill": "#388E3C",
    "weak_skill": "#81C784",
    "random_luck": "#9E9E9E",
    "negative_skill": "#C75D5D",
    "suspicious": "#E65100",
}

VERDICT_DISPLAY_NAMES = dict(EVIDENCE_DISPLAY_NAMES)
VERDICT_COLORS = dict(EVIDENCE_COLORS)
VERDICT_DISPLAY_NAMES.update(
    {
        "luck": "Luck",
        "inconclusive": "Inconclusive",
    }
)

CONFIDENCE_DISPLAY_NAMES = {
    "high": "High",
    "moderate": "Moderate",
    "low": "Low",
    "not_applicable": "Not Applicable",
}


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    """Normalize bool-like CSV columns loaded from pandas or plain text."""
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map({"true": True, "false": False, "1": True, "0": False}).fillna(False)


def evidence_label(evidence_bucket: str) -> str:
    """Return the display label for one evidence bucket."""
    return EVIDENCE_DISPLAY_NAMES.get(
        str(evidence_bucket),
        str(evidence_bucket).replace("_", " ").title(),
    )


def compact_evidence_label(evidence_bucket: str) -> str:
    """Return a short chart-friendly version of the classification."""
    mapping = {
        "no_trades": "No\nTrades",
        "strong_skill": "Strong\nSkill",
        "moderate_skill": "Moderate\nSkill",
        "weak_skill": "Weak\nSkill",
        "random_luck": "Random\n/ Luck",
        "negative_skill": "Negative\nSkill",
        "suspicious": "Suspicious",
    }
    return mapping.get(str(evidence_bucket), evidence_label(evidence_bucket))


def confidence_label(confidence_bucket: str) -> str:
    """Return the display label for one confidence bucket."""
    return CONFIDENCE_DISPLAY_NAMES.get(
        str(confidence_bucket),
        str(confidence_bucket).replace("_", " ").title(),
    )


def verdict_from_evidence_bucket(evidence_bucket: str) -> str:
    """Map current and legacy evidence buckets to a display verdict."""
    normalized = str(evidence_bucket).strip().lower()
    simplified_verdicts = {
        "no_trades",
        "strong_skill",
        "moderate_skill",
        "weak_skill",
        "random_luck",
        "negative_skill",
        "suspicious",
    }
    if normalized in simplified_verdicts:
        return normalized
    legacy_mapping = {
        "skill": "skill",
        "positive_inconclusive": "inconclusive",
        "neutral_inconclusive": "inconclusive",
        "mixed_inconclusive": "inconclusive",
        "likely_random": "luck",
        "strongly_random": "luck",
    }
    if normalized in legacy_mapping:
        return legacy_mapping[normalized]
    return "random_luck"


def format_verdict_label(verdict: str) -> str:
    """Return a display-friendly verdict label."""
    return VERDICT_DISPLAY_NAMES.get(str(verdict), str(verdict).replace("_", " ").title())


def classify_confidence(
    p_value: float,
    stability_classification: str | None = None,
) -> str:
    """Assign a lightweight confidence label from p-value and optional stability."""
    if pd.isna(p_value):
        return "not_applicable"

    p = float(p_value)
    stable = str(stability_classification or "not_available").strip().lower() in {
        "stable",
        "not available",
        "not_available",
        "n/a",
    }

    if p <= 0.05 and stable:
        return "high"
    if p <= 0.20:
        return "moderate"
    return "low"


def evidence_bucket_max_p_value(evidence_bucket: str) -> float | None:
    """Return the p-value cutoff tied to each positive-skill label."""
    normalized_bucket = str(evidence_bucket).strip().lower()
    if normalized_bucket == "strong_skill":
        return STRONG_SKILL_MAX_P_VALUE
    if normalized_bucket == "moderate_skill":
        return MODERATE_SKILL_MAX_P_VALUE
    if normalized_bucket == "weak_skill":
        return WEAK_SKILL_MAX_P_VALUE
    return None


def apply_multiple_testing_guard(
    evidence_bucket: str,
    adjusted_p_value: float,
) -> str:
    """Downgrade positive-skill labels when they fail BH-adjusted significance."""
    if not MULTIPLE_TESTING_GUARD_ENABLED or pd.isna(adjusted_p_value):
        return evidence_bucket

    allowed_p_value = evidence_bucket_max_p_value(evidence_bucket)
    if allowed_p_value is None:
        return evidence_bucket
    if float(adjusted_p_value) <= float(allowed_p_value):
        return evidence_bucket
    return "random_luck"


def confidence_bucket_from_score(confidence_score: float) -> str:
    """Translate a numeric confidence score to a coarse text label."""
    if confidence_score >= 0.80:
        return "high"
    if confidence_score >= 0.55:
        return "moderate"
    return "low"


def evaluation_power_score(
    number_of_outer_runs: float,
    simulations_per_run: float,
) -> float:
    """Estimate whether the repeated-run evaluation is deep enough to trust."""
    if pd.isna(number_of_outer_runs) or pd.isna(simulations_per_run):
        return 0.0

    run_component = min(max(float(number_of_outer_runs), 0.0) / MIN_RELIABLE_OUTER_RUNS, 1.0)
    simulation_component = min(
        max(float(simulations_per_run), 0.0) / MIN_RELIABLE_SIMULATIONS_PER_RUN,
        1.0,
    )

    if run_component == 0.0 or simulation_component == 0.0:
        return 0.0

    return float(np.sqrt(run_component * simulation_component))


def evaluation_power_label(power_score: float) -> str:
    """Return a coarse label for the depth of the repeated-run evaluation."""
    if power_score >= 0.85:
        return "Adequate"
    if power_score >= 0.55:
        return "Moderate"
    return "Low"


def adjust_for_evaluation_power(
    evidence_bucket: str,
    confidence_score: float,
    power_score: float,
) -> tuple[str, float]:
    """Downgrade strong claims when the repeated-run evaluation is underpowered."""
    adjusted_bucket = evidence_bucket

    if power_score < 0.75:
        if evidence_bucket in {"strong_skill", "skill"}:
            adjusted_bucket = "positive_inconclusive"
        elif evidence_bucket == "strongly_random":
            adjusted_bucket = "likely_random"

    adjusted_confidence = float(np.clip(confidence_score * max(power_score, 0.10), 0.05, 0.99))
    return adjusted_bucket, adjusted_confidence


def _has_metric_mismatch(rcsi_z: float, p_value: float, percentile: float) -> bool:
    """Return whether core metrics are directionally inconsistent."""
    if rcsi_z >= 1.0 and percentile < 50.0:
        return True
    if rcsi_z <= -1.0 and percentile > 50.0:
        return True
    if p_value <= 0.05 and percentile < 70.0:
        return True
    if p_value > 0.50 and percentile >= 85.0:
        return True
    return False


def _is_suspicious(rcsi_z: float, p_value: float, percentile: float) -> bool:
    """Flag only explicit suspicious cases from the requested rule set."""
    if rcsi_z >= 2.0 and p_value > 0.10:
        return True
    return _has_metric_mismatch(rcsi_z, p_value, percentile)


def classify_metrics_legacy(
    *,
    rcsi_z: float,
    p_value: float,
    percentile: float,
) -> str:
    """Return the pre-recalibration label set for audit comparisons."""
    if any(pd.isna(value) for value in [rcsi_z, p_value, percentile]):
        return "random_luck"

    z = float(rcsi_z)
    p = float(p_value)
    pct = float(percentile)

    if _is_suspicious(z, p, pct):
        return "suspicious"

    if z >= 2.0 and p <= 0.05 and pct >= 95.0:
        return "strong_skill"
    if z >= 1.0 and p <= 0.05 and pct >= 85.0:
        return "moderate_skill"
    if z >= 1.0 and p <= 0.10 and pct >= 80.0:
        return "moderate_skill"
    if z >= 0.5 and p <= 0.20 and pct >= 70.0:
        return "weak_skill"
    if z >= 0.5 and p <= 0.30 and pct >= 65.0:
        return "weak_skill"
    if z < -0.5 and p > 0.50 and pct < 30.0:
        return "negative_skill"
    if z <= -0.5 and pct < 40.0:
        return "negative_skill"
    if -0.5 <= z <= 0.5 and 0.30 <= p <= 0.70 and 30.0 <= pct <= 70.0:
        return "random_luck"

    return "random_luck"


def classify_metrics(
    *,
    rcsi_z: float,
    p_value: float,
    percentile: float,
) -> str:
    """Classify one strategy with the exact requested threshold structure."""
    if any(pd.isna(value) for value in [rcsi_z, p_value, percentile]):
        return "random_luck"

    z = float(rcsi_z)
    p = float(p_value)
    pct = float(percentile)

    if _is_suspicious(z, p, pct):
        return "suspicious"

    # Positive skill tiers require all three inferential metrics.
    if (
        z >= STRONG_SKILL_MIN_RCSI_Z
        and p <= STRONG_SKILL_MAX_P_VALUE
        and pct >= STRONG_SKILL_MIN_PERCENTILE
    ):
        return "strong_skill"
    if (
        z >= MODERATE_SKILL_MIN_RCSI_Z
        and p <= MODERATE_SKILL_MAX_P_VALUE
        and pct >= MODERATE_SKILL_MIN_PERCENTILE
    ):
        return "moderate_skill"

    if (
        z >= WEAK_SKILL_MIN_RCSI_Z
        and p <= WEAK_SKILL_MAX_P_VALUE
        and pct >= WEAK_SKILL_MIN_PERCENTILE
    ):
        return "weak_skill"

    if (
        RANDOM_LUCK_MIN_RCSI_Z <= z <= RANDOM_LUCK_MAX_RCSI_Z
        and RANDOM_LUCK_MIN_P_VALUE <= p <= RANDOM_LUCK_MAX_P_VALUE
    ):
        return "random_luck"

    if (
        z < NEGATIVE_SKILL_MAX_RCSI_Z
        and p > NEGATIVE_SKILL_MIN_P_VALUE
    ):
        return "negative_skill"

    return "random_luck"


def classify_robustness_evidence(
    p_value: float,
    rcsi: float,
    rcsi_z: float,
    percentile: float,
    proportion_significant: float,
    proportion_outperforming_null_median: float,
    stability_classification: str,
) -> tuple[str, float]:
    """Backward-compatible repeated-run evidence classifier for walk-forward."""
    if any(pd.isna(value) for value in [p_value, rcsi, percentile]):
        return "mixed_inconclusive", 0.30

    p_value = float(p_value)
    rcsi = float(rcsi)
    normalized_rcsi = float(rcsi_z) if not pd.isna(rcsi_z) else float(rcsi)
    percentile = float(percentile)
    proportion_significant = float(proportion_significant)
    proportion_outperforming_null_median = float(proportion_outperforming_null_median)
    is_stable = str(stability_classification).strip().lower() == "stable"

    if (
        p_value <= 0.05
        and normalized_rcsi >= 1.25
        and rcsi > 0
        and proportion_significant >= 0.50
        and proportion_outperforming_null_median >= 0.70
        and percentile >= 80
        and is_stable
    ):
        return "strong_skill", 0.92

    if (
        p_value <= 0.10
        and normalized_rcsi >= 0.75
        and rcsi > 0
        and proportion_significant >= 0.20
        and proportion_outperforming_null_median >= 0.55
        and percentile >= 65
    ):
        return "skill", 0.74 if is_stable else 0.60

    if abs(normalized_rcsi) < 0.35 and 0.30 <= p_value <= 0.70 and 35 <= percentile <= 65:
        return "neutral_inconclusive", 0.42 if is_stable else 0.34

    if (
        (normalized_rcsi >= 0.20 or rcsi > 0)
        and (
            p_value <= 0.20
            or percentile >= 55
            or proportion_outperforming_null_median >= 0.50
        )
    ):
        return "positive_inconclusive", 0.46 if is_stable else 0.36

    if (
        p_value >= 0.75
        and normalized_rcsi <= -1.0
        and rcsi < 0
        and proportion_significant == 0
        and proportion_outperforming_null_median <= 0.25
        and percentile <= 30
        and is_stable
    ):
        return "strongly_random", 0.90

    if p_value >= 0.20 and (normalized_rcsi <= -0.35 or percentile < 40):
        return "likely_random", 0.70 if is_stable else 0.52

    return "mixed_inconclusive", 0.36 if is_stable else 0.28


def build_verdict_reason(
    *,
    evidence_bucket: str,
    p_value: float,
    adjusted_p_value: float,
    rcsi_z: float,
    percentile: float,
    stability_classification: str,
    research_grade: bool,
) -> str:
    """Build a short plain-language explanation for one verdict."""
    if evidence_bucket == "no_trades":
        return (
            "No completed trades were generated for this strategy on the current ticker, "
            "so skill-versus-luck classification is not applicable."
        )

    explanation = {
        "strong_skill": "All three metrics point to a strong edge over the null distribution.",
        "moderate_skill": "The result is statistically strong, but not at the highest tier.",
        "weak_skill": "There is some evidence of edge, but it remains modest.",
        "random_luck": "The result is statistically consistent with the null distribution.",
        "negative_skill": "The result is worse than the random null on all three metrics.",
        "suspicious": "The inferential metrics disagree sharply and the result should be reviewed.",
    }.get(evidence_bucket, "The result is consistent with randomness.")

    stability_text = ""
    if str(stability_classification).strip().lower() not in {"", "not available", "not_available", "n/a"}:
        stability_text = f" Robustness stability: {stability_classification}."

    p_text = "n/a" if pd.isna(p_value) else f"{float(p_value):.3f}"
    adjusted_p_text = "n/a" if pd.isna(adjusted_p_value) else f"{float(adjusted_p_value):.3f}"
    z_text = "n/a" if pd.isna(rcsi_z) else f"{float(rcsi_z):.3f}"
    pct_text = "n/a" if pd.isna(percentile) else f"{float(percentile):.1f}"
    depth_text = "" if research_grade else " Monte Carlo depth is below research-grade, so the label is directional but lower-confidence."
    multiple_testing_text = ""
    if pd.notna(adjusted_p_value):
        multiple_testing_text = (
            " BH-adjusted p-value is shown for reference only and is not used"
            " by the final class label."
        )
    return (
        f"{explanation} raw p={p_text}; BH-adjusted p={adjusted_p_text}; "
        f"percentile={pct_text}; RCSI_z={z_text}.{stability_text}{depth_text}{multiple_testing_text}"
    )


def _resolve_current_ticker(current_ticker: str | None) -> str:
    """Normalize the ticker used to load saved outputs."""
    return (current_ticker or ticker).strip().upper()


def _load_required_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    """Load one CSV file and validate its expected columns."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")

    df = pd.read_csv(path)
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(
            f"Missing required columns in {path}: {', '.join(missing_columns)}"
        )
    return df


def _recompute_monte_carlo_metrics(
    *,
    current_ticker: str,
    agent_name: str,
    transaction_cost: float,
) -> dict[str, float | int]:
    """Recompute summary inference metrics from canonical trade/results artifacts."""
    results_path = data_clean_dir() / f"{current_ticker}_{agent_name}_monte_carlo_results.csv"
    trade_path = data_clean_dir() / f"{current_ticker}_{agent_name}_trades.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing Monte Carlo result file for {agent_name}: {results_path}")
    if not trade_path.exists():
        raise FileNotFoundError(f"Missing trade file for {agent_name}: {trade_path}")

    results_df = _load_required_csv(
        results_path,
        ["simulated_cumulative_return"],
    )
    simulated_returns = pd.to_numeric(
        results_df["simulated_cumulative_return"],
        errors="coerce",
    ).dropna()
    if simulated_returns.empty:
        raise ValueError(f"{results_path} has no valid simulated return values.")
    simulated_array = simulated_returns.to_numpy(dtype=float)

    trade_df = load_trade_data(trade_path, allow_empty=True)
    if trade_df.empty:
        actual_cumulative_return = 0.0
        number_of_trades = 0
    else:
        raw_returns = trade_df["return"].to_numpy(dtype=float)
        adjusted_returns = adjust_trade_returns(
            raw_returns=raw_returns,
            transaction_cost=float(transaction_cost),
            input_path=trade_path,
        )
        log_returns = convert_to_log_returns(adjusted_returns)
        actual_cumulative_return = calculate_cumulative_return_from_log_returns(log_returns)
        number_of_trades = int(len(trade_df))

    return {
        "simulation_count": int(len(simulated_array)),
        "median_simulated_return": float(np.median(simulated_array)),
        "mean_simulated_return": float(np.mean(simulated_array)),
        "std_simulated_return": float(np.std(simulated_array, ddof=0)),
        "lower_5pct": float(np.percentile(simulated_array, 5)),
        "upper_95pct": float(np.percentile(simulated_array, 95)),
        "actual_percentile": float(
            calculate_actual_percentile(
                simulated_array,
                float(actual_cumulative_return),
            )
        ),
        "p_value": float(
            calculate_p_value(
                simulated_array,
                float(actual_cumulative_return),
            )
        ),
        "actual_cumulative_return": float(actual_cumulative_return),
        "number_of_trades": int(number_of_trades),
    }


def _align_monte_carlo_row_with_artifacts(
    *,
    current_ticker: str,
    summary_row: pd.Series,
) -> pd.Series:
    """Guarantee that inference metrics match the same underlying run artifacts."""
    agent_name = str(summary_row["agent"]).strip()
    transaction_cost = pd.to_numeric(
        pd.Series([summary_row.get("transaction_cost", TRANSACTION_COST)]),
        errors="coerce",
    ).iloc[0]
    if pd.isna(transaction_cost):
        transaction_cost = float(TRANSACTION_COST)

    expected_values = _recompute_monte_carlo_metrics(
        current_ticker=current_ticker,
        agent_name=agent_name,
        transaction_cost=float(transaction_cost),
    )
    simulation_count = int(expected_values["simulation_count"])
    p_value_resolution = 1.0 / max(simulation_count + 1, 1)
    percentile_resolution = 100.0 / max(simulation_count + 1, 1)
    p_value_tolerance = max(p_value_resolution * 5.0, 2e-3)
    percentile_tolerance = max(percentile_resolution * 5.0, 2e-1)
    corrected_row = summary_row.copy()
    critical_mismatches: list[str] = []
    recomputed_mismatches: list[str] = []

    for field_name, expected_value in expected_values.items():
        if field_name in {"simulation_count", "number_of_trades"}:
            existing_value = pd.to_numeric(
                pd.Series([corrected_row.get(field_name, np.nan)]),
                errors="coerce",
            ).iloc[0]
            matches = (
                pd.notna(existing_value)
                and int(existing_value) == int(expected_value)
            )
        else:
            existing_value = pd.to_numeric(
                pd.Series([corrected_row.get(field_name, np.nan)]),
                errors="coerce",
            ).iloc[0]
            if field_name == "p_value":
                matches = (
                    pd.notna(existing_value)
                    and abs(float(existing_value) - float(expected_value)) <= p_value_tolerance
                )
            elif field_name == "actual_percentile":
                matches = (
                    pd.notna(existing_value)
                    and abs(float(existing_value) - float(expected_value)) <= percentile_tolerance
                )
            else:
                matches = (
                    pd.notna(existing_value)
                    and np.isclose(
                        float(existing_value),
                        float(expected_value),
                        rtol=METRIC_ALIGNMENT_RTOL,
                        atol=METRIC_ALIGNMENT_ATOL,
                    )
                )

        if matches:
            continue

        if field_name in {"actual_cumulative_return", "number_of_trades"}:
            critical_mismatches.append(field_name)
            continue

        corrected_row[field_name] = expected_value
        recomputed_mismatches.append(field_name)

    if critical_mismatches:
        message = (
            f"Monte Carlo summary is stale for {agent_name} on {current_ticker}. "
            f"Critical mismatches: {', '.join(critical_mismatches)}. "
            "Re-run the pipeline so summary, trade, and result files come from the same run."
        )
        if STRICT_MONTE_CARLO_ALIGNMENT:
            raise ValueError(message)
        print(f"Warning: {message}")
        for field_name in critical_mismatches:
            corrected_row[field_name] = expected_values[field_name]

    if recomputed_mismatches:
        print(
            "Warning: recomputed stale Monte Carlo summary fields for "
            f"{agent_name} on {current_ticker}: {', '.join(recomputed_mismatches)}"
        )

    return corrected_row


def _load_core_inference_table(current_ticker: str) -> pd.DataFrame:
    """Merge the single-run Monte Carlo and RCSI tables for classification."""
    monte_carlo_path = data_clean_dir() / f"{current_ticker}_monte_carlo_summary.csv"
    rcsi_path = data_clean_dir() / f"{current_ticker}_rcsi.csv"
    monte_carlo_run_id = artifact_run_id(monte_carlo_path)
    rcsi_run_id = artifact_run_id(rcsi_path)
    if monte_carlo_run_id and rcsi_run_id and monte_carlo_run_id != rcsi_run_id:
        raise ValueError(
            "The Monte Carlo summary and RCSI artifacts come from different runs. "
            f"{monte_carlo_path} -> {monte_carlo_run_id}; {rcsi_path} -> {rcsi_run_id}"
        )

    monte_carlo_df = _load_required_csv(
        monte_carlo_path,
        [
            "agent",
            "p_value",
            "actual_percentile",
            "actual_cumulative_return",
            "number_of_trades",
            "simulation_count",
        ],
    )
    rcsi_df = _load_required_csv(
        rcsi_path,
        ["agent", "RCSI", "RCSI_z"],
    )

    monte_carlo_df["agent"] = monte_carlo_df["agent"].astype(str).str.strip()
    monte_carlo_df = monte_carlo_df.loc[
        monte_carlo_df["agent"].isin(AGENT_ORDER)
    ].copy()
    monte_carlo_df = pd.DataFrame(
        [
            _align_monte_carlo_row_with_artifacts(
                current_ticker=current_ticker,
                summary_row=row,
            )
            for _, row in monte_carlo_df.iterrows()
        ]
    )
    monte_carlo_df["agent"] = monte_carlo_df["agent"].astype(str).str.strip()
    rcsi_df["agent"] = rcsi_df["agent"].astype(str).str.strip()
    rcsi_df = rcsi_df.loc[
        rcsi_df["agent"].isin(AGENT_ORDER)
    ].copy()

    merged_df = monte_carlo_df.merge(
        rcsi_df[["agent", "RCSI", "RCSI_z"]],
        on="agent",
        how="inner",
    )
    merged_df["reference_p_value"] = pd.to_numeric(merged_df["p_value"], errors="coerce")
    merged_df["reference_adjusted_p_value"] = pd.to_numeric(
        merged_df.get("bh_adjusted_p_value", pd.Series(np.nan, index=merged_df.index)),
        errors="coerce",
    )
    merged_df["reference_percentile"] = pd.to_numeric(
        merged_df["actual_percentile"],
        errors="coerce",
    )
    merged_df["reference_rcsi"] = pd.to_numeric(merged_df["RCSI"], errors="coerce")
    merged_df["reference_rcsi_z"] = pd.to_numeric(merged_df["RCSI_z"], errors="coerce")
    merged_df["number_of_trades"] = pd.to_numeric(
        merged_df["number_of_trades"],
        errors="coerce",
    )
    merged_df["simulations_per_run"] = pd.to_numeric(
        merged_df["simulation_count"],
        errors="coerce",
    )
    merged_df["research_grade"] = (
        _coerce_bool_series(
            merged_df.get(
                "research_grade",
                merged_df["simulations_per_run"].fillna(0).ge(MIN_RESEARCH_GRADE_SIMULATION_COUNT),
            )
        )
    )
    merged_df["artifact_run_id"] = monte_carlo_run_id or rcsi_run_id or ""

    return merged_df


def _load_optional_robustness_table(
    current_ticker: str,
    *,
    required_run_id: str | None = None,
) -> pd.DataFrame | None:
    """Load the robustness summary when it exists."""
    robustness_path = data_clean_dir() / f"{current_ticker}_monte_carlo_robustness_summary.csv"
    if not robustness_path.exists():
        return None
    robustness_run_id = artifact_run_id(robustness_path)
    if required_run_id and robustness_run_id and robustness_run_id != required_run_id:
        return None
    if required_run_id and not robustness_run_id:
        return None

    df = pd.read_csv(robustness_path)
    if "agent" not in df.columns:
        return None

    df["agent"] = df["agent"].astype(str).str.strip()
    keep_columns = [
        column
        for column in [
            "agent",
            "stability_classification",
            "number_of_outer_runs",
            "simulations_per_run",
            "mean_p_value",
            "mean_percentile",
            "mean_RCSI_z",
        ]
        if column in df.columns
    ]
    output_df = df[keep_columns].copy()
    if "simulations_per_run" in output_df.columns:
        output_df = output_df.rename(
            columns={"simulations_per_run": "robustness_simulations_per_run"}
        )
    return output_df


def _classify_inference_row(row: pd.Series) -> str:
    """Classify one row from the same raw metrics shown in the comparison table."""
    trade_count = pd.to_numeric(pd.Series([row["number_of_trades"]]), errors="coerce").fillna(0).iloc[0]
    if float(trade_count) <= 0:
        return "no_trades"

    return classify_metrics(
        rcsi_z=row["reference_rcsi_z"],
        p_value=row["reference_p_value"],
        percentile=row["reference_percentile"],
    )


def load_strategy_verdicts(current_ticker: str | None = None) -> pd.DataFrame:
    """Load one consistent, classification-ready verdict table."""
    current_ticker = _resolve_current_ticker(current_ticker)
    verdict_df = _load_core_inference_table(current_ticker)
    robustness_df = _load_optional_robustness_table(
        current_ticker,
        required_run_id=str(verdict_df["artifact_run_id"].iloc[0]).strip() or None,
    )

    if robustness_df is not None:
        verdict_df = verdict_df.merge(robustness_df, on="agent", how="left")
    else:
        verdict_df["stability_classification"] = "not available"
        verdict_df["number_of_outer_runs"] = np.nan
        verdict_df["robustness_simulations_per_run"] = np.nan

    if "robustness_simulations_per_run" not in verdict_df.columns:
        verdict_df["robustness_simulations_per_run"] = np.nan

    verdict_df["stability_classification"] = (
        verdict_df["stability_classification"]
        .fillna("not available")
        .astype(str)
    )

    verdict_df["evidence_bucket"] = verdict_df.apply(
        _classify_inference_row,
        axis=1,
    )

    verdict_df["skill_luck_verdict"] = verdict_df["evidence_bucket"].apply(
        verdict_from_evidence_bucket
    )
    verdict_df["verdict_label"] = verdict_df["skill_luck_verdict"].apply(format_verdict_label)
    verdict_df["evidence_label"] = verdict_df["evidence_bucket"].apply(evidence_label)
    verdict_df["compact_evidence_label"] = verdict_df["evidence_bucket"].apply(
        compact_evidence_label
    )
    verdict_df["confidence_bucket"] = verdict_df.apply(
        lambda row: (
            "not_applicable"
            if row["evidence_bucket"] == "no_trades"
            else classify_confidence(
                p_value=row["reference_p_value"],
                stability_classification=row["stability_classification"],
            )
        ),
        axis=1,
    )
    verdict_df["confidence_label"] = verdict_df["confidence_bucket"].apply(confidence_label)
    verdict_df["confidence_score"] = verdict_df["confidence_bucket"].map(
        {"high": 0.85, "moderate": 0.60, "low": 0.35, "not_applicable": np.nan}
    )
    verdict_df["final_classification"] = verdict_df["evidence_label"].str.lower()
    verdict_df["evaluation_power_score"] = verdict_df.apply(
        lambda row: evaluation_power_score(
            number_of_outer_runs=row.get("number_of_outer_runs", np.nan),
            simulations_per_run=row.get("robustness_simulations_per_run", np.nan),
        ),
        axis=1,
    )
    verdict_df["evaluation_power_label"] = verdict_df["evaluation_power_score"].apply(
        evaluation_power_label
    )
    verdict_df["verdict_reason"] = verdict_df.apply(
        lambda row: build_verdict_reason(
            evidence_bucket=row["evidence_bucket"],
            p_value=row["reference_p_value"],
            adjusted_p_value=row.get("reference_adjusted_p_value", np.nan),
            rcsi_z=row["reference_rcsi_z"],
            percentile=row["reference_percentile"],
            stability_classification=row["stability_classification"],
            research_grade=bool(row.get("research_grade", False)),
        ),
        axis=1,
    )
    verdict_df["verdict_source"] = "single_run"

    verdict_df["agent"] = pd.Categorical(
        verdict_df["agent"],
        categories=AGENT_ORDER,
        ordered=True,
    )
    verdict_df = verdict_df.sort_values("agent").reset_index(drop=True)
    verdict_df["agent"] = verdict_df["agent"].astype(str)

    expected_agents = set(AGENT_ORDER)
    available_agents = set(verdict_df["agent"].tolist())
    if available_agents != expected_agents:
        missing_agents = sorted(expected_agents - available_agents)
        if missing_agents:
            filler_rows = []
            for agent_name in missing_agents:
                filler_rows.append(
                    {
                        "agent": agent_name,
                        "skill_luck_verdict": "no_trades",
                        "verdict_label": "No Trades",
                        "evidence_bucket": "no_trades",
                        "evidence_label": "No Trades",
                        "compact_evidence_label": "No\nTrades",
                        "confidence_bucket": "not_applicable",
                        "confidence_label": "Not Applicable",
                        "confidence_score": np.nan,
                        "evaluation_power_score": np.nan,
                        "evaluation_power_label": "Not Used",
                        "verdict_reason": "No completed trades were generated for this strategy on the current ticker, so skill-versus-luck classification is not applicable.",
                        "final_classification": "no trades",
                        "stability_classification": "not available",
                        "reference_p_value": np.nan,
                        "reference_adjusted_p_value": np.nan,
                        "reference_rcsi": np.nan,
                        "reference_rcsi_z": np.nan,
                        "reference_percentile": np.nan,
                        "number_of_outer_runs": np.nan,
                        "simulations_per_run": np.nan,
                        "robustness_simulations_per_run": np.nan,
                        "research_grade": False,
                        "artifact_run_id": "",
                        "verdict_source": "single_run",
                    }
                )
            verdict_df = pd.concat([verdict_df, pd.DataFrame(filler_rows)], ignore_index=True)
            verdict_df["agent"] = pd.Categorical(
                verdict_df["agent"],
                categories=AGENT_ORDER,
                ordered=True,
            )
            verdict_df = verdict_df.sort_values("agent").reset_index(drop=True)
            verdict_df["agent"] = verdict_df["agent"].astype(str)

    return verdict_df[
        [
            "agent",
            "skill_luck_verdict",
            "verdict_label",
            "evidence_bucket",
            "evidence_label",
            "compact_evidence_label",
            "confidence_bucket",
            "confidence_label",
            "confidence_score",
            "evaluation_power_score",
            "evaluation_power_label",
            "verdict_reason",
            "final_classification",
            "stability_classification",
            "reference_p_value",
            "reference_adjusted_p_value",
            "reference_rcsi",
            "reference_rcsi_z",
            "reference_percentile",
            "number_of_outer_runs",
            "simulations_per_run",
            "robustness_simulations_per_run",
            "research_grade",
            "artifact_run_id",
            "verdict_source",
        ]
    ]
