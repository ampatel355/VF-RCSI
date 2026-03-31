"""Shared helpers for translating statistical results into evidence-based verdicts."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from pipeline_utils import data_clean_dir

try:
    from strategy_config import AGENT_ORDER
except ModuleNotFoundError:
    from Code.strategy_config import AGENT_ORDER
ticker = os.environ.get("TICKER", "SPY")
MIN_RELIABLE_OUTER_RUNS = int(os.environ.get("MIN_RELIABLE_OUTER_RUNS", "30"))
MIN_RELIABLE_SIMULATIONS_PER_RUN = int(
    os.environ.get("MIN_RELIABLE_SIMULATIONS_PER_RUN", "1000")
)

VERDICT_DISPLAY_NAMES = {
    "skill": "Skill",
    "luck": "Luck",
    "inconclusive": "Inconclusive",
    "no_trades": "No Trades",
}

VERDICT_COLORS = {
    "skill": "#2E7D32",
    "luck": "#B23A48",
    "inconclusive": "#8A6D1D",
    "no_trades": "#6B7280",
}

EVIDENCE_DISPLAY_NAMES = {
    "no_trades": "No Trades",
    "strong_skill": "Strong Skill",
    "skill": "Skill",
    "positive_inconclusive": "Positive but Inconclusive",
    "neutral_inconclusive": "Null-Like / Inconclusive",
    "mixed_inconclusive": "Mixed Evidence",
    "likely_random": "Likely Random",
    "strongly_random": "Strongly Random",
}

EVIDENCE_COLORS = {
    "no_trades": "#6B7280",
    "strong_skill": "#1F6B2A",
    "skill": "#2E7D32",
    "positive_inconclusive": "#8A6D1D",
    "neutral_inconclusive": "#7A6F5A",
    "mixed_inconclusive": "#9A6B39",
    "likely_random": "#B23A48",
    "strongly_random": "#8E2A37",
}

CONFIDENCE_DISPLAY_NAMES = {
    "high": "High",
    "moderate": "Moderate",
    "low": "Low",
    "not_applicable": "Not Applicable",
}


def verdict_from_evidence_bucket(evidence_bucket: str) -> str:
    """Map a detailed evidence bucket to the high-level verdict family."""
    if evidence_bucket == "no_trades":
        return "no_trades"
    if evidence_bucket in {"strong_skill", "skill"}:
        return "skill"
    if evidence_bucket in {"likely_random", "strongly_random"}:
        return "luck"
    return "inconclusive"


def verdict_from_final_classification(final_classification: str) -> str:
    """Backward-compatible mapping from legacy final classifications to verdicts."""
    normalized = str(final_classification).strip().lower()

    if normalized in {
        "statistically significant and robust",
        "strong skill",
        "skill",
        "single-run positive evidence",
    }:
        return "skill"

    if normalized in {
        "likely random",
        "strongly random",
        "single-run likely random",
    }:
        return "luck"

    if normalized in {
        "no trades",
    }:
        return "no_trades"

    if normalized in {
        "weak / inconsistent",
        "positive but inconclusive",
        "neutral / inconclusive",
        "mixed evidence",
        "single-run inconclusive",
    }:
        return "inconclusive"

    return "inconclusive"


def evidence_label(evidence_bucket: str) -> str:
    """Return a display label for one detailed evidence bucket."""
    return EVIDENCE_DISPLAY_NAMES.get(evidence_bucket, evidence_bucket.replace("_", " ").title())


def confidence_label(confidence_bucket: str) -> str:
    """Return a display label for one confidence bucket."""
    return CONFIDENCE_DISPLAY_NAMES.get(
        confidence_bucket,
        confidence_bucket.replace("_", " ").title(),
    )


def compact_evidence_label(evidence_bucket: str) -> str:
    """Return a shorter chart-friendly version of the detailed evidence label."""
    label = evidence_label(evidence_bucket)
    if label == "No Trades":
        return "No\nTrades"
    if label == "Positive but Inconclusive":
        return "Positive\nInconclusive"
    if label == "Null-Like / Inconclusive":
        return "Null-Like\nInconclusive"
    if label == "Mixed Evidence":
        return "Mixed\nEvidence"
    if label == "Strongly Random":
        return "Strongly\nRandom"
    if label == "Strong Skill":
        return "Strong\nSkill"
    return label


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
    """Estimate whether the repeated-run evaluation is deep enough to trust strongly."""
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


def classify_robustness_evidence(
    p_value: float,
    rcsi: float,
    rcsi_z: float,
    percentile: float,
    proportion_significant: float,
    proportion_outperforming_null_median: float,
    stability_classification: str,
) -> tuple[str, float]:
    """Classify repeated-run evidence using the full robustness statistics."""
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
        confidence_score = 0.74 if is_stable else 0.60
        return "skill", confidence_score

    if abs(normalized_rcsi) < 0.35 and 0.30 <= p_value <= 0.70 and 35 <= percentile <= 65:
        confidence_score = 0.42 if is_stable else 0.34
        return "neutral_inconclusive", confidence_score

    if (
        (normalized_rcsi >= 0.20 or rcsi > 0)
        and (
            p_value <= 0.20
            or percentile >= 55
            or proportion_outperforming_null_median >= 0.50
        )
    ):
        confidence_score = 0.46 if is_stable else 0.36
        return "positive_inconclusive", confidence_score

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
        confidence_score = 0.70 if is_stable else 0.52
        return "likely_random", confidence_score

    return "mixed_inconclusive", 0.36 if is_stable else 0.28


def classify_single_run_evidence(
    p_value: float,
    rcsi: float,
    rcsi_z: float,
    percentile: float,
) -> tuple[str, float]:
    """Fallback classification when only one Monte Carlo run is available."""
    if any(pd.isna(value) for value in [p_value, rcsi, percentile]):
        return "mixed_inconclusive", 0.28

    p_value = float(p_value)
    rcsi = float(rcsi)
    normalized_rcsi = float(rcsi_z) if not pd.isna(rcsi_z) else float(rcsi)
    percentile = float(percentile)

    if p_value <= 0.05 and normalized_rcsi >= 1.25 and rcsi > 0 and percentile >= 85:
        return "strong_skill", 0.82
    if p_value <= 0.10 and normalized_rcsi >= 0.75 and rcsi > 0 and percentile >= 65:
        return "skill", 0.66
    if abs(normalized_rcsi) < 0.35 and 0.30 <= p_value <= 0.70 and 35 <= percentile <= 65:
        return "neutral_inconclusive", 0.32
    if (normalized_rcsi >= 0.20 or rcsi > 0) and (p_value <= 0.20 or percentile >= 55):
        return "positive_inconclusive", 0.40
    if p_value >= 0.75 and normalized_rcsi <= -1.0 and rcsi < 0 and percentile <= 30:
        return "strongly_random", 0.78
    if p_value >= 0.20 and (normalized_rcsi <= -0.35 or percentile < 40):
        return "likely_random", 0.60
    return "mixed_inconclusive", 0.30


def format_verdict_label(verdict: str) -> str:
    """Return a display-friendly verdict label."""
    return VERDICT_DISPLAY_NAMES.get(verdict, verdict.title())


def build_verdict_reason(
    evidence_bucket: str,
    verdict: str,
    confidence_bucket: str,
    p_value: float,
    rcsi_z: float,
    percentile: float,
    final_classification: str,
    stability_classification: str,
    source: str,
) -> str:
    """Create one concise evidence statement for charts and terminal output."""
    if evidence_bucket == "no_trades" or verdict == "no_trades":
        return (
            "No completed trades were generated for this strategy on this ticker, "
            "so skill-versus-luck inference is not applicable."
        )

    p_value_text = "n/a" if pd.isna(p_value) else f"{float(p_value):.3f}"
    percentile_text = "n/a" if pd.isna(percentile) else f"{float(percentile):.1f}"
    rcsi_z_text = "n/a" if pd.isna(rcsi_z) else f"{float(rcsi_z):.3f}"
    evidence_text = evidence_label(evidence_bucket)
    confidence_text = confidence_label(confidence_bucket)

    if source == "robustness":
        if verdict == "skill":
            return (
                f"{evidence_text} ({confidence_text} confidence): repeated-run evidence supports skill; "
                f"stability={stability_classification}; mean p={p_value_text}; "
                f"mean percentile={percentile_text}; mean RCSI_z={rcsi_z_text}."
            )
        if verdict == "luck":
            return (
                f"{evidence_text} ({confidence_text} confidence): results are consistent with randomness; "
                f"stability={stability_classification}; mean p={p_value_text}; "
                f"mean percentile={percentile_text}; mean RCSI_z={rcsi_z_text}."
            )
        return (
            f"{evidence_text} ({confidence_text} confidence): evidence is mixed; "
            f"stability={stability_classification}; mean p={p_value_text}; "
            f"mean percentile={percentile_text}; mean RCSI_z={rcsi_z_text}."
        )

    if verdict == "skill":
        return (
            f"{evidence_text} ({confidence_text} confidence): single-run evidence leans skill; "
            f"p={p_value_text}; percentile={percentile_text}; RCSI_z={rcsi_z_text}."
        )
    if verdict == "luck":
        return (
            f"{evidence_text} ({confidence_text} confidence): single-run evidence is consistent with randomness; "
            f"p={p_value_text}; percentile={percentile_text}; RCSI_z={rcsi_z_text}."
        )
    return (
        f"{evidence_text} ({confidence_text} confidence): single-run evidence is inconclusive; "
        f"p={p_value_text}; percentile={percentile_text}; RCSI_z={rcsi_z_text}."
    )


def _load_robustness_verdicts(current_ticker: str) -> pd.DataFrame | None:
    """Load repeated-run verdicts when the robustness summary is available."""
    input_path = data_clean_dir() / f"{current_ticker}_monte_carlo_robustness_summary.csv"
    if not input_path.exists():
        return None

    df = pd.read_csv(input_path)
    required_columns = [
        "agent",
        "mean_p_value",
        "mean_RCSI",
        "mean_RCSI_z",
        "mean_percentile",
        "proportion_significant",
        "proportion_outperforming_null_median",
        "final_classification",
        "stability_classification",
        "number_of_outer_runs",
        "simulations_per_run",
    ]
    if any(column not in df.columns for column in required_columns):
        return None

    df = df.copy()
    df["agent"] = df["agent"].astype(str).str.strip()
    df["reference_p_value"] = pd.to_numeric(df["mean_p_value"], errors="coerce")
    df["reference_rcsi"] = pd.to_numeric(df["mean_RCSI"], errors="coerce")
    df["reference_rcsi_z"] = pd.to_numeric(df["mean_RCSI_z"], errors="coerce")
    df["reference_percentile"] = pd.to_numeric(df["mean_percentile"], errors="coerce")
    df["proportion_significant"] = pd.to_numeric(df["proportion_significant"], errors="coerce")
    df["proportion_outperforming_null_median"] = pd.to_numeric(
        df["proportion_outperforming_null_median"],
        errors="coerce",
    )
    df["number_of_outer_runs"] = pd.to_numeric(df["number_of_outer_runs"], errors="coerce")
    df["simulations_per_run"] = pd.to_numeric(df["simulations_per_run"], errors="coerce")
    if "mean_number_of_trades" in df.columns:
        df["mean_number_of_trades"] = pd.to_numeric(df["mean_number_of_trades"], errors="coerce")
    else:
        df["mean_number_of_trades"] = np.nan
    evidence_and_confidence = df.apply(
        lambda row: classify_robustness_evidence(
            p_value=row["reference_p_value"],
            rcsi=row["reference_rcsi"],
            rcsi_z=row["reference_rcsi_z"],
            percentile=row["reference_percentile"],
            proportion_significant=row["proportion_significant"],
            proportion_outperforming_null_median=row["proportion_outperforming_null_median"],
            stability_classification=str(row["stability_classification"]),
        ),
        axis=1,
    )
    df["evidence_bucket"] = evidence_and_confidence.map(lambda value: value[0])
    df["confidence_score"] = evidence_and_confidence.map(lambda value: value[1])
    df["evaluation_power_score"] = df.apply(
        lambda row: evaluation_power_score(
            number_of_outer_runs=row["number_of_outer_runs"],
            simulations_per_run=row["simulations_per_run"],
        ),
        axis=1,
    )
    adjusted_evidence = df.apply(
        lambda row: adjust_for_evaluation_power(
            evidence_bucket=row["evidence_bucket"],
            confidence_score=row["confidence_score"],
            power_score=row["evaluation_power_score"],
        ),
        axis=1,
    )
    df["evidence_bucket"] = adjusted_evidence.map(lambda value: value[0])
    df["confidence_score"] = adjusted_evidence.map(lambda value: value[1])
    no_trade_mask = df["mean_number_of_trades"].fillna(np.nan).le(0)
    df.loc[no_trade_mask, "evidence_bucket"] = "no_trades"
    df.loc[no_trade_mask, "confidence_score"] = np.nan
    df["confidence_bucket"] = df["confidence_score"].apply(confidence_bucket_from_score)
    df.loc[no_trade_mask, "confidence_bucket"] = "not_applicable"
    df["evaluation_power_label"] = df["evaluation_power_score"].apply(evaluation_power_label)
    df.loc[no_trade_mask, "evaluation_power_label"] = "Not Applicable"
    df.loc[no_trade_mask, "evaluation_power_score"] = np.nan
    df["skill_luck_verdict"] = df["evidence_bucket"].apply(verdict_from_evidence_bucket)
    df["verdict_label"] = df["skill_luck_verdict"].apply(format_verdict_label)
    df["evidence_label"] = df["evidence_bucket"].apply(evidence_label)
    df["compact_evidence_label"] = df["evidence_bucket"].apply(compact_evidence_label)
    df["confidence_label"] = df["confidence_bucket"].apply(confidence_label)
    df.loc[no_trade_mask, "final_classification"] = "no trades"
    df.loc[no_trade_mask, "reference_p_value"] = np.nan
    df.loc[no_trade_mask, "reference_rcsi"] = np.nan
    df.loc[no_trade_mask, "reference_rcsi_z"] = np.nan
    df.loc[no_trade_mask, "reference_percentile"] = np.nan
    df["verdict_reason"] = df.apply(
        lambda row: build_verdict_reason(
            evidence_bucket=row["evidence_bucket"],
            verdict=row["skill_luck_verdict"],
            confidence_bucket=row["confidence_bucket"],
            p_value=row["reference_p_value"],
            rcsi_z=row["reference_rcsi_z"],
            percentile=row["reference_percentile"],
            final_classification=str(row["final_classification"]),
            stability_classification=str(row["stability_classification"]),
            source="robustness",
        ),
        axis=1,
    )
    df["verdict_source"] = "robustness"
    output_df = df[
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
            "reference_rcsi",
            "reference_rcsi_z",
            "reference_percentile",
            "number_of_outer_runs",
            "simulations_per_run",
            "verdict_source",
        ]
    ]

    if set(output_df["agent"].astype(str)) != set(AGENT_ORDER):
        return None

    return output_df


def _load_single_run_verdicts(current_ticker: str) -> pd.DataFrame:
    """Build a fallback verdict table from the single-run Monte Carlo outputs."""
    monte_carlo_path = data_clean_dir() / f"{current_ticker}_monte_carlo_summary.csv"
    rcsi_path = data_clean_dir() / f"{current_ticker}_rcsi.csv"

    monte_carlo_df = pd.read_csv(monte_carlo_path)
    rcsi_df = pd.read_csv(rcsi_path)

    monte_carlo_df["agent"] = monte_carlo_df["agent"].astype(str).str.strip()
    rcsi_df["agent"] = rcsi_df["agent"].astype(str).str.strip()

    df = monte_carlo_df.merge(rcsi_df[["agent", "RCSI"]], on="agent", how="inner")
    df["reference_p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    df["reference_rcsi"] = pd.to_numeric(df["RCSI"], errors="coerce")
    df["reference_rcsi_z"] = pd.to_numeric(df.get("RCSI_z"), errors="coerce")
    if df["reference_rcsi_z"].isna().all():
        mean_simulated = pd.to_numeric(df.get("mean_simulated_return"), errors="coerce")
        std_simulated = pd.to_numeric(df.get("std_simulated_return"), errors="coerce")
        actual_return = pd.to_numeric(df.get("actual_cumulative_return"), errors="coerce")
        df["reference_rcsi_z"] = (actual_return - mean_simulated) / std_simulated.replace(0, pd.NA)
    df["reference_percentile"] = pd.to_numeric(df["actual_percentile"], errors="coerce")
    evidence_and_confidence = df.apply(
        lambda row: classify_single_run_evidence(
            p_value=row["reference_p_value"],
            rcsi=row["reference_rcsi"],
            rcsi_z=row["reference_rcsi_z"],
            percentile=row["reference_percentile"],
        ),
        axis=1,
    )
    df["evidence_bucket"] = evidence_and_confidence.map(lambda value: value[0])
    df["confidence_score"] = evidence_and_confidence.map(lambda value: value[1])
    if "number_of_trades" in df.columns:
        df["number_of_trades"] = pd.to_numeric(df["number_of_trades"], errors="coerce")
    else:
        df["number_of_trades"] = np.nan
    no_trade_mask = df["number_of_trades"].fillna(np.nan).le(0)
    df.loc[no_trade_mask, "evidence_bucket"] = "no_trades"
    df.loc[no_trade_mask, "confidence_score"] = np.nan
    df["confidence_bucket"] = df["confidence_score"].apply(confidence_bucket_from_score)
    df.loc[no_trade_mask, "confidence_bucket"] = "not_applicable"
    df["skill_luck_verdict"] = df.apply(
        lambda row: verdict_from_evidence_bucket(row["evidence_bucket"]),
        axis=1,
    )
    df["verdict_label"] = df["skill_luck_verdict"].apply(format_verdict_label)
    df["evidence_label"] = df["evidence_bucket"].apply(evidence_label)
    df["compact_evidence_label"] = df["evidence_bucket"].apply(compact_evidence_label)
    df["confidence_label"] = df["confidence_bucket"].apply(confidence_label)
    df["final_classification"] = df["evidence_label"].str.lower()
    df["stability_classification"] = "not available"
    df["evaluation_power_score"] = np.nan
    df["evaluation_power_label"] = "Single run"
    df["number_of_outer_runs"] = np.nan
    df["simulations_per_run"] = pd.to_numeric(df.get("simulation_count"), errors="coerce")
    df.loc[no_trade_mask, "final_classification"] = "no trades"
    df.loc[no_trade_mask, "reference_p_value"] = np.nan
    df.loc[no_trade_mask, "reference_rcsi"] = np.nan
    df.loc[no_trade_mask, "reference_rcsi_z"] = np.nan
    df.loc[no_trade_mask, "reference_percentile"] = np.nan
    df.loc[no_trade_mask, "evaluation_power_label"] = "Not Applicable"
    df["verdict_reason"] = df.apply(
        lambda row: build_verdict_reason(
            evidence_bucket=row["evidence_bucket"],
            verdict=row["skill_luck_verdict"],
            confidence_bucket=row["confidence_bucket"],
            p_value=row["reference_p_value"],
            rcsi_z=row["reference_rcsi_z"],
            percentile=row["reference_percentile"],
            final_classification=str(row["final_classification"]),
            stability_classification=str(row["stability_classification"]),
            source="single_run",
        ),
        axis=1,
    )
    df["verdict_source"] = "single_run"
    return df[
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
            "reference_rcsi",
            "reference_rcsi_z",
            "reference_percentile",
            "number_of_outer_runs",
            "simulations_per_run",
            "verdict_source",
        ]
    ]


def load_strategy_verdicts(current_ticker: str | None = None) -> pd.DataFrame:
    """Load one consistent verdict table for the active ticker."""
    current_ticker = current_ticker or ticker
    verdict_df = _load_robustness_verdicts(current_ticker)
    if verdict_df is None:
        verdict_df = _load_single_run_verdicts(current_ticker)

    verdict_df["agent"] = pd.Categorical(
        verdict_df["agent"],
        categories=AGENT_ORDER,
        ordered=True,
    )
    verdict_df = verdict_df.sort_values("agent").reset_index(drop=True)
    verdict_df["agent"] = verdict_df["agent"].astype(str)
    return verdict_df
