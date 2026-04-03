"""Audit legacy vs recalibrated classifications against the new Monte Carlo null."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from artifact_provenance import write_dataframe_artifact
    from monte_carlo import resolve_data_clean_dir
    from strategy_verdicts import (
        classify_metrics,
        classify_metrics_legacy,
        load_strategy_verdicts,
    )
except ModuleNotFoundError:
    from Code.artifact_provenance import write_dataframe_artifact
    from Code.monte_carlo import resolve_data_clean_dir
    from Code.strategy_verdicts import (
        classify_metrics,
        classify_metrics_legacy,
        load_strategy_verdicts,
    )


AUDIT_TICKERS = [
    ticker.strip().upper()
    for ticker in os.environ.get("CLASSIFIER_AUDIT_TICKERS", "SPY,QQQ,BTC-USD").split(",")
    if ticker.strip()
]


def build_strategy_audit_rows(tickers: list[str]) -> pd.DataFrame:
    """Compare legacy and recalibrated labels for the saved strategy verdict inputs."""
    rows: list[pd.DataFrame] = []
    for ticker in tickers:
        verdict_df = load_strategy_verdicts(ticker).copy()
        verdict_df["ticker"] = ticker
        verdict_df["legacy_evidence_bucket"] = verdict_df.apply(
            lambda row: (
                "no_trades"
                if str(row["evidence_bucket"]) == "no_trades"
                else classify_metrics_legacy(
                    rcsi_z=row["reference_rcsi_z"],
                    p_value=row["reference_p_value"],
                    percentile=row["reference_percentile"],
                )
            ),
            axis=1,
        )
        verdict_df["recalibrated_evidence_bucket"] = verdict_df["evidence_bucket"]
        rows.append(
            verdict_df[
                [
                    "ticker",
                    "agent",
                    "reference_p_value",
                    "reference_rcsi",
                    "reference_rcsi_z",
                    "reference_percentile",
                    "legacy_evidence_bucket",
                    "recalibrated_evidence_bucket",
                    "verdict_label",
                    "final_classification",
                ]
            ].copy()
        )

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def summarize_bucket_rates(series: pd.Series) -> dict[str, float]:
    """Return compact rate metrics for one set of evidence buckets."""
    normalized = series.astype(str)
    return {
        "random_luck_rate": float(np.mean(normalized == "random_luck")),
        "weak_skill_rate": float(np.mean(normalized == "weak_skill")),
        "moderate_or_strong_skill_rate": float(
            np.mean(normalized.isin(["moderate_skill", "strong_skill"]))
        ),
        "negative_skill_rate": float(np.mean(normalized == "negative_skill")),
        "suspicious_rate": float(np.mean(normalized == "suspicious")),
    }


def build_random_baseline_audit_rows(
    tickers: list[str],
    data_clean_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare legacy and recalibrated labels on repeated random-baseline runs."""
    run_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | int | str]] = []

    for ticker in tickers:
        input_path = data_clean_dir / f"{ticker}_monte_carlo_random_baseline_validation.csv"
        if not input_path.exists():
            continue

        runs_df = pd.read_csv(input_path)
        runs_df["ticker"] = ticker
        runs_df["legacy_evidence_bucket"] = runs_df.apply(
            lambda row: classify_metrics_legacy(
                rcsi_z=row["RCSI_z"],
                p_value=row["p_value"],
                percentile=row["actual_percentile"],
            ),
            axis=1,
        )
        runs_df["recalibrated_evidence_bucket"] = runs_df.apply(
            lambda row: classify_metrics(
                rcsi_z=row["RCSI_z"],
                p_value=row["p_value"],
                percentile=row["actual_percentile"],
            ),
            axis=1,
        )
        run_rows.append(runs_df.copy())

        legacy_summary = summarize_bucket_rates(runs_df["legacy_evidence_bucket"])
        recalibrated_summary = summarize_bucket_rates(runs_df["recalibrated_evidence_bucket"])
        summary_rows.append(
            {
                "ticker": ticker,
                "random_runs": int(len(runs_df)),
                "mean_percentile": float(np.nanmean(runs_df["actual_percentile"].to_numpy(dtype=float))),
                "mean_p_value": float(np.nanmean(runs_df["p_value"].to_numpy(dtype=float))),
                "legacy_random_luck_rate": legacy_summary["random_luck_rate"],
                "legacy_weak_skill_rate": legacy_summary["weak_skill_rate"],
                "legacy_moderate_or_strong_skill_rate": legacy_summary[
                    "moderate_or_strong_skill_rate"
                ],
                "legacy_negative_skill_rate": legacy_summary["negative_skill_rate"],
                "recalibrated_random_luck_rate": recalibrated_summary["random_luck_rate"],
                "recalibrated_weak_skill_rate": recalibrated_summary["weak_skill_rate"],
                "recalibrated_moderate_or_strong_skill_rate": recalibrated_summary[
                    "moderate_or_strong_skill_rate"
                ],
                "recalibrated_negative_skill_rate": recalibrated_summary[
                    "negative_skill_rate"
                ],
            }
        )

    runs_output_df = pd.concat(run_rows, ignore_index=True) if run_rows else pd.DataFrame()
    summary_output_df = pd.DataFrame(summary_rows)
    return runs_output_df, summary_output_df


def main() -> None:
    """Write classifier audit artifacts for the requested tickers."""
    project_root = Path(__file__).resolve().parents[1]
    data_clean_dir = resolve_data_clean_dir(project_root)

    strategy_audit_df = build_strategy_audit_rows(AUDIT_TICKERS)
    random_runs_df, random_summary_df = build_random_baseline_audit_rows(
        AUDIT_TICKERS,
        data_clean_dir,
    )

    strategy_output_path = data_clean_dir / "classifier_calibration_strategy_audit.csv"
    random_runs_output_path = data_clean_dir / "classifier_calibration_random_runs.csv"
    random_summary_output_path = data_clean_dir / "classifier_calibration_random_summary.csv"

    write_dataframe_artifact(
        strategy_audit_df,
        strategy_output_path,
        producer="classifier_calibration_audit.main",
        research_grade=True,
        canonical_policy="auto",
        parameters={"audit_tickers": AUDIT_TICKERS},
    )
    write_dataframe_artifact(
        random_runs_df,
        random_runs_output_path,
        producer="classifier_calibration_audit.main",
        research_grade=True,
        canonical_policy="auto",
        parameters={"audit_tickers": AUDIT_TICKERS},
    )
    write_dataframe_artifact(
        random_summary_df,
        random_summary_output_path,
        producer="classifier_calibration_audit.main",
        research_grade=True,
        canonical_policy="auto",
        parameters={"audit_tickers": AUDIT_TICKERS},
    )

    print("\nStrategy audit:")
    print(strategy_audit_df.to_string(index=False))
    print("\nRandom-baseline summary:")
    print(random_summary_df.to_string(index=False))
    print(
        "\nSaved audit outputs to:\n"
        f"  {strategy_output_path}\n"
        f"  {random_runs_output_path}\n"
        f"  {random_summary_output_path}"
    )


if __name__ == "__main__":
    main()
