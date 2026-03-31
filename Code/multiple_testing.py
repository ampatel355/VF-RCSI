"""Cross-ticker multiple-testing corrections for strategy p-values."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FULL_COMPARISON_SUFFIX = "_full_comparison.csv"
OUTPUT_FILENAME = "cross_ticker_fdr_adjusted.csv"


def benjamini_hochberg_q_values(p_values: pd.Series) -> pd.Series:
    """Compute Benjamini-Hochberg adjusted q-values."""
    clean_p_values = pd.to_numeric(p_values, errors="coerce")
    valid_mask = clean_p_values.notna()
    valid_p_values = clean_p_values.loc[valid_mask].astype(float)

    if valid_p_values.empty:
        return pd.Series(np.nan, index=p_values.index, dtype=float)

    order = np.argsort(valid_p_values.to_numpy(dtype=float))
    sorted_p_values = valid_p_values.iloc[order].to_numpy(dtype=float)
    test_count = len(sorted_p_values)
    ranks = np.arange(1, test_count + 1, dtype=float)
    adjusted = sorted_p_values * (test_count / ranks)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    q_values = pd.Series(np.nan, index=p_values.index, dtype=float)
    sorted_index = valid_p_values.iloc[order].index
    q_values.loc[sorted_index] = adjusted
    return q_values


def _full_comparison_paths(data_dir: Path) -> list[Path]:
    """Return all saved strategy-comparison files."""
    return sorted(
        path
        for path in data_dir.glob(f"*{FULL_COMPARISON_SUFFIX}")
        if path.is_file()
    )


def _ticker_from_path(input_path: Path) -> str:
    """Extract the ticker symbol from a saved comparison filename."""
    stem = input_path.stem
    if not stem.endswith("_full_comparison"):
        raise ValueError(f"Unexpected comparison filename: {input_path.name}")
    return stem[: -len("_full_comparison")]


def collect_cross_ticker_tests(data_dir: Path) -> pd.DataFrame:
    """Collect the per-strategy p-values used in the cross-ticker FDR correction."""
    rows: list[pd.DataFrame] = []

    for input_path in _full_comparison_paths(data_dir):
        df = pd.read_csv(input_path)
        if "agent" not in df.columns or "p_value" not in df.columns:
            continue

        working_df = df.copy()
        working_df["ticker"] = _ticker_from_path(input_path)
        working_df["agent"] = working_df["agent"].astype(str).str.strip()
        working_df["p_value"] = pd.to_numeric(working_df["p_value"], errors="coerce")
        working_df = working_df.dropna(subset=["agent", "p_value"]).reset_index(drop=True)
        if working_df.empty:
            continue

        rows.append(working_df[["ticker", "agent", "p_value"]])

    if not rows:
        return pd.DataFrame(columns=["ticker", "agent", "p_value", "fdr_q_value"])

    tests_df = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["ticker", "agent"])
    tests_df["fdr_q_value"] = benjamini_hochberg_q_values(tests_df["p_value"])
    tests_df["fdr_significant_0_05"] = tests_df["fdr_q_value"] <= 0.05
    tests_df["fdr_significant_0_10"] = tests_df["fdr_q_value"] <= 0.10
    return tests_df.sort_values(["ticker", "agent"]).reset_index(drop=True)


def apply_cross_ticker_fdr(data_dir: Path) -> pd.DataFrame:
    """Apply FDR-adjusted q-values to all saved full-comparison tables."""
    tests_df = collect_cross_ticker_tests(data_dir)
    output_path = data_dir / OUTPUT_FILENAME
    tests_df.to_csv(output_path, index=False)

    if tests_df.empty:
        return tests_df

    for comparison_path in _full_comparison_paths(data_dir):
        ticker = _ticker_from_path(comparison_path)
        comparison_df = pd.read_csv(comparison_path)
        comparison_df["agent"] = comparison_df["agent"].astype(str).str.strip()

        ticker_tests = tests_df.loc[tests_df["ticker"] == ticker, [
            "agent",
            "fdr_q_value",
            "fdr_significant_0_05",
            "fdr_significant_0_10",
        ]]
        comparison_df = comparison_df.drop(
            columns=["fdr_q_value", "fdr_significant_0_05", "fdr_significant_0_10"],
            errors="ignore",
        )
        comparison_df = comparison_df.merge(ticker_tests, on="agent", how="left")
        comparison_df.to_csv(comparison_path, index=False)

        legacy_path = comparison_path.with_name(comparison_path.name.replace("_full_", "_agent_"))
        if legacy_path.exists():
            comparison_df.to_csv(legacy_path, index=False)

    return tests_df


def main() -> None:
    """Run the cross-ticker multiple-testing correction from the command line."""
    project_root = Path(__file__).resolve().parents[1]
    uppercase_dir = project_root / "Data_Clean"
    lowercase_dir = project_root / "data_clean"
    data_dir = uppercase_dir if uppercase_dir.exists() else lowercase_dir
    results_df = apply_cross_ticker_fdr(data_dir)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
