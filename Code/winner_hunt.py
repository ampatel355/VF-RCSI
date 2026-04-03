"""Automated winner hunt with a strict pre-lockbox -> lockbox workflow.

This script performs a constrained randomized search across research-backed
rule families, picks candidates ONLY on pre-lockbox windows, and then evaluates
the selected candidate once on an untouched lockbox split.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:
    from monte_carlo import (
        TRANSACTION_COST,
        adjust_trade_returns,
        build_random_generator,
        calculate_actual_percentile,
        calculate_p_value,
        simulate_agent_null_cumulative_returns,
    )
    from regimes import build_regime_dataframe_for_ticker
    from strategy_discovery import (
        EntryPlan,
        augment_candidate_features,
        run_custom_strategy,
    )
except ModuleNotFoundError:
    from Code.monte_carlo import (
        TRANSACTION_COST,
        adjust_trade_returns,
        build_random_generator,
        calculate_actual_percentile,
        calculate_p_value,
        simulate_agent_null_cumulative_returns,
    )
    from Code.regimes import build_regime_dataframe_for_ticker
    from Code.strategy_discovery import (
        EntryPlan,
        augment_candidate_features,
        run_custom_strategy,
    )


TICKERS = [
    symbol.strip().upper()
    for symbol in os.environ.get(
        "WINNER_HUNT_TICKERS",
        "SPY,QQQ,BTC-USD,GC=F,CL=F",
    ).split(",")
    if symbol.strip()
]
SEED = int(os.environ.get("WINNER_HUNT_SEED", "20260402"))
SAMPLES_PER_FAMILY = int(os.environ.get("WINNER_HUNT_SAMPLES_PER_FAMILY", "30"))
PRELOCK_SIMULATIONS = int(os.environ.get("WINNER_HUNT_PRELOCK_SIMULATIONS", "90"))
SELECTION_SIMULATIONS = int(os.environ.get("WINNER_HUNT_SELECTION_SIMULATIONS", "150"))
LOCKBOX_SIMULATIONS = int(os.environ.get("WINNER_HUNT_LOCKBOX_SIMULATIONS", "4000"))

TRAIN_FRACTION = float(os.environ.get("WINNER_HUNT_TRAIN_FRACTION", "0.55"))
VALIDATION_FRACTION = float(os.environ.get("WINNER_HUNT_VALIDATION_FRACTION", "0.20"))
SELECTION_FRACTION = float(os.environ.get("WINNER_HUNT_SELECTION_FRACTION", "0.15"))

MIN_ROWS = int(os.environ.get("WINNER_HUNT_MIN_ROWS", "1000"))
MIN_TRAIN_TRADES = int(os.environ.get("WINNER_HUNT_MIN_TRAIN_TRADES", "20"))
MIN_VALIDATION_TRADES = int(os.environ.get("WINNER_HUNT_MIN_VALIDATION_TRADES", "20"))
MIN_SELECTION_TRADES = int(os.environ.get("WINNER_HUNT_MIN_SELECTION_TRADES", "10"))
MIN_LOCKBOX_TRADES = int(os.environ.get("WINNER_HUNT_MIN_LOCKBOX_TRADES", "15"))

WINNER_MIN_Z = float(os.environ.get("WINNER_HUNT_WINNER_MIN_Z", "1.0"))
WINNER_MAX_P = float(os.environ.get("WINNER_HUNT_WINNER_MAX_P", "0.05"))
WINNER_MIN_PERCENTILE = float(os.environ.get("WINNER_HUNT_WINNER_MIN_PERCENTILE", "95.0"))

DATA_CLEAN_DIR = Path(__file__).resolve().parents[1] / "Data_Clean"
OUTPUT_SCAN_PATH = DATA_CLEAN_DIR / "winner_hunt_scan_results.csv"
OUTPUT_LOCKBOX_PATH = DATA_CLEAN_DIR / "winner_hunt_lockbox_results.csv"

REQUIRED_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "avg_volume_20",
    "ema_20",
    "sma_20",
    "sma_50",
    "sma_100",
    "sma_200",
    "rsi_14",
    "adx_14",
    "atr_14",
    "atr_percent_ratio_60",
    "bollinger_lower",
    "bollinger_mid",
    "bollinger_width_ratio_60",
    "macd_line",
    "macd_signal",
    "zscore_20",
    "ma_50_over_ma_200",
    "volume_ratio_20",
    "trailing_return_20",
    "trailing_return_21",
    "trailing_return_63",
    "trailing_return_126",
    "trailing_return_252",
    "rolling_high_20_prev",
    "rolling_high_55_prev",
    "rolling_low_20_prev",
    "rolling_low_10_prev",
    "rolling_low_5_prev",
    "regime",
]


@dataclass(frozen=True)
class EvaluationResult:
    trades: int
    rcsi_z: float
    p_value: float
    percentile: float


RunnerBuilder = Callable[[dict[str, float | int]], Callable[[pd.DataFrame, str], pd.DataFrame]]


def _safe_stop(value: float | int | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def split_into_lockbox_windows(market_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ordered = market_df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    row_count = len(ordered)
    if row_count < MIN_ROWS:
        raise ValueError(f"Need at least {MIN_ROWS} rows for lockbox workflow; found {row_count}.")

    train_end = int(math.floor(row_count * TRAIN_FRACTION))
    validation_end = int(math.floor(row_count * (TRAIN_FRACTION + VALIDATION_FRACTION)))
    selection_end = int(math.floor(row_count * (TRAIN_FRACTION + VALIDATION_FRACTION + SELECTION_FRACTION)))
    if selection_end >= row_count:
        selection_end = row_count - 1

    train_df = ordered.iloc[:train_end].reset_index(drop=True)
    validation_df = ordered.iloc[train_end:validation_end].reset_index(drop=True)
    selection_df = ordered.iloc[validation_end:selection_end].reset_index(drop=True)
    lockbox_df = ordered.iloc[selection_end:].reset_index(drop=True)

    if min(len(train_df), len(validation_df), len(selection_df), len(lockbox_df)) < 150:
        raise ValueError(
            "One or more lockbox windows are too small: "
            f"train={len(train_df)}, validation={len(validation_df)}, "
            f"selection={len(selection_df)}, lockbox={len(lockbox_df)}."
        )

    return {
        "train": train_df,
        "validation": validation_df,
        "selection": selection_df,
        "lockbox": lockbox_df,
    }


def evaluate_runner(
    *,
    runner: Callable[[pd.DataFrame, str], pd.DataFrame],
    ticker: str,
    split_name: str,
    split_df: pd.DataFrame,
    simulation_count: int,
    seed: int,
    agent_name: str,
    run_id: str,
) -> EvaluationResult:
    trades_df = runner(split_df.copy(), ticker)
    trade_count = int(len(trades_df))
    if trade_count == 0:
        return EvaluationResult(trades=0, rcsi_z=float("nan"), p_value=1.0, percentile=50.0)

    adjusted_returns = adjust_trade_returns(
        trades_df["return"].to_numpy(dtype=float),
        TRANSACTION_COST,
        Path(f"{run_id}_{ticker}_{split_name}.csv"),
    )
    actual_cumulative_return = float(np.expm1(np.log1p(adjusted_returns).sum()))
    rng = build_random_generator(reproducible=True, seed=seed)
    simulated_returns, _ = simulate_agent_null_cumulative_returns(
        agent_name=agent_name,
        current_ticker=ticker,
        trade_df=trades_df.copy(),
        market_df=split_df,
        input_path=Path(f"{run_id}_{ticker}_{split_name}.csv"),
        simulation_count=simulation_count,
        rng=rng,
    )
    simulated_array = simulated_returns.to_numpy(dtype=float)
    std_simulated_return = float(np.std(simulated_array, ddof=0))
    mean_simulated_return = float(np.mean(simulated_array))
    rcsi_z = (
        float((actual_cumulative_return - mean_simulated_return) / std_simulated_return)
        if std_simulated_return > 0
        else float("nan")
    )
    p_value = float(calculate_p_value(simulated_array, actual_cumulative_return))
    percentile = float(calculate_actual_percentile(simulated_array, actual_cumulative_return))
    return EvaluationResult(trades=trade_count, rcsi_z=rcsi_z, p_value=p_value, percentile=percentile)


def sample_vol_mom_params(rng: np.random.Generator) -> dict[str, float | int]:
    return {
        "adx_min": int(rng.integers(8, 23)),
        "ret63_min": float(rng.choice([0.0, 0.01, 0.02])),
        "ret126_min": float(rng.choice([0.0, 0.02, 0.04])),
        "stop_mult": float(rng.uniform(2.0, 3.2)),
        "trail_mult": float(rng.uniform(2.6, 4.0)),
        "time_stop": int(rng.integers(70, 141)),
        "vol_cap": float(rng.uniform(1.0, 1.8)),
    }


def sample_pullback_params(rng: np.random.Generator) -> dict[str, float | int]:
    rsi_low = int(rng.integers(35, 46))
    rsi_high = int(max(rsi_low + 12, rng.integers(60, 73)))
    return {
        "rsi_low": rsi_low,
        "rsi_high": rsi_high,
        "adx_min": int(rng.integers(0, 21)),
        "stop_mult": float(rng.uniform(1.4, 2.4)),
        "tp_mult": float(rng.uniform(2.4, 4.8)),
        "time_stop": int(rng.integers(25, 61)),
    }


def sample_breakout_params(rng: np.random.Generator) -> dict[str, float | int]:
    return {
        "lookback": int(rng.choice([20, 55])),
        "adx_min": int(rng.integers(8, 23)),
        "volume_min": float(rng.uniform(0.8, 1.4)),
        "stop_mult": float(rng.uniform(1.6, 2.8)),
        "tp_mult": float(rng.uniform(2.4, 5.0)),
        "time_stop": int(rng.integers(25, 91)),
    }


def sample_mean_reversion_params(rng: np.random.Generator) -> dict[str, float | int]:
    return {
        "zscore_threshold": float(rng.uniform(0.7, 1.6)),
        "rsi_max": int(rng.integers(35, 56)),
        "adx_max": int(rng.integers(15, 36)),
        "atr_ratio_max": float(rng.uniform(1.0, 2.5)),
        "stop_mult": float(rng.uniform(0.8, 1.8)),
        "tp_mult": float(rng.uniform(0.8, 1.8)),
        "time_stop": int(rng.integers(5, 36)),
    }


def build_vol_mom_runner(params: dict[str, float | int]) -> Callable[[pd.DataFrame, str], pd.DataFrame]:
    adx_min = int(params["adx_min"])
    ret63_min = float(params["ret63_min"])
    ret126_min = float(params["ret126_min"])
    stop_mult = float(params["stop_mult"])
    trail_mult = float(params["trail_mult"])
    time_stop = int(params["time_stop"])
    vol_cap = float(params["vol_cap"])

    def runner(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        def entry_rule(row: pd.Series, _prev: pd.Series, _i: int, _df: pd.DataFrame) -> EntryPlan | None:
            if (
                row.trailing_return_126 > ret126_min
                and row.trailing_return_63 > ret63_min
                and row.macd_line > row.macd_signal
                and row.Close > row.sma_200
                and row.adx_14 >= adx_min
            ):
                vol_scale = float(np.clip(vol_cap / max(float(row.atr_percent_ratio_60), 0.25), 0.25, 1.0))
                return EntryPlan(
                    entry_reason="hunt_vol_mom",
                    stop_loss=_safe_stop(row.Close - (stop_mult * row.atr_14)),
                    capital_fraction=vol_scale,
                    state={"highest_close": float(row.Close)},
                )
            return None

        def update_rule(
            row: pd.Series,
            _previous_row: pd.Series,
            _index: int,
            _df: pd.DataFrame,
            state: dict[str, float | int | str],
        ) -> None:
            state["highest_close"] = max(float(state.get("highest_close", row.Close)), float(row.Close))

        def exit_rule(
            row: pd.Series,
            _previous_row: pd.Series,
            current_index: int,
            _df: pd.DataFrame,
            position,
            state: dict[str, float | int | str],
        ) -> str | None:
            trailing_stop = float(state.get("highest_close", row.Close)) - (trail_mult * float(row.atr_14))
            if float(row.Low) <= trailing_stop:
                return "trail"
            if row.trailing_return_21 < 0:
                return "mom_flip"
            if (current_index - int(position.entry_index)) >= time_stop:
                return "time"
            return None

        return run_custom_strategy(
            market_df=market_df,
            ticker=ticker,
            strategy_name="winner_hunt_vol_mom",
            required_columns=REQUIRED_COLUMNS,
            entry_rule=entry_rule,
            exit_rule=exit_rule,
            state_update_rule=update_rule,
        )

    return runner


def build_pullback_runner(params: dict[str, float | int]) -> Callable[[pd.DataFrame, str], pd.DataFrame]:
    rsi_low = int(params["rsi_low"])
    rsi_high = int(params["rsi_high"])
    adx_min = int(params["adx_min"])
    stop_mult = float(params["stop_mult"])
    tp_mult = float(params["tp_mult"])
    time_stop = int(params["time_stop"])

    def runner(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        def entry_rule(row: pd.Series, prev: pd.Series, _i: int, _df: pd.DataFrame) -> EntryPlan | None:
            if (
                row.sma_50 > row.sma_200
                and row.Close > row.sma_200
                and row.Low <= row.ema_20 * 1.01
                and rsi_low <= row.rsi_14 <= rsi_high
                and row.macd_line > row.macd_signal
                and row.adx_14 >= adx_min
                and row.Close >= prev.Close
            ):
                return EntryPlan(
                    entry_reason="hunt_pullback",
                    stop_loss=_safe_stop(min(row.rolling_low_10_prev, row.Close - (stop_mult * row.atr_14))),
                    take_profit=_safe_stop(row.Close + (tp_mult * row.atr_14)),
                )
            return None

        def exit_rule(
            row: pd.Series,
            _previous_row: pd.Series,
            current_index: int,
            _df: pd.DataFrame,
            position,
            _state: dict[str, float | int | str],
        ) -> str | None:
            if row.Close < row.sma_50:
                return "trend_break"
            if row.rsi_14 >= 76:
                return "rsi_exhaustion"
            if (current_index - int(position.entry_index)) >= time_stop:
                return "time"
            return None

        return run_custom_strategy(
            market_df=market_df,
            ticker=ticker,
            strategy_name="winner_hunt_pullback",
            required_columns=REQUIRED_COLUMNS,
            entry_rule=entry_rule,
            exit_rule=exit_rule,
        )

    return runner


def build_breakout_runner(params: dict[str, float | int]) -> Callable[[pd.DataFrame, str], pd.DataFrame]:
    lookback = int(params["lookback"])
    adx_min = int(params["adx_min"])
    volume_min = float(params["volume_min"])
    stop_mult = float(params["stop_mult"])
    tp_mult = float(params["tp_mult"])
    time_stop = int(params["time_stop"])
    breakout_column = "rolling_high_55_prev" if lookback >= 55 else "rolling_high_20_prev"

    def runner(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        def entry_rule(row: pd.Series, _prev: pd.Series, _i: int, _df: pd.DataFrame) -> EntryPlan | None:
            if (
                row.Close > row[breakout_column]
                and row.volume_ratio_20 >= volume_min
                and row.macd_line > row.macd_signal
                and row.adx_14 >= adx_min
            ):
                return EntryPlan(
                    entry_reason="hunt_breakout",
                    stop_loss=_safe_stop(row.Close - (stop_mult * row.atr_14)),
                    take_profit=_safe_stop(row.Close + (tp_mult * row.atr_14)),
                )
            return None

        def exit_rule(
            row: pd.Series,
            _previous_row: pd.Series,
            current_index: int,
            _df: pd.DataFrame,
            position,
            _state: dict[str, float | int | str],
        ) -> str | None:
            if row.Close < row.ema_20:
                return "ema_break"
            if (current_index - int(position.entry_index)) >= time_stop:
                return "time"
            return None

        return run_custom_strategy(
            market_df=market_df,
            ticker=ticker,
            strategy_name="winner_hunt_breakout",
            required_columns=REQUIRED_COLUMNS,
            entry_rule=entry_rule,
            exit_rule=exit_rule,
        )

    return runner


def build_mean_reversion_runner(params: dict[str, float | int]) -> Callable[[pd.DataFrame, str], pd.DataFrame]:
    zscore_threshold = float(params["zscore_threshold"])
    rsi_max = int(params["rsi_max"])
    adx_max = int(params["adx_max"])
    atr_ratio_max = float(params["atr_ratio_max"])
    stop_mult = float(params["stop_mult"])
    tp_mult = float(params["tp_mult"])
    time_stop = int(params["time_stop"])

    def runner(market_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        def entry_rule(row: pd.Series, _prev: pd.Series, _i: int, _df: pd.DataFrame) -> EntryPlan | None:
            if (
                row.zscore_20 <= -zscore_threshold
                and row.rsi_14 <= rsi_max
                and row.adx_14 <= adx_max
                and row.atr_percent_ratio_60 <= atr_ratio_max
            ):
                return EntryPlan(
                    entry_reason="hunt_mean_reversion",
                    stop_loss=_safe_stop(row.Close - (stop_mult * row.atr_14)),
                    take_profit=_safe_stop(row.Close + (tp_mult * row.atr_14)),
                )
            return None

        def exit_rule(
            row: pd.Series,
            _previous_row: pd.Series,
            current_index: int,
            _df: pd.DataFrame,
            position,
            _state: dict[str, float | int | str],
        ) -> str | None:
            if row.zscore_20 >= -0.2 or row.Close >= row.ema_20:
                return "mean_revert_complete"
            if (current_index - int(position.entry_index)) >= time_stop:
                return "time"
            return None

        return run_custom_strategy(
            market_df=market_df,
            ticker=ticker,
            strategy_name="winner_hunt_mean_reversion",
            required_columns=REQUIRED_COLUMNS,
            entry_rule=entry_rule,
            exit_rule=exit_rule,
        )

    return runner


def prelock_gate(train: EvaluationResult, validation: EvaluationResult, selection: EvaluationResult) -> bool:
    return (
        train.trades >= MIN_TRAIN_TRADES
        and validation.trades >= MIN_VALIDATION_TRADES
        and selection.trades >= MIN_SELECTION_TRADES
        and train.rcsi_z >= 0.2
        and validation.rcsi_z >= 0.1
        and selection.rcsi_z >= 0.0
        and train.p_value <= 0.45
        and validation.p_value <= 0.45
        and selection.p_value <= 0.50
    )


def prelock_score(train: EvaluationResult, validation: EvaluationResult, selection: EvaluationResult) -> float:
    z_sum = sum(0.0 if pd.isna(x) else float(x) for x in [train.rcsi_z, validation.rcsi_z, selection.rcsi_z])
    return float(z_sum - 0.5 * (train.p_value + validation.p_value + selection.p_value))


def winner_gate(lockbox: EvaluationResult) -> bool:
    return (
        lockbox.trades >= MIN_LOCKBOX_TRADES
        and lockbox.rcsi_z >= WINNER_MIN_Z
        and lockbox.p_value <= WINNER_MAX_P
        and lockbox.percentile >= WINNER_MIN_PERCENTILE
    )


def main() -> None:
    print("Starting winner hunt with strict lockbox protocol...")
    print(f"Tickers: {', '.join(TICKERS)}")
    print(
        f"Samples per family: {SAMPLES_PER_FAMILY}, pre sims: {PRELOCK_SIMULATIONS}, "
        f"selection sims: {SELECTION_SIMULATIONS}, lockbox sims: {LOCKBOX_SIMULATIONS}"
    )

    family_config: list[tuple[str, Callable[[np.random.Generator], dict[str, float | int]], RunnerBuilder]] = [
        ("vol_mom", sample_vol_mom_params, build_vol_mom_runner),
        ("trend_pullback", sample_pullback_params, build_pullback_runner),
        ("breakout", sample_breakout_params, build_breakout_runner),
        ("mean_reversion", sample_mean_reversion_params, build_mean_reversion_runner),
    ]

    rng = np.random.default_rng(SEED)
    scan_rows: list[dict[str, object]] = []
    lockbox_rows: list[dict[str, object]] = []

    for ticker in TICKERS:
        market_df = augment_candidate_features(build_regime_dataframe_for_ticker(ticker, save_output=False))
        splits = split_into_lockbox_windows(market_df)
        print(f"[{ticker}] split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

        for family_name, sampler, builder in family_config:
            best_row: dict[str, object] | None = None
            best_params: dict[str, float | int] | None = None
            best_score = float("-inf")
            for sample_index in range(SAMPLES_PER_FAMILY):
                params = sampler(rng)
                runner = builder(params)
                run_id = f"hunt_{family_name}_{sample_index}"
                seed_base = SEED + sum((i + 1) * ord(ch) for i, ch in enumerate(f"{ticker}|{family_name}|{sample_index}"))

                train_eval = evaluate_runner(
                    runner=runner,
                    ticker=ticker,
                    split_name="train",
                    split_df=splits["train"],
                    simulation_count=PRELOCK_SIMULATIONS,
                    seed=seed_base + 11,
                    agent_name=f"winner_hunt_{family_name}",
                    run_id=run_id,
                )
                validation_eval = evaluate_runner(
                    runner=runner,
                    ticker=ticker,
                    split_name="validation",
                    split_df=splits["validation"],
                    simulation_count=PRELOCK_SIMULATIONS,
                    seed=seed_base + 23,
                    agent_name=f"winner_hunt_{family_name}",
                    run_id=run_id,
                )
                selection_eval = evaluate_runner(
                    runner=runner,
                    ticker=ticker,
                    split_name="selection",
                    split_df=splits["selection"],
                    simulation_count=SELECTION_SIMULATIONS,
                    seed=seed_base + 37,
                    agent_name=f"winner_hunt_{family_name}",
                    run_id=run_id,
                )
                score = prelock_score(train_eval, validation_eval, selection_eval)
                gated = prelock_gate(train_eval, validation_eval, selection_eval)

                row = {
                    "ticker": ticker,
                    "family": family_name,
                    "sample_index": sample_index,
                    "params": repr(params),
                    "prelock_gate": bool(gated),
                    "prelock_score": float(score),
                    "train_trades": int(train_eval.trades),
                    "train_rcsi_z": float(train_eval.rcsi_z),
                    "train_p_value": float(train_eval.p_value),
                    "validation_trades": int(validation_eval.trades),
                    "validation_rcsi_z": float(validation_eval.rcsi_z),
                    "validation_p_value": float(validation_eval.p_value),
                    "selection_trades": int(selection_eval.trades),
                    "selection_rcsi_z": float(selection_eval.rcsi_z),
                    "selection_p_value": float(selection_eval.p_value),
                }
                scan_rows.append(row)

                if score > best_score:
                    best_score = score
                    best_row = row
                    best_params = dict(params)

            if best_row is None or best_params is None:
                continue

            # Evaluate exactly once on lockbox for the chosen pre-lockbox winner.
            best_runner = builder(best_params)
            lock_eval = evaluate_runner(
                runner=best_runner,
                ticker=ticker,
                split_name="lockbox",
                split_df=splits["lockbox"],
                simulation_count=LOCKBOX_SIMULATIONS,
                seed=SEED + sum((i + 1) * ord(ch) for i, ch in enumerate(f"{ticker}|{family_name}|lockbox")),
                agent_name=f"winner_hunt_{family_name}",
                run_id=f"hunt_{family_name}_lockbox",
            )
            lockbox_rows.append(
                {
                    "ticker": ticker,
                    "family": family_name,
                    "params": best_row["params"],
                    "prelock_gate": bool(best_row["prelock_gate"]),
                    "prelock_score": float(best_row["prelock_score"]),
                    "lockbox_trades": int(lock_eval.trades),
                    "lockbox_rcsi_z": float(lock_eval.rcsi_z),
                    "lockbox_p_value": float(lock_eval.p_value),
                    "lockbox_percentile": float(lock_eval.percentile),
                    "winner_gate": bool(winner_gate(lock_eval)),
                }
            )

    scan_df = pd.DataFrame(scan_rows)
    lockbox_df = pd.DataFrame(lockbox_rows)
    DATA_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    scan_df.to_csv(OUTPUT_SCAN_PATH, index=False)
    lockbox_df.to_csv(OUTPUT_LOCKBOX_PATH, index=False)

    winners_df = lockbox_df[lockbox_df["winner_gate"]].copy() if not lockbox_df.empty else pd.DataFrame()
    print("\nWinner hunt complete.")
    print(f"Scan results: {OUTPUT_SCAN_PATH}")
    print(f"Lockbox results: {OUTPUT_LOCKBOX_PATH}")
    if winners_df.empty:
        print("No lockbox winner found under the configured criteria.")
    else:
        print("\nLockbox winners:")
        print(winners_df.sort_values(["lockbox_p_value", "lockbox_rcsi_z"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
