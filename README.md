# Distinguishing Skill from Luck in Trading Strategies

## Overview

This project tests whether rule-based trading strategies outperform a defensible randomness baseline under realistic execution assumptions.

The central question is not whether a strategy made money on one historical path. It is whether the realized outcome is statistically unusual relative to randomized alternatives that preserve key structural features of the strategy, including trade count, holding durations, transaction costs, and capital constraints.

## Research Objective

The framework is built to answer one question:

> Does a strategy exhibit reproducible edge, or can its observed performance be explained by chance?

To answer that question, the project combines:

- deterministic backtests
- matched random-timing Monte Carlo baselines
- repeated-run robustness testing across seeds
- regime-conditioned analysis
- false-discovery control across ticker-strategy results
- multi-asset walk-forward evaluation

## Strategies Included

The active pipeline evaluates five live strategies plus a passive benchmark:

- `trend`
- `mean_reversion`
- `random`
- `momentum`
- `breakout`
- `buy_and_hold` benchmark

## Core Methodology

### 1. Strategy Execution

Strategies are generated from daily market data using past-only signals. Orders are executed on the next bar, not on the signal bar.

### 2. Realistic Execution Layer

The execution model applies:

- next-open fills
- commissions
- spread and randomized slippage
- finite capital
- integer share sizing
- liquidity caps based on recent average volume

### 3. Daily-Curve Performance Measurement

Performance is evaluated from daily equity curves rather than only from a list of isolated trade returns. The comparison layer reports:

- cumulative return
- annualized return
- annualized Sharpe ratio from daily returns
- max drawdown
- trade-level return ratio

### 4. Monte Carlo Null Model

The null model is a matched random-timing benchmark. It preserves each strategy's realized trade count and holding durations, then randomizes when those trades occur on the observed market path. This is materially stronger than a naive bootstrap of realized trade returns.

### 5. Skill Metrics

The framework reports:

- `RCSI = actual cumulative return - median simulated return`
- `RCSI_z` for scale-free comparison
- one-sided p-values
- percentile rank inside the null distribution
- p-value prominence `-log10(p)`

### 6. Robustness

Each strategy is re-evaluated across repeated Monte Carlo seeds. Verdicts are based on repeated-run evidence, not on a single simulation draw.

### 7. Multiple-Testing Control

Cross-ticker comparisons include Benjamini-Hochberg false-discovery-rate adjustment so that isolated significant results are not overinterpreted.

### 8. Walk-Forward Generalization

The project includes a multi-asset walk-forward engine that reruns each strategy inside each fold from a fresh capital state. This separates ticker-specific in-sample evidence from broader generalization claims.

## Why This Matters

A profitable backtest is not proof of skill.

A strategy can:

- make money
- look smooth
- outperform on one ticker
- appear repeatable in-sample

and still be consistent with randomness once tested against an appropriate null.

This project treats backtesting as a statistical validation problem rather than a performance-reporting exercise.

## Outputs

The pipeline produces:

- per-strategy trade logs
- metrics tables
- Monte Carlo summary files
- repeated-run robustness summaries
- RCSI and `RCSI_z` outputs
- regime analysis tables
- cross-ticker FDR-adjusted comparison tables
- strategy verdict summaries
- chart packs and combined PDFs
- multi-asset walk-forward panel summaries

## Sparse-Activity and No-Trade Cases

Some tickers can legitimately produce no completed trades under the live strategy rules and execution constraints. In those cases, the framework does not force a statistical judgment.

Instead, the pipeline now:

- labels the strategy as `No Trades`
- suppresses inferential fields such as p-value, percentile, and `RCSI_z` in the active comparison table
- records that skill-versus-luck inference is not applicable
- generates placeholder charts rather than crashing or plotting misleading bars

This matters for instruments whose market structure differs from the main research universe, including cases where external data sources provide limited liquidity information. A no-trade result is treated as non-evidence, not as evidence of either skill or luck.

## Project Structure

`Code/`  
Core scripts, strategy logic, Monte Carlo engine, charts, and pipeline entrypoints.

`Data_Raw/`  
Downloaded raw historical data.

`Data_Clean/`  
Feature files, trade logs, Monte Carlo outputs, comparison tables, walk-forward summaries, and benchmark outputs.

`Charts/`  
Generated figures and combined chart PDFs.

`Notes/`  
Research documentation, paper drafts, audit notes, math appendix, and methodology notes.

## How to Run

### Single-Ticker Pipeline

```bash
./.venv/bin/python Code/AAAmain.py
```

Then choose:

- `1` for the single-ticker research pipeline
- `2` for the multi-asset walk-forward study

### Single-Ticker Direct Run

```bash
export TICKER=SPY
./.venv/bin/python Code/AAAmain.py
```

### Multi-Asset Walk-Forward Direct Run

```bash
./.venv/bin/python Code/multi_asset_walk_forward.py
```

Optional overrides:

```bash
WALK_FORWARD_TICKERS=SPY,QQQ,AAPL,VOO,NVDA,TSM ./.venv/bin/python Code/multi_asset_walk_forward.py
```

```bash
WALK_FORWARD_OUTER_RUNS=50 WALK_FORWARD_SIMULATIONS_PER_RUN=1000 ./.venv/bin/python Code/multi_asset_walk_forward.py
```

## Interpretation Rules

The framework does not treat profitability as evidence.

A result is only treated as credible evidence of skill when it is:

- statistically rare under the null
- positive on scale-free evidence metrics
- stable across repeated runs
- not eliminated by broader walk-forward testing

Ticker-specific positive results and cross-asset generalizable results are treated as different claims.

## Main Limitations

- The system uses daily bars rather than intraday market microstructure.
- The null model randomizes trade timing on the observed market path rather than simulating entirely new market paths.
- Regime analysis is descriptive unless sufficient trade counts exist.
- Some strategy effects may be asset-specific even when they appear strong in one ticker.
- Some sparse-activity tickers may generate no completed trades under the current long-only daily implementation and execution constraints, which limits inference rather than contradicting it.
- Walk-forward evidence is stronger than in-sample evidence and should be weighted more heavily in final interpretation.

## Bottom Line

This repository is designed to answer a stricter question than a normal backtest:

> Was the observed performance meaningfully better than luck under realistic and repeated testing?

That standard is the basis for every strategy verdict in the project.
# VF-RCSI
# VF-RCSI
# VF-RCSI
