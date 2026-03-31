# Trading Research Project
## Internal Build Log, Technical Notes, and Research Documentation

> Archival note: this long-form reconstruction predates the March 30, 2026 audit refresh and the later same-day no-trade pipeline update. For the current live methodology and synchronized conclusions, use [README.md](/Users/aryanpatel/Downloads/Virtu_Fortuna_Project/README.md) and [paper.md](/Users/aryanpatel/Downloads/Virtu_Fortuna_Project/Notes/paper.md) as the primary sources of truth.

## March 30, 2026 Live Addendum

The live research system has evolved materially beyond the archival reconstruction below. The most important current-state updates are:

- The active pipeline evaluates five live strategies, `trend`, `mean_reversion`, `random`, `momentum`, and `breakout`, plus a contextual `buy_and_hold` benchmark.
- Statistical inference no longer uses a bootstrap of realized trade returns. The live null is a matched random-timing Monte Carlo benchmark that preserves trade count, holding durations, and capital-at-risk structure.
- The comparison layer now reports a true daily-curve `annualized_sharpe` and keeps the trade-based mean-to-standard-deviation metric explicitly labeled as `trade_level_return_ratio`.
- Regime labeling is forward-only, and regime charts suppress cells with fewer than 20 trades.
- The walk-forward engine reruns each fold from a fresh capital state instead of filtering fold trades out of one full-history run.
- Cross-ticker interpretation now uses Benjamini-Hochberg FDR control and emphasizes `RCSI_z`, percentile, and p-value prominence rather than raw `RCSI` alone.
- The chart and verdict layers now support a distinct `No Trades` state. When a ticker-strategy pair produces no completed trades, the pipeline suppresses inferential fields, labels the strategy `No Trades`, and emits placeholder charts instead of crashing.
- This no-trade path was validated explicitly on `EURUSD=X`. Under the current long-only daily implementation, all five active strategies produced zero completed trades. The full pipeline now completes successfully and records those strategies as non-inferable rather than misclassifying them as skill or luck.

The remainder of this document is preserved as an internal historical build log. Where it conflicts with the addendum above, the addendum reflects the live code.

## Abstract
This project is a trading research pipeline designed to evaluate whether the apparent performance of simple rule-based trading strategies reflects genuine skill or outcomes that are plausibly explained by randomness. The system studies three long-only strategies, trend following, mean reversion, and a random control, using daily historical market data downloaded from `yfinance`. It transforms raw prices into engineered features, labels volatility-based market regimes, generates completed trades with next-bar execution, computes trade-level performance metrics, and then evaluates each strategy against a Monte Carlo baseline built from bootstrapped trade returns.

The central methodological contribution of the project is the use of RCSI, the Regime-Conditional Skill Index, which measures the gap between actual cumulative strategy performance and the median outcome from the simulated distribution. This framework is paired with p-value estimation, percentile-based interpretation, regime-specific performance analysis, and robustness testing across repeated Monte Carlo seeds. Together, these components shift the analysis away from naive backtest celebration and toward a more disciplined question: whether a strategy's observed result is both better than a random baseline and stable enough to be taken seriously.

The final system produces a full research artifact set consisting of cleaned datasets, strategy trade logs, performance summaries, Monte Carlo result files, RCSI tables, robustness summaries, and publication-style visualizations. These outputs are bundled into a combined chart document for review. The project therefore functions simultaneously as a trading-strategy experiment, a statistical evaluation framework, and a reproducible internal research pipeline for studying the boundary between skill, luck, and regime dependence in historical strategy performance.

## Reconstruction Note
This document is a reconstructed internal build log based on the current workspace, the active codebase, the generated artifacts in `Data_Clean/` and `Charts/`, and the debugging/fix work performed in this environment.

Important context:

- The Git repository is initialized but currently has no commits, so there is no historical commit log to replay.
- The current live pipeline entrypoint is `Code/AAAmain.py`.
- There is no committed `Code/main.py` file in the current working tree.
- Some historical artifacts remain in `Data_Clean/` and `Charts/` from earlier phases of the project, which is itself part of the story of how the project evolved.

Because of that, the "build log" sections below are not a literal commit-by-commit changelog. They are a careful reconstruction of the project's development order from:

- script dependency structure,
- file naming conventions,
- current implementation details,
- generated data products,
- and the concrete issues that were encountered and fixed.

The result is intended to serve as official technical documentation, an internal research notebook, and a durable explanation of how the system was designed.

---

## 1. Project Overview

### 1.1 Core Research Question
The project asks a specific and important research question:

How can we tell whether the apparent success of a trading strategy reflects real skill rather than luck?

This is the central problem the system is built to investigate. Trading strategies can look impressive on a backtest for many reasons:

- the market happened to trend in a way that favored the rules,
- the sequence of wins and losses happened to be favorable,
- the sample may be too small,
- the strategy may simply be exploiting noise,
- or a control strategy may produce similar outcomes through randomness.

The project therefore does not stop at raw performance. Instead, it builds a full comparison framework that asks:

- What did the strategy actually earn?
- What would a randomized version of its trade-return process have earned?
- How different is the actual result from that random baseline?
- Is that difference stable across repeated random seeds?
- Does the answer change by market regime?

### 1.2 Why This Matters
This matters because a naive trading backtest often answers the wrong question.

A simple backtest tells you:

- "What happened when these rules were applied to historical data?"

But it does not tell you:

- "Was that result unusually good relative to a plausible randomness baseline?"
- "Would I reach the same conclusion if I reran the randomness model with a different seed?"
- "Did the strategy only work in one market condition?"
- "Is the result dominated by a few unusually favorable outcomes?"

This project matters because it tries to move from:

- performance description,

to:

- evidence of skill,
- quantified uncertainty,
- robustness,
- and regime sensitivity.

### 1.3 What the Project Tries to Prove
The project does not try to prove that any one strategy is universally profitable.
It tries to prove something more disciplined:

1. A strategy's observed return should be evaluated against a randomized baseline, not in isolation.
2. The distribution of randomized outcomes matters more than a single simulated average.
3. Median-based comparison is useful when simulation distributions are skewed.
4. Apparent outperformance may be fragile across Monte Carlo seeds.
5. Strategy performance depends on market regime, not just overall sample period.

In other words, the project is a research system for distinguishing:

- raw performance,
- from statistically persuasive performance,
- from robust performance,
- from regime-dependent performance.

### 1.4 Final Outputs
The project produces three kinds of outputs.

#### A. Structured Data Outputs
These are CSV files in `Data_Clean/`, including:

- cleaned feature tables,
- regime tables,
- trade logs for each strategy,
- performance metric tables,
- Monte Carlo simulation results,
- Monte Carlo summary tables,
- RCSI tables,
- regime analysis tables,
- robustness run tables,
- robustness summary tables,
- strategy comparison tables.

#### B. Visual Outputs
These are charts in `Charts/`, including:

- combined equity curve chart,
- Monte Carlo histogram per strategy,
- RCSI bar chart,
- regime Sharpe grouped bar chart,
- regime heatmap,
- robustness RCSI chart,
- robustness percentile chart,
- p-value bar chart,
- a combined multi-page PDF chart bundle.

#### C. Interpretive Conclusions
The system is designed to answer questions like:

- Which strategy performed best in raw terms?
- Which strategy performed best relative to its Monte Carlo baseline?
- Is the observed outperformance statistically persuasive?
- Is that conclusion robust across Monte Carlo seeds?
- Does a strategy work differently in calm, neutral, and stressed volatility environments?

---

## 2. High-Level System Architecture

### 2.1 End-to-End Pipeline Structure
At a high level, the system is a sequential research pipeline:

1. Load historical daily market data.
2. Engineer features from price history.
3. Label market regimes from rolling volatility.
4. Run strategy agents to create completed trade logs.
5. Compute trade-level summary metrics.
6. Run Monte Carlo simulations on trade returns.
7. Convert Monte Carlo outputs into summary statistics and p-values.
8. Compute RCSI as the key skill-versus-randomness metric.
9. Test Monte Carlo result stability across seeds.
10. Build a strategy comparison table.
11. Generate charts.
12. Bundle the charts into one combined PDF and open it.

### 2.2 Data Flow
The data flow is:

`yfinance raw price history`
→ `Data_Raw/{ticker}.csv`
→ `Data_Clean/{ticker}_features.csv`
→ `Data_Clean/{ticker}_regimes.csv`
→ `{ticker}_{agent}_trades.csv`
→ `{ticker}_{agent}_metrics.csv`
→ `{ticker}_{agent}_monte_carlo_results.csv`
→ `{ticker}_monte_carlo_summary.csv`
→ `{ticker}_rcsi.csv`
→ `{ticker}_regime_analysis.csv`
→ robustness CSVs
→ charts in `Charts/`
→ `{ticker}_pipeline_charts.pdf`

### 2.3 Major Components and Their Roles

#### Data Loading
`Code/data_loader.py`

- Downloads full daily history via `yfinance`.
- Normalizes columns.
- Writes a clean raw CSV.

#### Feature Engineering
`Code/features.py`

- Computes daily return.
- Computes moving averages.
- Computes rolling volatility.
- Produces a standardized feature table.

#### Regime Detection
`Code/regimes.py`

- Uses rolling volatility to classify each day as:
  - calm,
  - neutral,
  - stressed.

#### Strategy Agents

- `Code/trend_agent.py`
- `Code/mean_reversion_agent.py`
- `Code/random_agent.py`

Each agent:

- reads the regime-enriched feature table,
- applies its own entry/exit logic,
- executes on the next bar's open,
- and outputs completed trades.

#### Trade Generation
Trade generation is embedded inside each agent.

The trade logs record:

- entry date,
- exit date,
- entry price,
- exit price,
- realized return,
- regime at entry.

#### Metrics

- `Code/trend_metrics.py`
- `Code/mean_reversion_metrics.py`
- `Code/random_metrics.py`

These compute:

- total trades,
- average return,
- median return,
- win rate,
- standard deviation,
- Sharpe-like ratio,
- average win,
- average loss,
- expected value.

#### Monte Carlo Simulation
Primary active engine:

- `Code/monte_carlo.py`

Secondary / archival alternate engine:

- `Code/realistic_monte_carlo.py`

The active engine:

- adjusts trade returns for transaction cost,
- converts them to log returns,
- resamples them with replacement,
- creates 5,000 simulated paths per strategy,
- and records the resulting outcome distribution.

#### RCSI
`Code/rcsi.py`

This is the key research contribution.
It computes a skill-versus-randomness gap by comparing:

- actual cumulative return,
- against median simulated cumulative return.

#### Robustness Testing

- `Code/monte_carlo_robustness.py`
- `Code/monte_carlo_robustness_plot.py`

These scripts rerun Monte Carlo across multiple random seeds to test whether conclusions remain stable.

#### Benchmark Comparison

- Historical benchmark script: `Code/buy_and_hold.py`
- Strategy comparison table: `Code/compare_agents.py`

The benchmark was originally included directly in the comparison flow and chart outputs.
It has since been removed from the active pipeline outputs, but the script and some generated benchmark artifacts remain in the repository as part of the project history.

#### Visualization System

- `Code/equity_curve.py`
- `Code/monte_carlo_plot.py`
- `Code/rcsi_plot.py`
- `Code/regime_plot.py`
- `Code/rcsi_heatmap.py`
- `Code/monte_carlo_robustness_plot.py`
- `Code/p_value_plot.py`
- `Code/open_charts.py`
- `Code/plot_config.py`

These scripts convert the research outputs into publication-style charts.

#### Pipeline Automation
Current live orchestrator:

- `Code/AAAmain.py`

This script:

- prompts for a ticker,
- exports it into the environment,
- runs each module in order,
- stops on the first failure,
- and opens the final chart bundle.

---

## 3. Data Pipeline

### 3.1 Data Source
The project uses `yfinance` as the external data source.

In `Code/data_loader.py`, the call is:

- `yf.download(ticker, period="max", interval="1d", progress=False)`

This means:

- full available daily history is requested,
- for one ticker at a time,
- with no progress bar noise.

The dependency list in `requirements.txt` is intentionally small:

- `matplotlib==3.10.8`
- `pandas==3.0.1`
- `seaborn==0.13.2`
- `yfinance==1.2.0`

### 3.2 Raw Storage Design
The raw file is saved to:

- `Data_Raw/{ticker}.csv`

The raw loader keeps only:

- `Date`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

This is a deliberate narrowing of the raw file format.
The project avoids carrying extra provider-specific metadata through the rest of the pipeline.

### 3.3 Directory Handling
The project supports both lower- and upper-case directory naming:

- `Data_Raw` or `data_raw`
- `Data_Clean` or `data_clean`
- `Charts` or `charts`

This support appears repeatedly throughout the codebase.
The purpose is practical:

- some earlier runs or environments created upper-case folders,
- some scripts were later written to prefer lower-case naming,
- so helper functions resolve whichever exists.

This is important because it prevented path failures across iterations of the project.

### 3.4 Cleaning Rules
The raw loader does several important cleaning steps:

1. Reset the yfinance date index into a normal `Date` column.
2. Flatten any yfinance two-level column index.
3. Keep only the six expected market columns.
4. Drop rows with missing values.
5. Sort from oldest to newest.

The feature loader adds another layer of safety:

- `Date` is parsed with `errors="coerce"`.
- numeric columns are converted with `errors="coerce"`.
- invalid rows are dropped.

This two-layer cleaning matters because older CSVs may contain:

- malformed date rows,
- extra ticker rows beneath the header,
- strings in numeric columns,
- or inconsistent provider output.

### 3.5 Date Handling
Date handling is explicit throughout the pipeline:

- raw data is converted to datetime,
- feature files carry datetime-compatible `Date`,
- trades store both `entry_date` and `exit_date`,
- trade logs are sorted by `exit_date`,
- regime tables are sorted by `Date`,
- charts use actual datetime values on the x-axis.

This reduces:

- sorting ambiguity,
- stale string-based ordering problems,
- and date-format inconsistencies across CSV boundaries.

### 3.6 Missing Data Handling
The pipeline handles missing data conservatively:

- rows missing required inputs are dropped,
- files missing required columns trigger explicit errors,
- empty files trigger explicit errors,
- early rolling-window rows are intentionally removed,
- incomplete open trades at the end of the series are not saved.

This is a key design choice:
the project prefers losing a small amount of unusable data over carrying ambiguous or invalid rows deeper into the analysis.

### 3.7 Consistency Mechanisms
Consistency is maintained by repeated validation:

- every major CSV loader checks required columns,
- date columns are reparsed at each stage,
- numeric columns are reconverted at each stage,
- strategy files auto-create upstream data when absent,
- Monte Carlo plotting re-validates summary statistics against raw simulation outputs.

The system is therefore defensive, not optimistic.
It assumes files can be stale, incomplete, or mismatched and tries to detect that early.

---

## 4. Feature Engineering

### 4.1 Purpose of the Feature Layer
The feature layer turns raw daily OHLCV data into a compact state description that the strategies and regime model can use.

The current feature set is intentionally minimal:

- daily return,
- 20-day moving average,
- 50-day moving average,
- 20-day rolling volatility of returns.

This is a deliberate design decision.
The project is not trying to win a feature-engineering contest.
It is trying to build a transparent and auditable research framework for skill-versus-luck analysis.

### 4.2 Features Created
In `Code/features.py`, the following columns are added.

#### Daily Return
Formula:

`daily_return_t = (Close_t / Close_(t-1)) - 1`

Why it was chosen:

- it is the simplest daily price-change measure,
- it feeds volatility estimation,
- and it keeps the regime logic tied to realized movement, not price level.

#### 20-Day Moving Average
Formula:

`ma_20_t = mean(Close_(t-19) ... Close_t)`

Why it was chosen:

- it acts as the short-term reference line,
- it is used directly by the mean-reversion strategy,
- and it represents local price equilibrium.

#### 50-Day Moving Average
Formula:

`ma_50_t = mean(Close_(t-49) ... Close_t)`

Why it was chosen:

- it is a common medium-term trend filter,
- it is used directly by the trend-following strategy,
- and it gives a slower signal than the 20-day average.

#### 20-Day Rolling Standard Deviation of Returns
Formula:

`std_20_t = std(daily_return_(t-19) ... daily_return_t)`

Why it was chosen:

- it is a direct measure of local realized volatility,
- it is simple enough to explain clearly,
- it powers both the regime model and the mean-reversion entry threshold.

### 4.3 Why These Features Were Chosen
The feature set is intentionally interpretable.
Each feature has a direct conceptual role:

- `ma_50` answers: Is price above or below medium-term trend?
- `ma_20` answers: Where is short-term equilibrium?
- `std_20` answers: How turbulent has the recent environment been?
- `daily_return` answers: What was the one-step realized move?

This keeps the project understandable and makes the later research argument stronger.

If the feature set were much more complex, it would become harder to distinguish whether the project was about:

- alpha mining,
- or understanding skill vs luck.

This project prioritizes the latter.

### 4.4 Rolling Window Implication
Because `ma_50` and `std_20` require history, the first portion of the dataset cannot be used after feature creation.

Specifically:

- the 50-day moving average is the longer lookback,
- so the early rows before it becomes available are dropped,
- and the finished feature table begins only once all rolling columns are populated.

This is why the feature stage ends with:

- `df = df.dropna().reset_index(drop=True)`

### 4.5 Conceptual Trade-Off
The feature set sacrifices sophistication for clarity.

That trade-off is intentional:

- fewer features make errors easier to trace,
- feature meanings are economically interpretable,
- and the downstream strategies remain easy to audit.

The cost is that the strategies may miss richer signals.
But that is acceptable for a project whose main contribution is the evaluation framework, not feature novelty.

---

## 5. Regime Detection

### 5.1 Why Regime Analysis Exists
The project does not assume that one strategy should work equally well in all market conditions.

Instead, it explicitly asks:

- Does a strategy behave differently in low-volatility vs high-volatility environments?

This matters because many trading rules are regime-sensitive:

- trend-following often behaves differently in smooth directional markets,
- mean reversion behaves differently when markets are turbulent,
- random baselines may look surprisingly strong in some environments.

### 5.2 Regime Definition
In `Code/regimes.py`, regimes are defined using:

- `std_20`, the 20-day rolling standard deviation of daily returns.

Then `pd.qcut(..., q=3)` splits the sample into three equal-frequency buckets:

- lowest third → `calm`
- middle third → `neutral`
- highest third → `stressed`

This means the regime labels are:

- relative to the asset's own realized volatility history,
- not based on fixed absolute thresholds.

### 5.3 Why Terciles Were Chosen
Terciles are a practical compromise:

- two bins would be too coarse,
- more bins would thin the trade counts,
- three bins preserve enough structure to compare environments without starving the analysis.

The choice of:

- calm,
- neutral,
- stressed

is conceptually clean and visually easy to interpret in grouped charts and heatmaps.

### 5.4 What the Labels Actually Mean
The labels do not mean:

- calm = "safe market"
- stressed = "crash"

They only mean:

- lower recent volatility relative to this ticker's own history,
- middle recent volatility,
- higher recent volatility relative to this ticker's own history.

This is important because the model is cross-time relative, not absolute.
For one ticker, "stressed" may still be milder than another ticker's "neutral."

### 5.5 How Thresholds Were Chosen
The thresholds are not hand-picked numerical cutoffs.
They are empirical quantiles derived from the observed `std_20` distribution.

That choice avoids:

- arbitrary hand tuning,
- ticker-specific manual thresholds,
- and brittle hard-coded volatility boundaries.

### 5.6 Limitations of the Regime Model
The regime model is useful, but simplified.

Its limitations include:

- it uses only volatility,
- it ignores trend slope, volume shocks, or macro context,
- it treats each row independently once `std_20` is computed,
- it does not model regime persistence explicitly,
- it forces equal-sized bins even if the true market structure is uneven.

So the regime model is best interpreted as:

- a simple volatility-conditioning layer,

not:

- a full hidden-state market model.

---

## 6. Strategy Agents

### 6.1 Shared Design Assumptions
All three strategy agents share several important assumptions.

They are:

- long-only,
- single-position,
- non-overlapping,
- next-bar execution,
- full-entry / full-exit event logic,
- regime-at-entry metadata recorded from the signal day.

All three use the same execution discipline:

- the signal is evaluated on day `t`,
- the actual trade is executed at day `t+1` open.

This is a major anti-look-ahead-bias decision.

### 6.2 Trend Agent
File:

- `Code/trend_agent.py`

#### Entry Rule
Enter a long position when:

- `Close > ma_50`
- and the strategy is not already in a position.

Execution:

- buy at the next bar's open.

#### Exit Rule
Exit the long position when:

- `Close < ma_50`
- and the strategy is already in a position.

Execution:

- sell at the next bar's open.

#### Interpretation
This is a simple moving-average trend filter.
The idea is:

- above the 50-day average implies medium-term trend support,
- below the 50-day average implies trend deterioration.

#### Strengths

- simple and interpretable,
- naturally adapts to long trends,
- avoids heavy parameterization,
- behaves like a plausible baseline rule-based strategy.

#### Weaknesses

- whipsaw risk in sideways markets,
- no stop-loss,
- no partial exits,
- no volatility scaling,
- binary logic may be too crude for choppy environments.

### 6.3 Mean Reversion Agent
File:

- `Code/mean_reversion_agent.py`

#### Entry Rule
Define:

`buy_threshold = ma_20 - 2 * std_20`

Enter when:

- `Close < buy_threshold`
- and not already in a position.

Execution:

- buy at next bar's open.

#### Exit Rule
Exit when:

- `Close >= ma_20`
- and already in a position.

Execution:

- sell at next bar's open.

#### Interpretation
This strategy assumes that large downside deviations below a short-term moving average may revert toward the mean.

The threshold uses:

- a rolling average as equilibrium,
- plus rolling return volatility as a deviation scale.

#### Strengths

- explicitly tied to short-term overextension,
- more volatility-aware than a plain moving-average cross,
- often produces high win rates in calmer environments.

#### Weaknesses

- can keep buying into breakdowns,
- assumes reversion will happen,
- may underperform in prolonged trends or crashes,
- uses return volatility as a price-deviation scale, which is simple but not perfect.

### 6.4 Random Agent
File:

- `Code/random_agent.py`

#### Entry Rule
When flat:

- enter with probability `0.05` on a given signal day.

Execution:

- buy at next bar's open.

#### Exit Rule
When in position:

- exit with probability `0.05` on a given signal day.

Execution:

- sell at next bar's open.

#### Interpretation
This strategy is intentionally not signal-based.
It is a behavioral control.

It exists to answer:

- what happens when trades are created without information-driven entry logic?

#### Randomness Design
By default:

- it uses `random.SystemRandom()`,
- so it is truly stochastic across runs.

Optional reproducibility mode exists via environment variables:

- `RANDOM_AGENT_REPRODUCIBLE=1`
- `RANDOM_AGENT_SEED=42`

#### Strengths

- provides a control baseline,
- uses the same execution mechanics as the rule-based agents,
- helps expose whether "performance" can emerge from random position timing.

#### Weaknesses

- trade count is not forced to match other agents,
- no informational edge by construction,
- its performance is highly run-dependent unless reproducibility is enabled.

### 6.5 Why These Three Were Included
The three-agent design is conceptually balanced:

- `trend` tests momentum/trend continuation,
- `mean_reversion` tests oversold snapback,
- `random` tests a no-skill control.

This makes the project stronger because it is not only comparing:

- one clever strategy to itself,

but:

- multiple logic families,
- plus a stochastic control.

---

## 7. Trade Generation

### 7.1 Trade Construction Logic
Each agent emits trades as dictionaries with:

- `entry_date`
- `exit_date`
- `entry_price`
- `exit_price`
- `return`
- `regime_at_entry`

The project intentionally stores completed trades, not signals.

This is important because later stages care about realized trade outcomes.

### 7.2 Return Calculation
Trade return is calculated as:

`trade_return = (exit_price - entry_price) / entry_price`

This is:

- simple gross long return,
- before Monte Carlo transaction-cost adjustment,
- and before any compounding is applied at the portfolio level.

### 7.3 Execution Ordering
Execution is always:

- signal on current row,
- trade on next row's open.

This ordering was an important methodological choice.
It prevents the unrealistic assumption that the system can observe a closing condition and immediately trade at that same close.

### 7.4 No Overlap Assumption
Each agent uses:

- `in_position = False/True`

This enforces:

- only one open position at a time,
- no pyramiding,
- no overlapping trades,
- no portfolio of multiple concurrent positions.

This simplifies both the strategy logic and the Monte Carlo interpretation.

### 7.5 Regime Attribution
The regime attached to a trade is:

- the regime on the signal day,
- not the next day when execution happens.

This is conceptually important.
It keeps the regime label tied to information that was actually available when the decision was made.

### 7.6 End-of-Series Handling
If a strategy is still in an open trade at the end of the file:

- that trade is not written.

This avoids:

- inventing an exit,
- using partial / open mark-to-market values,
- or mixing completed and incomplete trades in the downstream analysis.

### 7.7 Embedded Assumptions
The trade generator assumes:

- long-only exposure,
- full capital allocated to each trade in a normalized sense,
- no leverage,
- no shorting,
- no partial fills,
- no slippage,
- no financing cost,
- and no execution delay beyond one-bar-next-open logic.

These assumptions make the project cleaner, but also set its modeling limits.

---

## 8. Performance Metrics

### 8.1 Purpose
The metrics scripts summarize trade logs before Monte Carlo is applied.
They answer:

- how often each strategy trades,
- how often it wins,
- how large its wins and losses tend to be,
- and whether the average trade looks attractive on its own terms.

### 8.2 Metrics Computed

#### Total Trades
Formula:

`total_trades = number of completed trades`

Why it matters:

- sample size affects reliability,
- small samples can create misleading impressions.

#### Average Return
Formula:

`average_return = mean(trade_return_i)`

Why it matters:

- gives the mean trade payoff.

#### Median Return
Formula:

`median_return = median(trade_return_i)`

Why it matters:

- more robust than the mean when outliers exist.

#### Win Rate
Formula:

`win_rate = count(return > 0) / total_trades`

Why it matters:

- distinguishes frequent small wins from infrequent large wins.

#### Standard Deviation of Trade Returns
Formula:

`std_return = std(trade_return_i)`

Why it matters:

- measures trade-level volatility,
- needed for the Sharpe-style ratio.

#### Sharpe Ratio (Trade-Level, Non-Annualized)
Formula used in the code:

`sharpe_ratio = average_return / std_return`

if `std_return != 0`, otherwise `0`.

Important note:

- this is not an annualized institutional Sharpe ratio,
- it is a simplified trade-level dispersion-adjusted return measure.

#### Average Win
Formula:

`average_win = mean(return_i | return_i > 0)`

#### Average Loss
Formula:

`average_loss = abs(mean(return_i | return_i < 0))`

The absolute value is taken so that the magnitude of losses is easy to compare against wins.

#### Expected Value
Formula:

`expected_value = (win_rate * average_win) - ((1 - win_rate) * average_loss)`

Why it matters:

- it combines win frequency and payoff asymmetry,
- and helps explain why a strategy can have:
  - a high win rate but poor expectancy,
  - or a lower win rate but good expectancy.

### 8.3 Example from Current SPY Outputs
In the current SPY metrics files:

- trend:
  - 290 trades,
  - average return about `0.00633`,
  - win rate about `0.3448`,
  - trade-level Sharpe about `0.1427`.
- mean reversion:
  - 479 trades,
  - average return about `0.00451`,
  - win rate about `0.7662`,
  - trade-level Sharpe about `0.1786`.
- random:
  - 218 trades,
  - average return about `0.01011`,
  - win rate about `0.6468`,
  - trade-level Sharpe about `0.2478`.

These numbers already show why raw metrics alone are not enough:

- the random agent can look strong on simple trade statistics,
- but that does not mean it demonstrates skill.

That is precisely why the Monte Carlo and RCSI layers exist.

---

## 9. Monte Carlo Simulation

### 9.1 Why Monte Carlo Is Used
Monte Carlo is the core anti-self-deception mechanism in the project.

A single backtest result does not say whether the outcome is unusual.
Monte Carlo creates a distribution of plausible randomized outcomes based on the observed trade-return process.

The key question becomes:

- Is the actual result meaningfully different from what repeated random resampling would produce?

### 9.2 Active Monte Carlo Engine
The production engine is `Code/monte_carlo.py`.

Its key settings are:

- `NUMBER_OF_SIMULATIONS = 5000`
- `TRANSACTION_COST = 0.001`
- `REPRODUCIBLE = False`
- `SEED = 42`

### 9.3 Transaction Cost Adjustment
Before simulation, raw trade returns are adjusted:

`adjusted_return = raw_return - transaction_cost`

Transaction cost is fixed at:

- `0.001`, meaning `0.1%` per trade.

This adjustment serves two purposes:

1. It makes the strategy result less optimistic.
2. It ensures actual and simulated returns are compared on the same friction-adjusted basis.

### 9.4 Why Log Returns Are Used
The simulation converts adjusted returns into log returns:

`log_return = ln(1 + adjusted_return)`

This matters because compounding multiplicative returns is cleaner in log space.

If a sequence of returns is `r1, r2, ..., rn`, then:

- arithmetic accumulation is awkward,
- but log returns sum naturally.

Then final cumulative return is reconstructed with:

`cumulative_return = exp(sum(log_returns)) - 1`

This gives numerical stability and correct multiplicative compounding.

### 9.5 Simulation Mechanics
For each strategy:

1. Load the completed trade returns.
2. Adjust for transaction cost.
3. Convert to log returns.
4. Compute the actual cumulative return from the observed log-return sequence.
5. Resample the trade-level log returns with replacement.
6. For each simulated path:
   - draw exactly as many trades as the original strategy had,
   - sum the sampled log returns,
   - convert back to cumulative return.
7. Repeat 5,000 times.

In `monte_carlo.py`, this is vectorized:

- a matrix of random indices is created,
- each row is one simulated path,
- each row has length equal to the actual number of trades,
- and all cumulative returns are computed efficiently.

### 9.6 What Randomness Represents Here
Randomness in this Monte Carlo does not mean:

- random price paths,
- random market simulation,
- or a synthetic stochastic process model.

Instead, it represents:

- randomized resampling from the observed trade-return distribution of the strategy.

So it asks:

- given the empirical distribution of this strategy's trade outcomes,
- what range of final compounded outcomes can arise by chance through resampling?

This is a bootstrap-style randomness baseline.

### 9.7 What Assumptions Are Made
The simulation assumes:

- trade returns are exchangeable enough to resample with replacement,
- the empirical return set is informative about the strategy's trade-level outcome distribution,
- path dependence is sufficiently approximated by compounding sampled trades,
- no additional market-state model is necessary for the baseline.

This is useful, but imperfect.
It ignores:

- autocorrelation,
- regime sequencing,
- changing volatility structure,
- and portfolio context.

### 9.8 Summary Statistics Produced
The Monte Carlo summary table stores:

- actual cumulative return,
- median simulated return,
- mean simulated return,
- standard deviation of simulated returns,
- actual percentile,
- p-value,
- lower 5th percentile,
- upper 95th percentile,
- trade count,
- transaction cost,
- simulation count,
- reproducibility metadata.

### 9.9 Baseline Simulation vs Improved Realistic Simulation
There are two Monte Carlo implementations in the repository.

#### A. Active Production Engine: `monte_carlo.py`
This is the version used by the live pipeline.

Its strengths:

- vectorized,
- rigorous validation,
- explicit cost adjustment,
- log-return compounding,
- output summary table,
- output raw simulation results,
- p-value interpretation,
- reproducibility toggle.

#### B. Alternate / Historical Engine: `realistic_monte_carlo.py`
This file is not currently called by `AAAmain.py`, but it is still present.

It performs a similar idea:

- adjust returns for cost,
- convert to log returns,
- resample them,
- compute final outcomes.

Its style is more explicit and stepwise:

- it loops over simulations,
- writes a separate `*_realistic_monte_carlo.csv`,
- and appears to represent an intermediate or alternate implementation path.

The documentation implication is:

- the project explored more than one Monte Carlo design,
- and eventually consolidated around `monte_carlo.py` as the active research engine.

### 9.10 Why Resampling Works for This Use Case
Resampling works here because the project is not claiming to simulate the entire market.
It is trying to simulate:

- the variability of compounded outcomes implied by the observed trade-return set.

In other words:

- if the strategy's realized trades are one sample from an underlying trade-return process,
- bootstrapping that sample gives a practical baseline distribution.

This is enough to ask:

- is the actual compounded result meaningfully above the center of that randomized distribution?

---

## 10. RCSI (Key Contribution)

### 10.1 Definition
RCSI is defined in `Code/rcsi.py` as:

`RCSI = actual_cumulative_return - median_simulated_return`

There is also a standardized companion metric:

`RCSI_z = (actual_cumulative_return - mean_simulated_return) / std_simulated_return`

when the simulated standard deviation is nonzero.

### 10.2 Intuition
RCSI answers a direct question:

- how much better or worse was the real strategy outcome than the center of the Monte Carlo baseline?

If a strategy's actual performance is only equal to the center of its simulated distribution, there is weak evidence of skill.

If it is materially above that center, that suggests the actual result is better than what random reshuffling would usually produce.

### 10.3 Why Median Is Used Instead of Mean
This is one of the most important methodological decisions in the project.

The simulation distributions can be highly skewed.
That is visible in current artifacts such as AAPL, where:

- the trend strategy's mean simulated return is enormous because of heavy right tails,
- while the median simulated return is much smaller and more representative of the typical outcome.

If the mean alone were used:

- a few extreme simulated paths could dominate the baseline,
- and the "expected" simulated result could become misleading.

The median is used because it is:

- robust to skew,
- more representative of the typical simulated outcome,
- and better aligned with the project's goal of comparing actual performance against a sensible central random baseline.

### 10.4 How RCSI Differs from Standard Metrics
RCSI is not just:

- return,
- Sharpe ratio,
- or win rate.

Those metrics describe strategy performance in isolation.
RCSI is explicitly comparative:

- it measures performance relative to a Monte Carlo baseline.

That makes it a bridge between:

- raw backtest metrics,
- and statistical interpretation.

### 10.5 Interpretation

#### Positive RCSI
Means:

- actual strategy return was above the median simulated return.

Interpretation:

- evidence that observed performance beat the typical randomized baseline.

#### Zero RCSI
Means:

- actual return was approximately equal to the median simulated return.

Interpretation:

- no obvious edge over the randomness baseline.

#### Negative RCSI
Means:

- actual performance was below the median simulated baseline.

Interpretation:

- the strategy underperformed what the random baseline would typically produce.

### 10.6 Current SPY Example
From the current rerun SPY outputs:

- trend: `RCSI ≈ +0.0605`
- random: `RCSI ≈ -0.0552`
- mean reversion: `RCSI ≈ -0.1106`

So, in the current SPY run:

- trend is the only strategy with positive raw RCSI,
- but the edge is small,
- and later sections show it is not supported by strong p-value evidence.

---

## 11. Robustness Across Seeds

### 11.1 Why One Monte Carlo Run Is Not Enough
A single Monte Carlo run can look precise while still being seed-sensitive.

If the baseline distribution changes meaningfully when the random seed changes, then:

- the percentile,
- the p-value,
- and the RCSI-based interpretation

may not be stable.

This matters especially when the observed edge is small.

### 11.2 Robustness Design
`Code/monte_carlo_robustness.py` addresses this by running:

- `OUTER_RUNS = 30`
- `SIMULATIONS_PER_RUN = 5000`

with:

- `BASE_SEED = 100`

Each outer run uses:

- `seed_used = BASE_SEED + outer_run`

Then a `SeedSequence` spawns child seeds for each agent.

This is a careful design because it avoids:

- one shared generator being used inconsistently across agents,
- while still making the outer run's seed explicit and reproducible.

### 11.3 What Gets Measured
For every outer run and every strategy, the robustness script records:

- actual cumulative return,
- median simulated return,
- mean simulated return,
- std simulated return,
- actual percentile,
- p-value,
- RCSI,
- RCSI_z,
- 5th and 95th percentiles,
- trade count,
- transaction cost.

### 11.4 Aggregated Stability Summary
The summary table then computes, per strategy:

- mean percentile,
- std percentile,
- min percentile,
- max percentile,
- mean p-value,
- std p-value,
- min p-value,
- max p-value,
- mean RCSI,
- std RCSI,
- min RCSI,
- max RCSI,
- mean median simulated return,
- std median simulated return.

### 11.5 Stability Classification
The code defines explicit interpretation thresholds.

Let:

- `percentile_range = max_percentile - min_percentile`
- `rcsi_relative_std = std_RCSI / max(abs(mean_RCSI), 0.05)`

Then:

- `stable` if:
  - percentile range ≤ 5.0
  - and relative RCSI std ≤ 0.10
- `moderately variable` if:
  - percentile range ≤ 15.0
  - and relative RCSI std ≤ 0.25
- otherwise `unstable`

### 11.6 Why This Matters
This is a major methodological strength of the project.

It recognizes that:

- a small positive RCSI is not persuasive if it moves around a lot across seeds,
- and a strategy can appear "better than random" in one run but not robustly so.

### 11.7 Current SPY Robustness Interpretation
Using the current SPY robustness summary:

- trend:
  - percentile range ≈ `2.58`
  - relative RCSI std ≈ `0.666`
  - classification: `unstable`
- mean reversion:
  - percentile range ≈ `3.98`
  - relative RCSI std ≈ `0.518`
  - classification: `unstable`
- random:
  - percentile range ≈ `2.42`
  - relative RCSI std ≈ `0.996`
  - classification: `unstable`

This is an important research conclusion:

- even though percentile ranges are fairly narrow,
- the RCSI effect sizes themselves are small,
- so their relative variability is large.

In plain language:

- the seed-to-seed Monte Carlo conclusion is not robust enough to call these strong skill signals.

---

## 12. P-Value and Significance

### 12.1 What the p-Value Represents Here
The project uses a one-sided p-value:

`p_value = proportion(simulated_return >= actual_return)`

This answers:

- if the randomized Monte Carlo baseline were true, how often would we see a result at least as good as the actual one?

### 12.2 Why This Definition Is Appropriate
The project is specifically testing for unusually strong actual performance.

So the p-value is framed as:

- right-tail probability of the simulated distribution relative to the actual result.

### 12.3 Interpretation

#### Low p-value
Means:

- few simulated outcomes beat the actual outcome.

Interpretation:

- stronger evidence that actual performance was unusually good relative to the random baseline.

#### High p-value
Means:

- many simulated outcomes are at least as good as the actual outcome.

Interpretation:

- weak evidence of skill over the simulated baseline.

### 12.4 Evidence Labels in Code
`monte_carlo.py` maps p-values into text:

- `< 0.05` → `strong evidence`
- `< 0.10` → `weak evidence`
- otherwise → `no evidence`

### 12.5 Current SPY Example
Current SPY p-values are:

- trend: `0.4898`
- mean reversion: `0.5180`
- random: `0.5056`

These are all far above `0.05`.

So for the current SPY run, the system concludes:

- no strategy shows strong statistical evidence against the random baseline.

### 12.6 Visual Significance Marker
`Code/p_value_plot.py` adds a dashed line at:

- `p = 0.05`

This makes the threshold visually explicit.

---

## 13. Buy-and-Hold Benchmark

### 13.1 Why This Benchmark Was Added
The buy-and-hold benchmark answers a different but still important question:

- if a passive investor had simply bought the asset and held it, how would that compare with active rule-based trading?

This matters because:

- a strategy can beat a randomness baseline but still lag a trivial passive exposure,
- and passive appreciation is often the real practical benchmark in long-only equity research.

### 13.2 How It Was Calculated
The benchmark lives in `Code/buy_and_hold.py`.

It works from the feature table and computes:

- `daily_return = Close.pct_change()`
- `wealth_index = Close / first_close`
- `cumulative_return = wealth_index - 1`
- `rolling_peak = wealth_index.cummax()`
- `drawdown = wealth_index / rolling_peak - 1`

Sharpe-like benchmark metric:

- mean daily return / std daily return

with:

- `BUY_HOLD_TRANSACTION_COST = 0.0`

So the benchmark was intentionally simple:

- buy once,
- hold through the full sample,
- no turnover friction.

### 13.3 How It Was Used
Historically, the benchmark fed:

- a benchmark metrics file,
- a benchmark curve file,
- the strategy comparison table,
- and, at one point, the combined equity chart.

### 13.4 Why It Was Removed from the Active Pipeline
Later in development, the active pipeline was simplified to focus on:

- trend,
- mean reversion,
- random.

The benchmark script remains in the repository, but it has been removed from:

- the current active comparison table,
- the current equity curve chart,
- and the current live pipeline steps.

This was a scope/focus decision:

- the core research question became skill vs luck among the explicit agents,
- not active-vs-passive portfolio management.

### 13.5 How Strategies Compared to It Conceptually
When the benchmark was active, it played two roles:

1. absolute passive baseline,
2. practical investor baseline.

That comparison is still valuable conceptually, even if it is no longer in the live chart outputs.

---

## 14. Visualization System

### 14.1 Design Philosophy
The visualization system is intentionally academic in tone.

It uses:

- serif fonts,
- muted colors,
- minimal clutter,
- explanatory note boxes,
- consistent labeling,
- publication-style spacing.

This is centralized in `Code/plot_config.py`.

### 14.2 Shared Visual Style
Key design choices:

- background: light off-white (`#FCFCFA`)
- grid: subtle dashed light gray
- spines: top/right hidden
- title font: serif, semibold
- colors:
  - trend: muted blue
  - mean reversion: muted amber
  - random: muted purple

The goal is not flashy UI.
It is readable research communication.

### 14.3 Equity Curve Chart
File:

- `Code/equity_curve.py`

What it shows:

- cumulative return path for each strategy over exit dates,
- with transaction cost adjustment applied before compounding.

How it works:

- converts trade returns to log returns,
- compounds them into a wealth index,
- converts wealth index into cumulative return,
- plots all three strategies on one figure,
- adds final-point markers,
- adds a note box with final return and trade counts.

Interpretation:

- path shape matters, not just final return,
- large drawdowns and unstable path behavior become visible immediately.

### 14.4 Monte Carlo Histogram
File:

- `Code/monte_carlo_plot.py`

What it shows:

- empirical distribution of simulated final cumulative returns for one strategy,
- actual result as dashed red vertical line,
- median simulation as dotted black line,
- 5th and 95th percentiles as gray reference lines.

Key design decisions:

- stale summary files are validated against raw simulation results before plotting,
- extreme tails may be clipped for display only,
- if linear scale still compresses the body, log1p spacing can be used,
- note boxes explain clipping and display mode.

Interpretation:

- where the actual result sits relative to the simulated mass,
- how skewed the simulation distribution is,
- whether the mean is likely to be misleading.

### 14.5 RCSI Bar Chart
File:

- `Code/rcsi_plot.py`

What it shows:

- one bar per strategy,
- positive values above zero line,
- negative values below zero line.

Special design feature:

- if one bar dominates, a zoom inset is added to reveal smaller bars.

Interpretation:

- direct skill-vs-randomness gap at a glance.

### 14.6 Regime Sharpe Grouped Bar Chart
File:

- `Code/regime_plot.py`

What it shows:

- Sharpe ratio by strategy within each regime.

Interpretation:

- whether a strategy is regime-specific or broadly consistent.

### 14.7 Regime Heatmap
File:

- `Code/rcsi_heatmap.py`

What it shows:

- strategy by regime matrix of Sharpe ratio values.

Design details:

- diverging academic palette,
- optional centering at zero when values cross sign,
- direct cell annotation,
- explicit regime and strategy labels,
- colorbar legend.

Interpretation:

- faster matrix-style comparison than grouped bars,
- useful for spotting clusters of strong/weak performance.

### 14.8 Robustness Charts
File:

- `Code/monte_carlo_robustness_plot.py`

There are two charts:

1. mean RCSI ± SD across seeds
2. mean actual percentile ± SD across seeds

Interpretation:

- whether the "skill" conclusion is stable,
- how much it moves when random sampling is rerun.

### 14.9 p-Value Chart
File:

- `Code/p_value_plot.py`

What it shows:

- one p-value bar per strategy,
- with a dashed significance threshold at `0.05`.

Interpretation:

- quick view of statistical-evidence strength.

### 14.10 Combined Chart Bundle
File:

- `Code/open_charts.py`

Current behavior:

- gather expected pipeline charts,
- convert them into one multi-page PDF with Pillow,
- open that single PDF in Preview.

This is important because the chart bundle is now a real combined artifact:

- `{ticker}_pipeline_charts.pdf`

rather than just a loose set of separately opened PNGs.

---

## 15. Pipeline Automation

### 15.1 Important Naming Note
The user-facing idea of a master pipeline script is correct, but in the current repository the live file is:

- `Code/AAAmain.py`

There is no committed `Code/main.py` file in the present working tree.

So, functionally, `AAAmain.py` is the project's current "main" script.

### 15.2 Ticker Input Flow
`AAAmain.py` prompts:

- "Enter a ticker symbol..."

The input is:

- stripped,
- validated as non-empty,
- uppercased,
- injected into the environment as `TICKER`.

Every downstream script reads:

- `os.environ.get("TICKER", "SPY")`

This is how a single ticker choice propagates through the full system.

### 15.3 Script Execution Order
Current pipeline order:

1. `data_loader.py`
2. `features.py`
3. `regimes.py`
4. `trend_agent.py`
5. `mean_reversion_agent.py`
6. `random_agent.py`
7. `trend_metrics.py`
8. `mean_reversion_metrics.py`
9. `random_metrics.py`
10. `regime_analysis.py`
11. `monte_carlo.py`
12. `monte_carlo_robustness.py`
13. `rcsi.py`
14. `compare_agents.py`
15. `rcsi_plot.py`
16. `regime_plot.py`
17. `rcsi_heatmap.py`
18. `equity_curve.py`
19. `monte_carlo_plot.py`
20. `monte_carlo_robustness_plot.py`
21. `p_value_plot.py`
22. `open_charts.py`

### 15.4 Execution Model
The orchestrator:

- resolves the `Code/` directory,
- uses the current Python interpreter,
- runs each script with `subprocess.run(..., check=True)`,
- and stops immediately on the first failure.

This is a fail-fast research pipeline.

### 15.5 Plot Handling
The pipeline sets:

- `SHOW_PLOTS = 0`
- `MPLBACKEND = Agg`
- `MPLCONFIGDIR = .matplotlib`

This prevents:

- intermediate pop-up windows during generation,
- backend/display issues during noninteractive runs,
- and global Matplotlib cache pollution.

### 15.6 Automatic Upstream Recovery
Many scripts can auto-create missing upstream files.

Examples:

- `features.py` can call `data_loader.py`
- `regimes.py` can call `features.py`
- agents can call `regimes.py`
- metrics can call agent scripts

This makes the system more forgiving when individual modules are run manually.

### 15.7 Final Outputs
At the end, the pipeline:

- leaves CSVs in `Data_Clean/`,
- leaves charts in `Charts/`,
- and opens the combined PDF chart bundle.

---

## 16. Errors and Obstacles

This section records the major problems that either clearly occurred during development or are directly evidenced by the current code and artifact history.

### 16.1 Problem: Missing Upstream Files During Partial Runs
Issue:

- downstream scripts could be run before their prerequisites existed.

Why it happened:

- the project supports both full pipeline execution and individual script execution.

Fix:

- many scripts import upstream `main()` functions and create missing prerequisites automatically.

What was learned:

- research pipelines become much easier to work with when each module can recover missing dependencies.

### 16.2 Problem: Directory Naming Inconsistency
Issue:

- some code paths used `Data_Clean`, others `data_clean`; similarly for raw data and charts.

Why it happened:

- the project evolved across runs and environments with different naming preferences.

Fix:

- repeated `resolve_named_dir` helpers were added.

What was learned:

- path normalization should ideally live in one shared module from the beginning.

### 16.3 Problem: yfinance Output Shape Variability
Issue:

- `yfinance` may return a two-level column index even for a single ticker.

Why it happened:

- provider output format is not always a simple flat frame.

Fix:

- `data_loader.py` flattens columns and then keeps only the expected six fields.

What was learned:

- provider-specific quirks must be normalized immediately at ingestion time.

### 16.4 Problem: Dirty or Legacy CSV Rows
Issue:

- older CSVs may contain malformed rows, stray ticker lines, or bad numeric values.

Why it happened:

- artifacts can persist across earlier experiments and provider changes.

Fix:

- repeated `to_datetime(..., errors="coerce")` and `to_numeric(..., errors="coerce")`,
- then drop invalid rows.

What was learned:

- every stage should revalidate the columns it depends on instead of trusting upstream outputs blindly.

### 16.5 Problem: Look-Ahead Bias Risk
Issue:

- a strategy could unrealistically trade at the same close that triggered its signal.

Why it happened:

- this is a common backtesting mistake when rules are written directly against end-of-day data.

Fix:

- all agents evaluate the signal on the current bar and execute at the next bar's open.

What was learned:

- simple execution realism changes matter more than adding flashy indicators.

### 16.6 Problem: Open Trades at End of Sample
Issue:

- a trade can be open when the dataset ends.

Why it happened:

- signal logic may enter but never receive a closing rule before the final row.

Fix:

- incomplete trades are excluded from output.

What was learned:

- mixing open and completed trades makes downstream metric interpretation ambiguous.

### 16.7 Problem: Monte Carlo Could Be Misread If Only One Seed Is Used
Issue:

- one Monte Carlo run can give a deceptively stable-looking answer.

Why it happened:

- bootstrap results naturally depend on the random seed.

Fix:

- the project added `monte_carlo_robustness.py` with 30 outer runs and explicit seed-sweep analysis.

What was learned:

- robustness across seeds is not optional when effect sizes are small.

### 16.8 Problem: Histogram Scaling and Heavy-Tail Compression
Issue:

- Monte Carlo distributions can be heavily right-skewed,
- causing the visible body of the histogram to compress.

Why it happened:

- compounded returns can produce rare but very large simulated outcomes,
- especially when trade counts are high or return dispersion is large.

Fix:

- display-only percentile clipping,
- optional log1p display spacing,
- custom tick construction for transformed axes.

What was learned:

- chart readability can fail even when the underlying statistics are correct.

### 16.9 Problem: Mean Simulated Return Could Be Misleading
Issue:

- the Monte Carlo mean can be dominated by tail events.

Why it happened:

- compounded distributions are often skewed and heavy-tailed.

Fix:

- RCSI was built around the median simulated return,
- and the code also computes `RCSI_z` separately for standardized context.

What was learned:

- central tendency choice is not cosmetic; it changes the research conclusion.

### 16.10 Problem: Stale Summary Files vs Simulation Results
Issue:

- summary CSVs, result CSVs, and trade files can drift out of sync when only some steps are rerun.

Why it happened:

- research workflows often rerun only selected scripts,
- leaving some downstream outputs stale.

Fix:

- `monte_carlo_plot.py` re-computes and cross-checks:
  - simulation count,
  - medians,
  - means,
  - standard deviations,
  - percentiles,
  - p-values,
  - actual return,
  - trade count.

What was learned:

- plotting should validate the numbers it visualizes, not just trust summary files.

### 16.11 Problem: RCSI Charts Could Be Visually Dominated by One Strategy
Issue:

- one large RCSI bar can make the others unreadable.

Why it happened:

- RCSI magnitudes can differ dramatically across strategies and tickers.

Fix:

- the RCSI bar chart adds a zoom inset when the dominant ratio is large enough.

What was learned:

- a good research chart needs graceful handling of scale imbalance.

### 16.12 Problem: Heatmap Annotation Broke Under pandas 3
Issue:

- a heatmap annotation path relied on `DataFrame.applymap`, which is not available in the current `pandas==3.0.1` environment.

Why it happened:

- the API changed across pandas versions.

Fix:

- the heatmap implementation was refactored to use a direct NumPy/imshow workflow with manual annotation logic.

What was learned:

- chart annotation pipelines should avoid brittle convenience APIs when simple explicit loops are clearer and more version-stable.

### 16.13 Problem: Heatmap Tick Label Handling Produced Matplotlib Errors/Warnings
Issue:

- tick-label updates can break when locator positions and label counts drift.

Why it happened:

- Matplotlib is strict about fixed locator / fixed label alignment.

Fix:

- the heatmap now sets explicit tick positions and labels rather than relying on implicit state.

What was learned:

- explicit tick management is safer in matrix plots than post-hoc label mutation.

### 16.14 Problem: Equity Chart Was Not Appearing Reliably in the Combined Chart View
Issue:

- the equity chart was not reliably surfacing in the "all charts" output flow.

Why it happened:

- older behavior relied on Preview opening multiple separate PNGs,
- and stale per-strategy equity files also created confusion.

Fix:

- `equity_curve.py` now removes stale per-strategy equity PNGs,
- `open_charts.py` now builds one combined multi-page PDF and opens that single file.

What was learned:

- bundling research visuals into one real artifact is more reliable than asking the OS to group many loose files.

### 16.15 Problem: Buy-and-Hold Added Scope Creep to Strategy-Focused Outputs
Issue:

- benchmark data was useful historically, but it cluttered the active equity chart and comparison flow once the project focus narrowed to the three strategies.

Why it happened:

- benchmark comparison was added as a natural extension of the research,
- but later the main question became strategy-vs-randomness rather than strategy-vs-passive.

Fix:

- buy-and-hold was removed from the active pipeline outputs while the script itself was retained as an archival benchmark module.

What was learned:

- baseline breadth is useful early, but final presentation should match the core research question.

### 16.16 Problem: Naming Confusion Around the Master Pipeline Script
Issue:

- the requested conceptual `main.py` is not present as a committed file in the current tree.

Why it happened:

- the active orchestrator currently lives as `AAAmain.py`.

Fix:

- documentation must explicitly identify `AAAmain.py` as the live entrypoint.

What was learned:

- a pipeline entrypoint should have one stable canonical filename to reduce confusion.

---

## 17. Limitations

### 17.1 Simplified Strategy Logic
The strategies are deliberately simple.
That is good for interpretability, but it limits realism.

There is:

- no position sizing,
- no portfolio allocation model,
- no leverage,
- no shorting,
- no stop-loss framework,
- no partial exits,
- no cash management.

### 17.2 Simplified Execution Model
Execution is more realistic than same-close execution, but still simplified.

There is:

- no slippage,
- one fixed transaction cost,
- no bid/ask spread modeling,
- no liquidity constraints,
- no market impact.

### 17.3 Simplified Regime Model
Regimes are based only on:

- 20-day realized return volatility terciles.

This omits:

- trend persistence,
- volume regime,
- correlation regime,
- macro state,
- event risk,
- volatility clustering models.

### 17.4 Trade-Return Monte Carlo Assumptions
The Monte Carlo baseline assumes trade returns can be resampled with replacement.

That means it ignores:

- autocorrelation in trade outcomes,
- trade sequencing effects,
- regime clustering,
- time-varying opportunity set,
- dependence between entry logic and market phase.

### 17.5 No Annualization or Risk-Free Adjustment in Sharpe
The Sharpe ratio here is:

- trade-level average return divided by trade-level standard deviation.

It is not:

- annualized,
- time-normalized,
- or excess-over-risk-free.

So it should be interpreted as a relative internal metric, not a canonical portfolio Sharpe.

### 17.6 Historical Dependence
Everything is historical and backward-looking.

The project does not claim:

- out-of-sample predictive validity,
- forward profitability,
- or production readiness.

### 17.7 Ticker Dependence
The system is one-ticker-at-a-time.

That is useful for controlled analysis, but it means:

- cross-asset inference requires repeated runs,
- and conclusions must be stated carefully by ticker.

### 17.8 Benchmark Status
The buy-and-hold benchmark remains in the repository but is no longer part of the active output flow.

That means:

- the conceptual benchmark still exists,
- but the default presentation now emphasizes strategy-vs-randomness rather than strategy-vs-passive.

### 17.9 Artifact Staleness Risk
Because the repo contains historical outputs for several tickers, partial reruns can leave some artifacts stale.

This is partly mitigated by validation in the charting layer, but it remains an operational limitation unless the full pipeline is rerun consistently.

---

## 18. Final Interpretation

### 18.1 What the Current Results Show
Using the current fully rerun SPY outputs:

- trend has the highest current RCSI and is the only positive one,
- mean reversion and random are both negative on RCSI,
- but all three strategies have p-values near `0.5`,
- and all three are unstable under the current robustness classification.

This leads to a disciplined interpretation:

- the project currently finds weak evidence for strong skill in SPY,
- even though the trend strategy looks directionally better than the others on raw RCSI.

### 18.2 Which Strategies Have Real Skill?
The answer depends on how "skill" is defined.

If skill means:

- positive raw gap versus median Monte Carlo baseline,

then:

- trend currently shows the strongest evidence among the three in SPY.

If skill means:

- positive gap plus strong p-value evidence plus robustness across seeds,

then:

- none of the currently rerun SPY strategies clearly qualifies as strong skill.

That distinction is the whole point of the project.

### 18.3 How Randomness Plays a Role
Randomness matters in two separate ways.

#### A. Random Agent
The random strategy itself can look surprisingly good on standard performance metrics.

That shows:

- naive return metrics are not sufficient evidence of skill.

#### B. Monte Carlo Baseline
Even for non-random strategies, randomized resampling can generate a wide range of final outcomes.

That shows:

- some apparently impressive realized results may still be ordinary relative to the simulated baseline.

### 18.4 How Performance Depends on Market Conditions
From current SPY regime analysis:

- trend has positive but modest Sharpe ratios across calm, neutral, and stressed regimes.
- mean reversion is strongest in calm and neutral conditions and weakens materially in stressed conditions.
- random is positive across regimes in the current SPY artifact, which is precisely why control baselines matter.

The regime takeaway is:

- strategy behavior is not regime-invariant,
- and mean reversion appears especially regime-sensitive.

### 18.5 Cross-Ticker Note
Across the saved `*_rcsi.csv` artifacts currently in the repository:

- trend ranks highest on raw RCSI in every saved ticker file.

But that should be interpreted carefully because:

- not every ticker has a fully rerun, fully validated modern summary/robustness set in the current workspace,
- and raw RCSI alone is not enough without p-values and stability context.

So the broader conclusion is:

- trend often emerges as the strongest candidate,
- but the project's stricter evidence tests still demand caution before calling it robust skill.

---

## 19. Key Insights

### 19.1 Skill vs Luck Must Be Framed as a Distributional Question
A strategy's realized return means little by itself.
What matters is where that result sits inside a randomized outcome distribution.

### 19.2 Median-Based Comparison Is More Trustworthy Than Mean-Based Comparison in Skewed Simulations
The project repeatedly shows why:

- Monte Carlo means can be distorted by heavy tails,
- while medians remain a better "typical baseline."

### 19.3 Regime Dependency Is Real
Performance is not uniform across market states.
Strategies can appear strong overall while being concentrated in particular volatility conditions.

### 19.4 Random Controls Are Essential
The random agent demonstrates that:

- simple backtest metrics can flatter a no-skill process.

That is one of the strongest reasons the project exists in its current form.

### 19.5 Robustness Across Seeds Changes the Standard of Evidence
An effect that is positive in one Monte Carlo run may still be too fragile to trust.

### 19.6 Visualization Is Part of the Research, Not Just Presentation
The project had to solve:

- skewed histogram scaling,
- dominant-bar compression,
- heatmap annotation compatibility,
- combined artifact bundling,

because the conclusions are only useful if the evidence is interpretable.

### 19.7 The Current System Favors Transparency Over Complexity
This is a major strength.
Every major conclusion can be traced back to:

- a small set of features,
- simple strategy rules,
- explicit compounding math,
- and inspectable CSV artifacts.

---

## 20. Full Pipeline Summary

1. Load data
   - `data_loader.py` downloads full daily history from `yfinance`, flattens columns, cleans rows, and stores `Data_Raw/{ticker}.csv`.

2. Create features
   - `features.py` computes daily return, 20-day MA, 50-day MA, and 20-day rolling return volatility, then writes `{ticker}_features.csv`.

3. Detect regimes
   - `regimes.py` uses `pd.qcut` on `std_20` to label each day as calm, neutral, or stressed, then writes `{ticker}_regimes.csv`.

4. Run strategies
   - `trend_agent.py`, `mean_reversion_agent.py`, and `random_agent.py` create completed trade logs using next-bar execution.

5. Compute returns and trade metrics
   - each metrics script summarizes its trade file into total trades, mean/median return, win rate, standard deviation, Sharpe-like ratio, average win/loss, and expected value.

6. Simulate randomness
   - `monte_carlo.py` subtracts transaction cost, converts to log returns, bootstraps 5,000 trade paths per strategy, and stores raw result files plus a summary table.

7. Compare actual results to simulated baselines
   - the summary table computes median simulated return, mean simulated return, percentiles, p-values, and confidence-style bands.

8. Compute RCSI
   - `rcsi.py` calculates `actual_cumulative_return - median_simulated_return` and also computes `RCSI_z`.

9. Test robustness
   - `monte_carlo_robustness.py` reruns the simulation across 30 outer seeds, aggregates variability, and classifies stability.

10. Compare strategies
   - `compare_agents.py` builds the combined comparison table for trend, mean reversion, and random.

11. Analyze regime-specific performance
   - `regime_analysis.py` groups trades by agent and entry regime, computing per-regime performance metrics.

12. Generate visuals
   - the chart scripts render:
     - combined equity curves,
     - Monte Carlo histograms,
     - RCSI bars,
     - regime bars,
     - regime heatmap,
     - robustness charts,
     - and p-value bars.

13. Bundle outputs
   - `open_charts.py` converts the expected chart set into one combined PDF and opens it as the final research artifact.

---

## Closing Note
This project is not just a trading backtest.
It is a trading-strategy evaluation framework built around a sharper research question:

When a strategy appears to work, how much of that result looks like skill, how much looks like luck, and how stable is that conclusion once we account for randomness and market regime?

That framing is the real contribution of the system.
