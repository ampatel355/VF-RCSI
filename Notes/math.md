# Trading Research Project
## Mathematical Appendix and Formula Reference

> Archival note: this appendix was written before the March 30, 2026 audit refresh and the later same-day no-trade pipeline update. The live comparison layer now emphasizes annualized daily Sharpe, `RCSI_z`, p-value prominence, forward-only regimes, active buy-and-hold benchmarking, expanded walk-forward testing, and explicit handling of zero-trade cases. Use [README.md](/Users/aryanpatel/Downloads/Virtu_Fortuna_Project/README.md) and [paper.md](/Users/aryanpatel/Downloads/Virtu_Fortuna_Project/Notes/paper.md) for the current synchronized methodology.

## March 30, 2026 Live Addendum: No-Trade Conventions

The live code now includes an explicit mathematical convention for ticker-strategy pairs that produce no completed trades.

### 1. Daily-Curve Convention

If the completed trade count is zero, the active strategy equity curve is defined as:

```text
Equity_t = StartingCapital
```

for every day `t` in the market sample. Therefore:

```text
CumulativeReturn = 0
DailyReturn_t = 0
AnnualizedSharpe = 0
MaxDrawdown = 0
```

This is a mechanical path property, not inferential evidence.

### 2. Monte Carlo Compatibility Convention

For pipeline compatibility, the Monte Carlo layer stores a degenerate zero-return simulation distribution:

```text
S_m = 0  for all m = 1, ..., M
```

which implies:

```text
Median(S) = 0
Mean(S) = 0
Std(S) = 0
RCSI = 0
RCSI_z = 0
```

inside the raw compatibility files when actual return is also zero.

### 3. Comparison-Layer Convention

Although the raw compatibility files contain zeros, the active comparison layer suppresses inferential fields for no-trade strategies rather than presenting those zeros as meaningful evidence. In the live comparison table:

- `p_value = NA`
- `p_value_prominence = NA`
- `actual_percentile = NA`
- `RCSI = NA`
- `RCSI_z = NA`

The verdict engine then labels the case as:

```text
Evidence = No Trades
Verdict = No Trades
Confidence = Not Applicable
```

### 4. Interpretation Rule

A no-trade outcome is treated as:

- non-activity,
- non-inferable,
- and not evidence for either skill or luck.

This convention was added to support sparse-activity cases such as `EURUSD=X`, where the current long-only daily strategy set produced zero completed trades under the live execution constraints.

## Purpose
This document is the mathematical companion to the main project documentation. Its purpose is to state, precisely and explicitly:

- every major formula used in the trading research pipeline,
- what each symbol means,
- where each formula is used in the code,
- why the formula was chosen,
- and where the implementation uses different conventions in different modules.

This appendix is written from the live code in the current repository. It therefore documents the mathematics of the actual implementation, not an idealized version of what the system might do in theory.

---

## 1. Scope of the Math

The project uses mathematics in six main places:

1. Price-to-feature transformation
2. Regime labeling
3. Trading signal generation
4. Trade return and performance measurement
5. Monte Carlo resampling and statistical comparison
6. Visualization scaling and summary interpretation

The main source files behind this appendix are:

- `Code/data_loader.py`
- `Code/features.py`
- `Code/regimes.py`
- `Code/trend_agent.py`
- `Code/mean_reversion_agent.py`
- `Code/random_agent.py`
- `Code/trend_metrics.py`
- `Code/mean_reversion_metrics.py`
- `Code/random_metrics.py`
- `Code/compare_agents.py`
- `Code/monte_carlo.py`
- `Code/realistic_monte_carlo.py`
- `Code/rcsi.py`
- `Code/monte_carlo_robustness.py`
- `Code/regime_analysis.py`
- `Code/equity_curve.py`
- `Code/monte_carlo_plot.py`
- `Code/rcsi_plot.py`
- `Code/rcsi_heatmap.py`
- `Code/monte_carlo_robustness_plot.py`
- `Code/p_value_plot.py`

---

## 2. Notation

This appendix uses the following symbols.

### 2.1 Market Data

Let:

- `t` = trading day index
- `T` = final day index
- `O_t` = open price on day `t`
- `H_t` = high price on day `t`
- `L_t` = low price on day `t`
- `C_t` = close price on day `t`
- `V_t` = volume on day `t`

### 2.2 Rolling Feature Notation

Let:

- `r_t` = daily close-to-close return on day `t`
- `MA_20,t` = 20-day simple moving average of close
- `MA_50,t` = 50-day simple moving average of close
- `sigma_20,t` = 20-day rolling standard deviation of daily returns

### 2.3 Trade Notation

For trade `i`, let:

- `P_i^entry` = entry price
- `P_i^exit` = exit price
- `R_i` = raw trade return
- `R_i^adj` = transaction-cost-adjusted trade return
- `ell_i` = trade log return
- `N` = number of completed trades for a strategy

### 2.4 Monte Carlo Notation

Let:

- `M` = number of Monte Carlo simulations
- `S_m` = simulated final cumulative return in simulation `m`
- `A` = actual final cumulative return of the real strategy
- `Med(S)` = median of the simulated final return distribution
- `Mean(S)` = mean of the simulated final return distribution
- `Std(S)` = standard deviation of the simulated final return distribution

### 2.5 Regime Notation

Let:

- `Regime_t ∈ {calm, neutral, stressed}`
- `q_1` = first tercile cut point of the `sigma_20` distribution
- `q_2` = second tercile cut point of the `sigma_20` distribution

---

## 3. Raw Data Mathematics

### 3.1 Raw Data Columns
The raw ingestion step does not perform deep mathematics. It standardizes the data structure so later math can be trusted.

The project keeps:

- `Date`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

The first true numeric transformation happens in feature engineering.

---

## 4. Feature Engineering Mathematics

File:

- `Code/features.py`

### 4.1 Daily Return

Formula:

```text
r_t = (C_t / C_(t-1)) - 1
```

Implementation:

- `df["daily_return"] = df["Close"].pct_change()`

Use:

- this is the base return series for volatility estimation,
- and it expresses one-day percentage movement in the close.

Conceptual role:

- this is the simplest scale-free representation of daily price change.

### 4.2 20-Day Simple Moving Average

Formula:

```text
MA_20,t = (1 / 20) * sum_{j=0 to 19} C_(t-j)
```

Implementation:

- `df["ma_20"] = df["Close"].rolling(window=20).mean()`

Use:

- this is the equilibrium line for the mean-reversion strategy.

### 4.3 50-Day Simple Moving Average

Formula:

```text
MA_50,t = (1 / 50) * sum_{j=0 to 49} C_(t-j)
```

Implementation:

- `df["ma_50"] = df["Close"].rolling(window=50).mean()`

Use:

- this is the trend filter for the trend-following strategy.

### 4.4 20-Day Rolling Volatility of Returns

Formula:

```text
sigma_20,t = std(r_(t-19), r_(t-18), ..., r_t)
```

Implementation:

- `df["std_20"] = df["daily_return"].rolling(window=20).std()`

Important implementation note:

- `pandas.Series.rolling(...).std()` uses sample standard deviation by default, which means `ddof = 1`.

So the actual implementation is:

```text
sigma_20,t = sqrt( (1 / (20 - 1)) * sum_{j=0 to 19} (r_(t-j) - mean_r_t)^2 )
```

where:

```text
mean_r_t = (1 / 20) * sum_{j=0 to 19} r_(t-j)
```

Use:

- regime classification,
- mean-reversion threshold construction.

### 4.5 Feature Table Truncation

Because moving averages and rolling volatility need lookback data, the feature table discards the early rows where the windows are incomplete.

Mathematically:

- any row where one of the rolling quantities is undefined is removed.

Implementation:

- `df = df.dropna().reset_index(drop=True)`

---

## 5. Regime Detection Mathematics

File:

- `Code/regimes.py`

### 5.1 Volatility Terciles
Regimes are defined by splitting the distribution of `sigma_20,t` into three quantile bins.

Let:

```text
q_1 = 33.33rd percentile of {sigma_20,t}
q_2 = 66.67th percentile of {sigma_20,t}
```

Then:

```text
Regime_t = calm      if sigma_20,t <= q_1
Regime_t = neutral   if q_1 < sigma_20,t <= q_2
Regime_t = stressed  if sigma_20,t > q_2
```

Implementation:

- `pd.qcut(df["std_20"], q=3, labels=["calm", "neutral", "stressed"])`

### 5.2 Interpretation
The regime model is not absolute-volatility classification.
It is relative to the ticker's own empirical volatility distribution.

So:

- "calm" means lower relative realized volatility,
- "stressed" means higher relative realized volatility.

### 5.3 Mathematical Limitation
This is a univariate state model:

```text
Regime_t = f(sigma_20,t)
```

It does not depend on:

- trend slope,
- drawdown state,
- volume shock,
- macro state,
- or transition probability.

---

## 6. Strategy Signal Mathematics

Files:

- `Code/trend_agent.py`
- `Code/mean_reversion_agent.py`
- `Code/random_agent.py`

All strategies use the same execution timing rule:

Signal is observed on day `t`, but execution occurs at day `t+1` open.

That means the trading loop is mathematically defined only for:

```text
t = 1, 2, ..., T - 1
```

because the next bar must exist.

### 6.1 Trend Strategy

#### Entry Signal

```text
Enter if C_t > MA_50,t and no position is open
```

Execution:

```text
P_i^entry = O_(t+1)
```

#### Exit Signal

```text
Exit if C_t < MA_50,t and a position is open
```

Execution:

```text
P_i^exit = O_(t+1)
```

#### Indicator Logic
This is a binary moving-average state model:

```text
Signal_t^trend = 1 if C_t > MA_50,t
Signal_t^trend = 0 otherwise
```

The strategy is effectively long when the signal flips on and exits when the signal flips off.

### 6.2 Mean-Reversion Strategy

#### Entry Threshold
As implemented in code:

```text
Threshold_t^MR = MA_20,t - 2 * sigma_20,t
```

Entry rule:

```text
Enter if C_t < Threshold_t^MR and no position is open
```

Execution:

```text
P_i^entry = O_(t+1)
```

#### Exit Rule

```text
Exit if C_t >= MA_20,t and a position is open
```

Execution:

```text
P_i^exit = O_(t+1)
```

#### Important Mathematical Caveat
This is the exact implemented formula, but it mixes units:

- `MA_20,t` is in price units,
- `sigma_20,t` is in return units.

So the code literally subtracts return volatility from a price-level moving average.

That means the implementation is:

```text
price threshold = price average - 2 * return volatility
```

which is dimensionally simplified rather than fully consistent.

A more mathematically consistent mean-reversion rule would typically use one of:

- price standard deviation,
- Bollinger-style price bands,
- or return z-scores applied in return space.

But this appendix documents the actual implementation, and the actual implementation is:

```text
MA_20,t - 2 * sigma_20,t
```

### 6.3 Random Strategy

Parameters:

```text
p_entry = 0.05
p_exit = 0.05
```

#### Entry Rule
If no position is open, draw a random number `U_t` from Uniform(0, 1):

```text
Enter if U_t < p_entry
```

Execution:

```text
P_i^entry = O_(t+1)
```

#### Exit Rule
If a position is open, draw a random number `V_t` from Uniform(0, 1):

```text
Exit if V_t < p_exit
```

Execution:

```text
P_i^exit = O_(t+1)
```

#### Reproducibility Mode
If reproducibility is enabled, the random stream is pseudo-random with a fixed seed.
Otherwise:

- `SystemRandom()` is used,
- making the strategy nondeterministic across runs.

---

## 7. Trade Construction Mathematics

Files:

- all three agent scripts

### 7.1 Raw Trade Return
For completed trade `i`:

```text
R_i = (P_i^exit - P_i^entry) / P_i^entry
```

This is a simple gross long return.

### 7.2 Regime-at-Entry Mapping
The regime assigned to a trade is the regime from the signal day, not the execution day.

So if the signal occurs on day `t`, then:

```text
Regime_i^entry = Regime_t
```

even though:

```text
P_i^entry = O_(t+1)
```

This avoids attaching future regime information to the trade record.

### 7.3 Trade Count
For a strategy:

```text
N = number of completed trades
```

Open but unfinished trades at the end of the sample are excluded.

---

## 8. Trade-Level Performance Metric Mathematics

Files:

- `Code/trend_metrics.py`
- `Code/mean_reversion_metrics.py`
- `Code/random_metrics.py`

These scripts operate on raw trade returns `R_i`, not transaction-cost-adjusted returns.

### 8.1 Total Trades

```text
total_trades = N
```

### 8.2 Average Return

```text
average_return = (1 / N) * sum_{i=1 to N} R_i
```

### 8.3 Median Return

```text
median_return = median(R_1, R_2, ..., R_N)
```

### 8.4 Win Rate

```text
win_rate = (1 / N) * sum_{i=1 to N} 1[R_i > 0]
```

where `1[condition]` is the indicator function.

### 8.5 Standard Deviation of Trade Returns
In the metric scripts, `pandas.Series.std()` is used with its default behavior:

- sample standard deviation,
- `ddof = 1`.

So:

```text
std_return = sqrt( (1 / (N - 1)) * sum_{i=1 to N} (R_i - average_return)^2 )
```

when `N > 1`, else `0`.

### 8.6 Sharpe-Like Ratio

```text
sharpe_ratio = average_return / std_return
```

if:

```text
std_return != 0
```

otherwise:

```text
sharpe_ratio = 0
```

Important note:

- this is a trade-level Sharpe-like statistic,
- not an annualized portfolio Sharpe ratio.

### 8.7 Average Win
Let:

```text
W = {R_i : R_i > 0}
```

Then:

```text
average_win = mean(W)
```

If `W` is empty, the code returns `0`.

### 8.8 Average Loss
Let:

```text
L = {R_i : R_i < 0}
```

Then:

```text
average_loss = abs(mean(L))
```

The absolute value is taken so the loss size is reported as a positive magnitude.

### 8.9 Expected Value

```text
expected_value = (win_rate * average_win) - ((1 - win_rate) * average_loss)
```

This is the standard trade expectancy identity:

```text
expectancy = P(win) * avg_win - P(loss) * avg_loss
```

---

## 9. Strategy Comparison Table Mathematics

File:

- `Code/compare_agents.py`

This file uses different math conventions from the simple metrics scripts.

That distinction is important.

### 9.1 Transaction-Cost-Adjusted Trade Returns
The comparison table uses:

```text
R_i^adj = R_i - c
```

where:

```text
c = 0.001
```

### 9.2 Trade Log Return

```text
ell_i = ln(1 + R_i^adj)
```

Constraint:

```text
1 + R_i^adj > 0
```

or equivalently:

```text
R_i^adj > -1
```

### 9.3 Wealth Index Path

```text
W_k = exp( sum_{i=1 to k} ell_i )
```

for each trade step `k`.

### 9.4 Cumulative Return

```text
C_k = W_k - 1
```

Final cumulative return in the comparison table is:

```text
cumulative_return = C_N
```

### 9.5 Rolling Peak

```text
Peak_k = max(W_1, W_2, ..., W_k)
```

### 9.6 Drawdown

```text
Drawdown_k = (W_k / Peak_k) - 1
```

### 9.7 Maximum Drawdown

```text
max_drawdown = min_k Drawdown_k
```

### 9.8 Comparison-Table Sharpe
The comparison table uses adjusted returns and NumPy population standard deviation:

```text
sharpe_ratio = mean(R_i^adj) / std_pop(R_i^adj)
```

where:

```text
std_pop(R_i^adj) = sqrt( (1 / N) * sum_{i=1 to N} (R_i^adj - mean(R^adj))^2 )
```

This differs from the simple metrics scripts, which use:

- raw returns,
- sample standard deviation.

That means the `*_metrics.csv` Sharpe ratio and the `*_full_comparison.csv` Sharpe ratio are not mathematically identical objects.

### 9.9 Number of Trades

```text
number_of_trades = N
```

### 9.10 RCSI and p-Value Columns
These are copied from upstream Monte Carlo outputs.

---

## 10. Buy-and-Hold Benchmark Mathematics

File:

- `Code/buy_and_hold.py`

This script is still present in the repository even though it is no longer part of the active default pipeline output bundle.

### 10.1 Daily Benchmark Return

```text
r_t^BH = (C_t / C_(t-1)) - 1
```

Implementation:

- `curve_df["daily_return"] = curve_df["Close"].pct_change().fillna(0.0)`

### 10.2 Wealth Index

```text
W_t^BH = C_t / C_0
```

where `C_0` is the first close in the sample.

If benchmark transaction cost `c_BH > 0`, then:

```text
W_t^BH = (C_t / C_0) * (1 - c_BH)
```

Current implementation:

```text
c_BH = 0
```

### 10.3 Cumulative Return

```text
C_t^BH = W_t^BH - 1
```

### 10.4 Rolling Peak and Drawdown

```text
Peak_t^BH = max(W_1^BH, ..., W_t^BH)
Drawdown_t^BH = (W_t^BH / Peak_t^BH) - 1
max_drawdown^BH = min_t Drawdown_t^BH
```

### 10.5 Buy-and-Hold Sharpe-Like Ratio
This script uses daily returns and NumPy population standard deviation:

```text
sharpe_ratio^BH = mean(r_t^BH) / std_pop(r_t^BH)
```

where:

```text
std_pop(r_t^BH) = sqrt( (1 / T) * sum (r_t^BH - mean(r^BH))^2 )
```

---

## 11. Equity Curve Mathematics

File:

- `Code/equity_curve.py`

The equity curve chart uses adjusted trade returns, not raw trade returns.

### 11.1 Cost-Adjusted Return

```text
R_i^adj = R_i - c
```

with:

```text
c = TRANSACTION_COST = 0.001
```

### 11.2 Trade Log Return

```text
ell_i = ln(1 + R_i^adj)
```

### 11.3 Cumulative Log Return

```text
L_k = sum_{i=1 to k} ell_i
```

### 11.4 Wealth Index

```text
W_k = exp(L_k)
```

### 11.5 Cumulative Return

```text
C_k = W_k - 1
```

This is the series plotted on the y-axis.

### 11.6 Final Marker
The chart places a point at:

```text
(exit_date_N, C_N)
```

for each strategy.

### 11.7 Display Range Padding
If:

```text
y_min = min_k C_k
y_max = max_k C_k
y_span = max(y_max - y_min, 0.5)
y_padding = 0.14 * y_span
```

then the displayed axis limits are:

```text
lower = min(y_min - y_padding, -0.25 * y_padding)
upper = max(y_max + y_padding,  0.25 * y_padding)
```

This is display math, not strategy math, but it directly affects interpretation of the chart.

---

## 12. Monte Carlo Core Mathematics

File:

- `Code/monte_carlo.py`

### 12.1 Adjusted Trade Returns

```text
R_i^adj = R_i - c
```

where:

```text
c = 0.001
```

### 12.2 Validity Constraint for Log Conversion

```text
R_i^adj > -1
```

This is required because:

```text
ln(1 + R_i^adj)
```

must be defined.

### 12.3 Log Returns

```text
ell_i = ln(1 + R_i^adj)
```

### 12.4 Actual Strategy Cumulative Return
The actual realized strategy result is:

```text
A = exp( sum_{i=1 to N} ell_i ) - 1
```

### 12.5 Resampling Scheme
For each simulation `m = 1, 2, ..., M`, with `M = 5000`:

Sample indices:

```text
K_(m,1), K_(m,2), ..., K_(m,N)
```

independently and uniformly from:

```text
{1, 2, ..., N}
```

with replacement.

Then simulated log return sum is:

```text
L_m^sim = sum_{j=1 to N} ell_(K_(m,j))
```

and simulated cumulative return is:

```text
S_m = exp(L_m^sim) - 1
```

### 12.6 Why the Simulated Path Length Equals the Actual Trade Count
The code resamples exactly `N` trades per simulation.

That means the simulated baseline is conditional on:

- the strategy's actual trade count.

Mathematically:

```text
length(simulated path) = N
```

This is a fairer apples-to-apples comparison than changing both:

- trade count,
- and return ordering.

### 12.7 Monte Carlo Summary Statistics
Given the set:

```text
S = {S_1, S_2, ..., S_M}
```

the project computes:

#### Median Simulated Return

```text
Med(S) = median(S)
```

#### Mean Simulated Return

```text
Mean(S) = (1 / M) * sum_{m=1 to M} S_m
```

#### Simulated Standard Deviation
The code uses NumPy population standard deviation:

```text
Std(S) = sqrt( (1 / M) * sum_{m=1 to M} (S_m - Mean(S))^2 )
```

because:

- `np.std(..., ddof=0)` is used.

#### Lower and Upper Simulation Bands

```text
lower_5pct = 5th percentile of S
upper_95pct = 95th percentile of S
```

### 12.8 Actual Percentile
The actual percentile is:

```text
actual_percentile = 100 * (1 / M) * sum_{m=1 to M} 1[S_m <= A]
```

Interpretation:

- the percentage of simulated outcomes that are less than or equal to the actual strategy outcome.

### 12.9 p-Value
The project uses a one-sided right-tail p-value:

```text
p_value = (1 / M) * sum_{m=1 to M} 1[S_m >= A]
```

Interpretation:

- the probability, under the simulated baseline, of seeing an outcome at least as large as the actual one.

### 12.10 Evidence Label Mapping

```text
if p_value < 0.05: label = "strong evidence"
elif p_value < 0.10: label = "weak evidence"
else: label = "no evidence"
```

---

## 13. Alternate Monte Carlo Mathematics

File:

- `Code/realistic_monte_carlo.py`

This file expresses a mathematically similar simulation using a more explicit loop-based structure.

Its core formulas are the same:

```text
R_i^adj = R_i - c
ell_i = ln(1 + R_i^adj)
A = exp(sum ell_i) - 1
S_m = exp(sum sampled ell_i) - 1
```

The main difference is implementation style rather than underlying mathematics.

---

## 14. RCSI Mathematics

File:

- `Code/rcsi.py`

### 14.1 Primary Definition

```text
RCSI = A - Med(S)
```

where:

- `A` = actual cumulative return
- `Med(S)` = median simulated cumulative return

### 14.2 Standardized RCSI

```text
RCSI_z = (A - Mean(S)) / Std(S)
```

if:

```text
Std(S) > 0
```

otherwise:

```text
RCSI_z = NaN
```

### 14.3 Interpretation

```text
RCSI > 0  => actual return exceeded typical simulated outcome
RCSI = 0  => actual return matched typical simulated outcome
RCSI < 0  => actual return underperformed typical simulated outcome
```

### 14.4 Why the Median Is the Reference Baseline
The project uses:

```text
Med(S)
```

instead of:

```text
Mean(S)
```

because Monte Carlo compounded-return distributions can be strongly right-skewed.

The median is therefore a more robust estimate of the "typical" simulated outcome.

---

## 15. Regime Analysis Mathematics

File:

- `Code/regime_analysis.py`

The regime analysis groups trades by:

```text
(agent, regime_at_entry)
```

For each group `G`, it computes:

### 15.1 Group Trade Count

```text
total_trades(G) = |G|
```

### 15.2 Group Average Return

```text
average_return(G) = mean(R_i for i in G)
```

### 15.3 Group Median Return

```text
median_return(G) = median(R_i for i in G)
```

### 15.4 Group Win Rate

```text
win_rate(G) = mean(1[R_i > 0] for i in G)
```

### 15.5 Group Return Standard Deviation
This uses pandas sample standard deviation:

```text
std_return(G) = sample_std(R_i for i in G)
```

### 15.6 Group Sharpe-Like Ratio

```text
sharpe_ratio(G) = average_return(G) / std_return(G)
```

if:

```text
std_return(G) != 0
```

otherwise:

```text
sharpe_ratio(G) = 0
```

### 15.7 Matrix Construction for Charts
The grouped result is pivoted into a matrix:

```text
H[a, r] = sharpe_ratio for agent a in regime r
```

where:

- `a ∈ {trend, mean_reversion, random}`
- `r ∈ {calm, neutral, stressed}`

This matrix powers:

- the grouped Sharpe bar chart,
- and the regime heatmap.

---

## 16. Monte Carlo Robustness Mathematics

File:

- `Code/monte_carlo_robustness.py`

### 16.1 Outer Seed Sweep
Parameters:

```text
R = OUTER_RUNS = 30
M = SIMULATIONS_PER_RUN = 5000
BASE_SEED = 100
```

For outer run `r`:

```text
seed_r = BASE_SEED + r
```

Each outer seed generates child seeds for each strategy using `SeedSequence`.

### 16.2 Per-Run Statistics
For each strategy and each outer run:

- `A_r` = actual cumulative return
- `Med(S_r)` = median simulated return
- `Mean(S_r)` = mean simulated return
- `Std(S_r)` = std simulated return
- `Percentile_r`
- `p_r`
- `RCSI_r = A_r - Med(S_r)`
- `RCSI_z,r = (A_r - Mean(S_r)) / Std(S_r)`

### 16.3 Aggregated Means
Over the outer runs:

```text
mean_percentile = (1 / R) * sum Percentile_r
mean_p_value = (1 / R) * sum p_r
mean_RCSI = (1 / R) * sum RCSI_r
```

### 16.4 Aggregated Standard Deviations
The aggregation uses pandas groupby standard deviation, which is sample standard deviation unless otherwise overridden.

So:

```text
std_percentile = sample_std(Percentile_r)
std_p_value = sample_std(p_r)
std_RCSI = sample_std(RCSI_r)
```

with missing one-point std values filled as `0`.

### 16.5 Range Statistics

```text
min_percentile = min_r Percentile_r
max_percentile = max_r Percentile_r
min_p_value = min_r p_r
max_p_value = max_r p_r
min_RCSI = min_r RCSI_r
max_RCSI = max_r RCSI_r
```

### 16.6 Stability Classification
Let:

```text
percentile_range = max_percentile - min_percentile
scale_RCSI = max(|mean_RCSI|, 0.05)
relative_std_RCSI = std_RCSI / scale_RCSI
```

Then:

```text
stable
    if percentile_range <= 5.0
    and relative_std_RCSI <= 0.10

moderately variable
    if percentile_range <= 15.0
    and relative_std_RCSI <= 0.25

unstable
    otherwise
```

### 16.7 Why the Floor of 0.05 Exists
The denominator in relative RCSI variability uses:

```text
scale_RCSI = max(|mean_RCSI|, 0.05)
```

instead of just:

```text
|mean_RCSI|
```

to avoid exploding the relative-variability statistic when mean RCSI is extremely close to zero.

---

## 17. p-Value Visualization Mathematics

File:

- `Code/p_value_plot.py`

The chart itself is simple, but it formalizes one threshold.

### 17.1 Significance Threshold

```text
alpha = 0.05
```

This is drawn as a horizontal reference line.

### 17.2 Display Limits
If the strategy p-values are:

```text
p_1, p_2, p_3
```

then the chart upper bound is:

```text
upper = min(max(max(p_i) * 1.25, 0.12), 1.0)
```

This is not inferential math, but it affects how the evidence threshold is displayed.

---

## 18. Monte Carlo Histogram Display Mathematics

File:

- `Code/monte_carlo_plot.py`

This file contains some of the most detailed display logic in the project.

### 18.1 Display Clipping Quantiles
The chart may clip extreme tails for display only.

Default clip range:

```text
(0.5th percentile, 99.5th percentile)
```

Tighter clip range:

```text
(1st percentile, 99th percentile)
```

### 18.2 Tail Jump Test
Define:

```text
q_99 = Q_0.99(S)
q_995 = Q_0.995(S)
median_S = median(S)
tail_jump = |q_995 - q_99|
comparison_scale = max(|q_99|, |median_S|, 1.0)
```

If:

```text
tail_jump / comparison_scale > 0.75
```

then the tighter clipping window is used.

### 18.3 Display Window
Given chosen quantile bounds:

```text
lower = Q_low(S)
upper = Q_high(S)
```

the display subset is:

```text
S_display = {s in S : lower <= s <= upper}
```

The actual and median markers are still guaranteed to be included in the display span:

```text
display_min = min(lower, A, Med(S))
display_max = max(upper, A, Med(S))
```

### 18.4 Log1p Display Decision
The chart may use log1p spacing when linear spacing is too compressed.

Let:

```text
Q1 = 25th percentile of S_display
Q3 = 75th percentile of S_display
IQR = max(Q3 - Q1, 1e-9)
span = display_max - display_min
```

Use log1p display if:

```text
display_min > -0.999999
and span / IQR > 18
```

### 18.5 Log1p Display Transform
If log1p display is used:

```text
x_display = ln(1 + x_actual)
```

for all displayed x-values.

This is a display transform only.
The underlying simulation values are not altered.

### 18.6 Histogram Bin Count
The histogram uses Freedman-Diaconis suggested bin edges, then constrains the result:

```text
bins = min(45, max(24, suggested_bins))
```

where `suggested_bins` comes from `np.histogram_bin_edges(..., bins="fd")`.

This is a balance between:

- responsiveness to data shape,
- and visual stability across tickers.

---

## 19. RCSI Chart Display Mathematics

File:

- `Code/rcsi_plot.py`

### 19.1 Symmetric Axis Limits Around Zero
Let:

```text
M = max(|RCSI_1|, |RCSI_2|, |RCSI_3|)
```

If `M = 0`, the code sets:

```text
M = 0.1
```

Then:

```text
limit = 1.22 * M
y_lower = -limit
y_upper =  limit
```

This keeps the RCSI chart honest around the zero baseline.

### 19.2 Label Offset

```text
label_offset = max(0.02 * M, 0.008)
```

### 19.3 Dominant-Bar Ratio
To determine whether a zoom inset is needed, define:

```text
dominant_ratio = largest_abs_RCSI / second_largest_abs_RCSI
```

If:

```text
dominant_ratio >= 8
```

then the chart adds a zoom inset for the smaller bars.

---

## 20. Heatmap Mathematics

File:

- `Code/rcsi_heatmap.py`

### 20.1 Heatmap Data Matrix
The plotted matrix is:

```text
H[a, r] = mean Sharpe ratio for agent a in regime r
```

where the mean is whatever value exists in the grouped table after the pivot.

### 20.2 Color Normalization
If the heatmap contains both positive and negative values:

```text
min(H) < 0 < max(H)
```

then the color normalization is centered at zero.

Define:

```text
L = max(|min(H)|, |max(H)|)
```

Then the color scale is:

```text
vmin = -L
vcenter = 0
vmax = L
```

This is implemented with `TwoSlopeNorm`.

### 20.3 Annotation Logic
Each cell displays:

```text
H[a, r]
```

rounded to two decimals.

If the cell is missing:

```text
text = "NA"
```

The text color is chosen relative to the mean finite value of the matrix:

```text
midpoint = mean of finite entries
```

If:

```text
cell_value < midpoint
```

then white text is used, otherwise dark text is used.

This is a contrast heuristic, not inferential math.

---

## 21. Robustness Chart Mathematics

File:

- `Code/monte_carlo_robustness_plot.py`

### 21.1 RCSI Stability Chart
Each bar height is:

```text
mean_RCSI
```

Each error bar is:

```text
± std_RCSI
```

So the displayed interval is:

```text
[mean_RCSI - std_RCSI, mean_RCSI + std_RCSI]
```

### 21.2 Percentile Stability Chart
Each bar height is:

```text
mean_percentile
```

Each error bar is:

```text
± std_percentile
```

### 21.3 Label Placement
For either chart, if:

```text
mean = bar height
std = error size
span = max(means + stds) - min(means - stds)
offset = 0.04 * max(span, 1.0)
```

then labels are placed at:

```text
mean + std + offset     if mean >= 0
mean - std - offset     if mean < 0
```

---

## 22. Combined PDF Output Mathematics

File:

- `Code/open_charts.py`

There is no inferential mathematics here, but there is a compositing rule:

- the pipeline chart set is ordered as a list of expected output images,
- each image is converted to RGB,
- the first image becomes PDF page 1,
- the remaining images are appended as later pages.

This is artifact-assembly math, not financial math.

---

## 23. Important Implementation Differences in Statistical Conventions

This is one of the most important sections in the appendix.
The project does not use exactly one standard deviation convention everywhere.

### 23.1 Sample Standard Deviation Is Used In

- `features.py` rolling `std_20`
- `trend_metrics.py`
- `mean_reversion_metrics.py`
- `random_metrics.py`
- `regime_analysis.py`
- pandas groupby standard deviations in robustness aggregation

This means:

```text
std_sample(x) = sqrt( (1 / (n - 1)) * sum (x_i - mean(x))^2 )
```

### 23.2 Population Standard Deviation Is Used In

- `monte_carlo.py` simulated return standard deviation
- `compare_agents.py` strategy Sharpe denominator
- `buy_and_hold.py` daily-return Sharpe denominator

This means:

```text
std_pop(x) = sqrt( (1 / n) * sum (x_i - mean(x))^2 )
```

### 23.3 Why This Matters
Two values labeled "standard deviation" or "Sharpe ratio" in different files may not be numerically identical even if they are based on related data.

That is not a bug by itself.
It is a consequence of:

- different intended uses,
- different code paths,
- and the fact that the project evolved over time.

But it is mathematically important and should not be ignored.

---

## 24. Mathematical Caveats and Model Limits

### 24.1 Mean-Reversion Entry Formula Uses Mixed Units
As implemented:

```text
Threshold_t^MR = MA_20,t - 2 * sigma_20,t
```

This combines:

- a price-level average,
- with a return-level standard deviation.

That is a simplification.

### 24.2 Monte Carlo Resamples Trade Returns, Not Market Paths
The simulation does not model:

- stochastic price evolution,
- regime transition probability,
- serial dependence in market structure.

It models:

- randomized recombination of observed trade returns.

### 24.3 Sharpe Ratio Is Not Annualized
Every Sharpe-like quantity in the project is internal to the project and should be interpreted as:

- a dispersion-adjusted return signal,

not:

- a strict institutional Sharpe ratio.

### 24.4 Random Agent Is Not Deterministic by Default
If reproducibility mode is off, then:

- the random strategy itself changes across runs,
- which means downstream outputs tied to it can also change across runs.

### 24.5 RCSI Is an Effect-Size Style Metric, Not a Formal Test Statistic
RCSI is:

```text
A - Med(S)
```

It is very useful, but it is not a standalone significance test.
That is why it is paired with:

- p-values,
- actual percentiles,
- and robustness analysis.

---

## 25. Formula Map by File

This section is a compact cross-reference.

### `Code/features.py`

- `r_t = C_t / C_(t-1) - 1`
- `MA_20,t`
- `MA_50,t`
- `sigma_20,t = rolling std of r_t`

### `Code/regimes.py`

- `Regime_t = qcut(sigma_20,t, 3 bins)`

### `Code/trend_agent.py`

- enter if `C_t > MA_50,t`
- exit if `C_t < MA_50,t`
- `R_i = (P_i^exit - P_i^entry) / P_i^entry`

### `Code/mean_reversion_agent.py`

- `Threshold_t = MA_20,t - 2 * sigma_20,t`
- enter if `C_t < Threshold_t`
- exit if `C_t >= MA_20,t`
- `R_i = (P_i^exit - P_i^entry) / P_i^entry`

### `Code/random_agent.py`

- enter if `U_t < 0.05`
- exit if `V_t < 0.05`
- `R_i = (P_i^exit - P_i^entry) / P_i^entry`

### `Code/*_metrics.py`

- mean, median, win rate
- sample standard deviation
- `Sharpe = mean / std`
- `EV = p_win * avg_win - p_loss * avg_loss`

### `Code/compare_agents.py`

- `R_i^adj = R_i - 0.001`
- `ell_i = ln(1 + R_i^adj)`
- `W_k = exp(sum ell_i)`
- `C_k = W_k - 1`
- `Drawdown_k = W_k / Peak_k - 1`

### `Code/monte_carlo.py`

- bootstrap resampling of `ell_i`
- `A = exp(sum ell_i) - 1`
- `S_m = exp(sum sampled ell_i) - 1`
- percentile and p-value formulas

### `Code/rcsi.py`

- `RCSI = A - Med(S)`
- `RCSI_z = (A - Mean(S)) / Std(S)`

### `Code/monte_carlo_robustness.py`

- repeated outer-seed aggregation
- stability classification formulas

### `Code/buy_and_hold.py`

- `W_t = C_t / C_0`
- `C_t = W_t - 1`
- benchmark drawdown and Sharpe-like ratio

---

## Closing Note
The project is mathematically simple in its building blocks, but not simplistic in its structure. The important idea is not any one formula by itself. The important idea is how the formulas connect:

- raw prices become engineered state variables,
- state variables generate strategy trades,
- trades become compounded outcomes,
- compounded outcomes are tested against a simulated randomness baseline,
- the gap between actual and typical simulated outcome becomes RCSI,
- and seed-robustness determines whether the apparent edge is stable enough to trust.

That chain is the mathematical backbone of the entire research system.
