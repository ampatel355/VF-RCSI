# Title Page

**Virtu, Fortuna, and the Statistical Evaluation of Rule-Based Trading Strategies**

**Author:** Aryan Patel  
**Institution:** [Institution Name]  
**Course:** [Course Name]  
**Date:** March 30, 2026

## Abstract

This paper evaluates whether simple trading strategies display evidence of genuine skill or whether their observed performance can be explained by randomness. The project is conceptually inspired by Niccolo Machiavelli's distinction between *virtu* and *fortuna* in *The Prince*. In that framework, *virtu* denotes deliberate capability and effective action, whereas *fortuna* denotes luck, contingency, and forces beyond direct control. Applied to trading, the relevant question is not whether a strategy made money, but whether its outcome stands out from what chance alone could plausibly generate. To address that question, this study tests five long-only strategies, trend, mean reversion, random, momentum, and breakout, plus a passive buy-and-hold benchmark, under a common execution model that includes next-bar fills, slippage, spread costs, commissions, finite capital, integer share sizing, and liquidity caps. Statistical inference is based on a matched random-timing Monte Carlo null that preserves realized trade count, holding durations, and capital-at-risk structure while randomizing trade timing on the observed market path. The framework reports cumulative return, annualized return, annualized Sharpe ratio from daily equity curves, trade-level return ratio, maximum drawdown, percentile rank, p-value, raw Regime-Conditional Skill Index (RCSI), and standardized RCSI. It further applies repeated-run robustness testing, Benjamini-Hochberg false-discovery-rate control, regime-conditioned analysis with minimum-sample suppression, explicit no-trade handling, and multi-asset walk-forward testing. The current March 30, 2026 results show several ticker-specific positive cases, including SPY and VOO mean reversion, QQQ momentum, and multiple BTC-USD strategies. However, none of the 85 active ticker-strategy tests survives false-discovery control at q <= 0.10, and the broader 1,120-panel walk-forward study does not support a universal edge for any strategy. The project therefore finds asset-conditional historical alignment, not generalizable proof of persistent trading skill.

## Introduction

Backtests are often granted more evidentiary weight than they deserve. If a strategy produces a rising equity curve, the result is frequently interpreted as proof of insight, forecasting ability, or structural edge. That inference is weak. A realized profit path can emerge from favorable sequencing, persistent drift, concentrated exposure to a supportive regime, or repeated experimentation on the same historical sample. Profitability alone does not distinguish skill from luck.

The conceptual starting point of this project comes from Machiavelli's treatment of *virtu* and *fortuna* in *The Prince*. Machiavelli did not equate success with mastery. He argued instead that outcomes reflect both deliberate capacity and the pressure of contingency. This distinction maps naturally onto financial markets. A strategy may appear successful because it reflects genuine decision-making structure, which is the analog of *virtu*. It may also appear successful because randomness happened to be favorable, which is the analog of *fortuna*. If one observes only the final outcome, those two possibilities can look identical.

This project was designed to separate them. The research question is straightforward: when simple rule-based strategies are tested under realistic execution, matched Monte Carlo null models, repeated-run robustness analysis, false-discovery-rate control, regime-conditioned summaries, and fold-local walk-forward evaluation, do any of them demonstrate reproducible evidence of skill rather than randomness?

The working hypothesis is intentionally demanding. A strategy should only be treated as evidence of skill if it satisfies several conditions simultaneously. It should outperform a defensible randomness benchmark, achieve low one-sided p-values, rank high within the null distribution, maintain positive standardized effect size, remain stable across repeated runs, and survive broader out-of-sample testing. If those conditions fail, then realized profitability should be interpreted at most as asset-specific historical success rather than as evidence of generalizable predictive edge.

This distinction matters because the project now operates at two different inferential levels. The first is ticker-specific evidence: a strategy may look unusually strong on a particular instrument under the current null. The second is generalizable evidence: that same strategy should continue to look strong after cross-ticker multiple-testing control and across rolling out-of-sample panels. The present paper keeps those levels separate and treats them differently.

## Background and Related Literature

Empirical finance provides reasons both to search for trading structure and to distrust easy inferences from historical success. On one side, research on return continuation shows that momentum-type behavior can exist under specific horizons and asset classes. Moskowitz, Ooi, and Pedersen (2012) document time-series momentum across futures markets and show that past directional movement can retain predictive value. On the other side, reversal research indicates that short-run mean reversion can also occur. Poterba and Summers (1988) and Lehmann (1990) both show that short-horizon reversals can emerge within broader market dynamics.

That literature motivates testing multiple strategy families, but it does not justify naive backtest interpretation. Lo (2002) demonstrates that Sharpe ratios are estimates whose meaning depends on dependence structure, sampling variation, and time scaling. White (2000) and Bailey, Borwein, Lopez de Prado, and Zhu (2017) show how repeated testing and model search can produce impressive in-sample results that do not generalize. Those critiques are especially relevant to retail and student research environments, where one or two summary metrics often substitute for actual inference.

Simulation and resampling methods offer a stronger framework. Efron and Tibshirani (1993) explain why Monte Carlo and bootstrap-style inference are useful when analytic reference distributions are unavailable or when dependence structures make closed-form testing awkward. In trading research, the crucial question is not whether to simulate, but what the simulation means. A weak null can make the actual strategy look ordinary by construction. A stronger null should preserve the structural features that matter while still representing luck in a way that is financially interpretable.

Regime-sensitive finance research adds another important dimension. Ang and Bekaert (2002) show that market behavior varies across states, which implies that a single unconditional summary can conceal important structure. However, regime splitting introduces another problem: once results are sliced into calm, neutral, and stressed environments, some cells become too sparse for reliable interpretation. This project addresses that issue by suppressing low-count regime cells rather than forcing every cell into the same visual or inferential status.

The contribution of the present study is therefore practical. It combines realistic execution, a matched random-timing null, repeated-run robustness, false-discovery-rate control, regime-conditioned summaries, explicit no-trade handling, and fold-local walk-forward testing into one synchronized research pipeline. The objective is not to prove that simple strategies never work. The objective is to measure how often apparently strong outcomes continue to look strong once the main sources of false confidence have been controlled.

## Data and Experimental Design

### Asset Universe

The current single-ticker evaluation layer covers 17 instruments:

- `AAPL`
- `BTC-USD`
- `CL=F`
- `ES=F`
- `EURUSD=X`
- `FXI`
- `GC=F`
- `MRNA`
- `NQ=F`
- `NVAX`
- `NVDA`
- `QQQ`
- `SPY`
- `TQQQ`
- `TSM`
- `VOO`
- `XLU`

The out-of-sample walk-forward layer uses a stricter 11-ticker core panel with sufficient rolling-fold depth and synchronized data support:

- `SPY`
- `QQQ`
- `AAPL`
- `VOO`
- `NVDA`
- `TSM`
- `MRNA`
- `NVAX`
- `BTC-USD`
- `NQ=F`
- `ES=F`

All data are downloaded through `yfinance` with adjusted pricing enabled. Prices are sorted chronologically, features are built from past information only, and the live pipeline now treats sparse-activity or zero-trade cases as legitimate methodological states rather than as failures.

### Strategy Set

The active experiment evaluates five long-only strategies under one common framework:

1. **Trend:** enter when price crosses above the 50-day moving average and exit when price crosses below it.
2. **Mean reversion:** enter when the 20-day z-score is -2 or lower and exit when price crosses back above the 20-day moving average.
3. **Random:** enter and exit probabilistically with fixed daily probabilities, subject to the same execution rules as the rule-based strategies.
4. **Momentum:** enter when trailing 120-day return becomes positive and exit when it turns non-positive.
5. **Breakout:** enter when price closes above the prior 20-day breakout high and exit when price closes below the prior 20-day breakout low.

The system also includes a **buy-and-hold benchmark**. This benchmark is kept in the comparison tables and charts for context, but it is not included in the skill-versus-luck null because it is not an active timing rule.

### Feature Engineering

The project now uses a synchronized feature layer that includes:

- daily simple return
- 20-day moving average
- 50-day moving average
- 20-day rolling price standard deviation
- 20-day z-score
- 20-day trailing average volume
- 120-day trailing return for momentum
- prior 20-day breakout high
- prior 20-day breakout low

This layer reflects several fixes made during the project. In particular, the mean-reversion rule now uses a price-based z-score rather than mixing price levels with return volatility, and the momentum and breakout rules are now evaluated under the same execution and inference framework as the original trend, mean-reversion, and random strategies.

### Realistic Execution Layer

All active strategies share one execution model. The model imposes:

- next-bar execution at the following open
- \$100,000 starting capital
- integer share sizing
- 100% maximum capital deployment
- 5% maximum participation rate of trailing 20-day average volume
- 0.5 basis point half-spread cost
- randomized slippage between 0.5 and 3.0 basis points
- commission of \$0.005 per share with a \$1.00 minimum per order

This design is still a daily-bar approximation rather than a full order-book simulator, but it is far more realistic than a frictionless close-to-close backtest. It also standardizes treatment across strategy families so that the newer momentum and breakout strategies receive no special handling.

## Statistical Framework

### Core Performance Metrics

The live pipeline distinguishes clearly between daily-curve metrics and trade-level metrics.

From daily equity curves, it computes:

- cumulative return
- annualized return
- annualized Sharpe ratio
- maximum drawdown

From completed trades, it computes:

- mean trade return
- median trade return
- standard deviation of trade returns
- win rate
- expected value
- trade-level return ratio

One of the major fixes made since the earlier paper version is that the old generic label `Sharpe ratio` is no longer misused for a trade-level mean-to-standard-deviation quantity. The project now reports a true time-scaled annualized Sharpe from the daily equity curve and labels the trade-level quantity explicitly as `trade_level_return_ratio`.

### Monte Carlo Null Model

The inferential core of the project is a matched random-timing null. For each ticker-strategy pair, the null model preserves:

1. realized trade count
2. realized holding durations
3. realized position-value fractions
4. transaction-cost assumptions
5. non-overlap between simulated trades

It then randomizes trade placement on the observed market path. The principal inferential measures are:

\[
RCSI = CR_{actual} - \operatorname{Median}(CR^{sim})
\]

\[
RCSI_z = \frac{CR_{actual} - \operatorname{Mean}(CR^{sim})}{SD(CR^{sim})}
\]

\[
p = P(CR^{sim} \ge CR_{actual})
\]

The pipeline also reports percentile placement inside the null distribution and p-value prominence:

\[
\text{p-value prominence} = -\log_{10}(p)
\]

Raw `RCSI` remains useful within one ticker, but cross-asset interpretation now emphasizes `RCSI_z`, percentile, and p-value prominence because these are more scale-free.

### Repeated-Run Robustness

Each ticker-strategy pair is reevaluated across 30 outer runs with 1,000 simulations per outer run. The robustness layer measures:

- mean p-value
- mean percentile
- mean `RCSI_z`
- mean p-value prominence
- proportion of runs with \( p \le 0.05 \)
- proportion of runs above the null median
- stability classification

This repeated-run layer is important because a single Monte Carlo draw can be misleading, especially for borderline cases. The current evidence labels and verdicts are therefore based on repeated-run summaries rather than on single-run null draws alone.

### False-Discovery-Rate Control

The expanded ticker universe produces 85 active ticker-strategy tests:

\[
17 \text{ tickers} \times 5 \text{ active strategies} = 85 \text{ tests}
\]

The project now applies Benjamini-Hochberg false-discovery-rate control across all 85 p-values. This is one of the most important upgrades relative to earlier project stages because isolated nominal significance is expected when many related hypotheses are tested at once.

### Regime Analysis

Regimes are defined using expanding historical quantiles of 20-day volatility, shifted by one day to avoid forward-looking leakage. Each observation is labeled:

- calm
- neutral
- stressed

Regime results are descriptive rather than inferential. Any strategy-regime cell with fewer than 20 completed trades is suppressed in the heatmaps and regime bar charts. This rule was added to prevent low-count cells from appearing visually authoritative.

### No-Trade Handling

The live pipeline explicitly handles zero-trade outcomes. If a strategy produces no completed trades on a ticker:

- the active equity curve remains flat
- cumulative return, annualized return, annualized Sharpe, and drawdown are mechanically zero for the active strategy path
- inferential fields such as p-value, percentile, and `RCSI_z` are suppressed
- the verdict engine labels the case `No Trades`
- the chart system emits placeholder pages rather than misleading inferential graphics

This feature was added after the pipeline failed on `EURUSD=X` because downstream scripts treated empty trade files as invalid input.

### Walk-Forward Generalization Test

The strongest out-of-sample test in the project is the fold-local walk-forward engine. It uses:

- 504-bar test windows
- 252-bar step size
- 30 outer runs
- 1,000 simulations per outer run

Most importantly, each fold is rerun from the fold start with fresh capital. The current engine no longer filters trades out of a full-history simulation. That correction materially improves validity.

## Results

### Conceptual Result: Apparent Virtu Often Dissolves Into Fortuna

The empirical findings fit the project’s motivating idea. Many strategies produce realized outcomes that initially look like *virtu*: large raw returns, attractive charts, or strong single-ticker summaries. However, once those outcomes are judged relative to a structured null, repeated-run robustness, false-discovery correction, and walk-forward persistence, much of that apparent skill is reclassified as *fortuna*. This is not a rhetorical flourish. It is the operational meaning of the project’s statistics.

### Ticker-Specific Positive Cases

The refreshed single-ticker layer still finds several positive local cases. The strongest examples in the current results are:

- `SPY` mean reversion
- `VOO` mean reversion
- `QQQ` momentum
- `BTC-USD` trend
- `BTC-USD` momentum
- `BTC-USD` breakout

Representative values are shown below.

| Ticker | Strategy | Evidence Label | Ann. Sharpe | RCSI_z | p-value | Percentile | FDR q-value |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| SPY | Mean reversion | Strong Skill | 0.4148 | 3.3067 | 0.0060 | 99.40 | 0.1500 |
| VOO | Mean reversion | Strong Skill | 0.5429 | 2.9983 | 0.0052 | 99.48 | 0.1500 |
| QQQ | Momentum | Strong Skill | 0.7337 | 1.9335 | 0.0474 | 95.26 | 0.5925 |
| BTC-USD | Trend | Strong Skill | 1.2362 | 7.4641 | 0.0024 | 99.76 | 0.1500 |
| BTC-USD | Momentum | Strong Skill | 1.1452 | 4.1541 | 0.0092 | 99.08 | 0.1725 |
| BTC-USD | Breakout | Strong Skill | 1.0858 | 2.6004 | 0.0220 | 97.80 | 0.3300 |

These results show that the framework is capable of identifying strong ticker-conditional outcomes. They should not, however, be treated as universal evidence. After Benjamini-Hochberg adjustment across all 85 active tests, none of these cases remains significant at q <= 0.10. That is one of the central conclusions of the study.

### Passive Benchmark Context

The buy-and-hold benchmark clarifies an additional distinction. Null-relative evidence and passive benchmark dominance are not the same thing.

Examples from the current results include:

- `SPY` buy-and-hold cumulative return: `24.5085`, versus `3.1259` for mean reversion
- `QQQ` buy-and-hold cumulative return: `10.1971`, versus `12.0321` for momentum
- `VOO` buy-and-hold cumulative return: `5.3227`, versus `1.2369` for mean reversion
- `BTC-USD` buy-and-hold cumulative return: `317.5346`, versus `603.6129` for trend

These contrasts show why the benchmark is contextual rather than inferential in the present framework. A strategy can beat the null without beating passive exposure, and it can exceed passive exposure without surviving broader statistical controls.

### Regime-Conditioned Descriptive Patterns

The regime layer reveals several plausible patterns, while also enforcing sample discipline through suppression.

- `SPY` mean reversion remains positive in all three valid regimes, with trade-level return ratios of approximately `1.1733` in calm markets, `0.2633` in neutral markets, and `0.2589` in stressed markets.
- `QQQ` momentum remains positive across all three valid regimes, with trade-level return ratios of approximately `0.3767`, `0.2109`, and `0.1874`.
- `BTC-USD` trend is positive across all three valid regimes, with trade-level return ratios of approximately `0.3194`, `0.0915`, and `0.2370`.

The suppression rule matters. `SPY` momentum in calm markets has only 18 trades, and `SPY` breakout in calm markets has only 17 trades, so those cells are not plotted as if they were inferentially comparable with higher-count cells. Likewise, several BTC-USD mean-reversion and breakout regime cells are suppressed because they do not meet the 20-trade threshold.

### No-Trade Validation Case

`EURUSD=X` is the clearest demonstration of the project’s improved handling of sparse-activity assets. Under the current long-only daily strategy set, all five active strategies generated zero completed trades on this ticker. The present pipeline now handles this outcome explicitly:

- all five active strategies are labeled `No Trades`
- inferential fields are suppressed
- active path metrics are mechanically zero
- the buy-and-hold benchmark remains available for context
- placeholder charts are generated instead of broken inferential pages

This case does not supply positive or negative evidence about strategy skill. It demonstrates that the project can distinguish between meaningful inference and non-engagement.

### Generalizable Evidence: Walk-Forward Panel

The strongest empirical test in the project is the fold-local 11-ticker walk-forward panel. The current synchronized summary contains 1,120 ticker-fold-strategy evaluations and is substantially more conservative than the single-ticker layer.

| Strategy | Panels | Mean Ann. Sharpe | Mean p-value | Mean Percentile | Mean RCSI_z | Share of Panels with Mean p <= 0.05 | Final Classification |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Trend | 249 | 0.2306 | 0.7040 | 29.60 | -0.7941 | 0.40% | Likely Random |
| Mean reversion | 242 | 0.3405 | 0.4319 | 56.81 | 0.3033 | 12.81% | Null-Like / Inconclusive |
| Random | 249 | 0.3967 | 0.5621 | 43.79 | -0.2268 | 3.21% | Null-Like / Inconclusive |
| Momentum | 187 | -0.0573 | 0.7251 | 27.49 | -0.8929 | 0.53% | Likely Random |
| Breakout | 193 | 0.2238 | 0.6970 | 30.30 | -0.7815 | 0.00% | Likely Random |

This is the most important quantitative result in the paper. None of the five active strategies demonstrates generalizable robust skill across the current out-of-sample panel. Mean reversion is the strongest of the five under this harsher test, but it remains `Null-Like / Inconclusive` rather than `Skill`. Trend, momentum, and breakout are more consistent with randomness than with persistent edge. The random control also remains non-skill-like, which supports the internal coherence of the inferential framework.

## Discussion

The results support a much stricter interpretation than a naive reading of the equity curves would suggest. At the local level, several strategies look compelling. SPY and VOO mean reversion produce low p-values, high percentiles, and large positive `RCSI_z`. QQQ momentum appears strong in a technology-heavy index. BTC-USD generates especially large trend, momentum, and breakout outcomes. None of those results is inherently implausible. Each can be connected to recognizable market behavior.

SPY and VOO mean reversion make sense because broad equity indexes can display short-horizon rebound behavior within a long-run upward drift. QQQ momentum is plausible because that market experienced persistent growth phases. BTC-USD trend, momentum, and breakout are also plausible because the sample contains extreme directional episodes and clustered volatility. These are reasonable asset-specific stories.

The stronger contribution of the project, however, is that it does not stop at local plausibility. Once cross-ticker false-discovery control is imposed, those apparently strong local cases no longer justify a broad claim of skill. Once the analysis shifts to the 11-ticker fold-local walk-forward layer, even the best-performing strategy families fail to maintain a universal edge. The meaning of the results therefore changes when the standard of evidence becomes more demanding. The paper becomes less a catalog of profitable rules and more a quantitative demonstration of how often apparent *virtu* is inseparable from *fortuna*.

The behavior of the individual strategy families also makes sense under this interpretation. Trend can underperform on broad equity indexes because a slow crossover rule repeatedly exits and re-enters around noise while random timing may remain exposed during persistent drift. Mean reversion can look especially strong on index ETFs because the rule buys temporary dislocations in markets with long-run upward bias. Momentum can look strongest on QQQ because the underlying asset went through persistent directional phases. BTC-USD can favor trend, momentum, and breakout because its sample contains unusually strong serial movement and large trend episodes.

The random strategy remains a useful diagnostic. It does not need to be maximally bad in every case. A realized random path can land near the middle of its own null distribution, which correctly produces an inconclusive label rather than an artificially extreme negative label. What matters is that the random control does not emerge as stable, reproducible skill under the full framework.

## Limitations

Several limitations remain and should be acknowledged directly.

1. The project is still a daily-bar system rather than an intraday microstructure simulator.
2. The null model randomizes trade timing on the observed market path rather than generating entirely new market paths.
3. The walk-forward panel, although much stronger than earlier versions, is still finite and does not span the full space of global assets.
4. Some instruments, especially outside equity-like structures, can produce sparse or zero-trade outcomes under the present long-only daily design.
5. Buy-and-hold is contextual rather than inferential in this framework and should not be described as having undergone the same timing null test.

These limitations restrict the scope of the conclusions, but they do not negate the value of the study. They simply define what can and cannot be claimed.

## Conclusion

The final answer to the research question is cautious and precise. The refreshed experiment does identify asset-conditional historical success. SPY and VOO mean reversion, QQQ momentum, and several BTC-USD strategies are the clearest examples at the single-ticker level. However, none of those results survives false-discovery-rate control at q <= 0.10 across the expanded 17-ticker universe. When the analysis is moved to the fold-local 11-ticker walk-forward panel, no strategy demonstrates a universal robust edge.

The most defensible conclusion is therefore this: the project finds ticker-specific historical alignment, not generalizable proof of persistent trading skill. In Machiavelli's terms, many outcomes that first appear to reflect *virtu* are better understood as mixtures of *virtu* and *fortuna*, and often as *fortuna* alone once the evidence standard is raised. That is not a failure of the project. It is the central contribution of the project. The system does not accept profitability as proof. It forces the question that the original inspiration demanded: when something works, how do we know it was not just luck?

## References

Ang, A., & Bekaert, G. (2002). International asset allocation with regime shifts. *The Review of Financial Studies, 15*(4), 1137-1187. https://doi.org/10.1093/rfs/15.4.1137

Bailey, D. H., Borwein, J. M., Lopez de Prado, M., & Zhu, Q. J. (2017). The probability of backtest overfitting. *Journal of Computational Finance, 20*(4), 39-69. https://doi.org/10.21314/JCF.2016.322

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B, 57*(1), 289-300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x

Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall/CRC.

Lehmann, B. N. (1990). Fads, martingales, and market efficiency. *The Quarterly Journal of Economics, 105*(1), 1-28. https://doi.org/10.2307/2937816

Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal, 58*(4), 36-52. https://doi.org/10.2469/faj.v58.n4.2453

Machiavelli, N. (1532/2008). *The prince* (W. K. Marriott, Trans.). Oxford University Press.

Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics, 104*(2), 228-250. https://doi.org/10.1016/j.jfineco.2011.11.003

Poterba, J. M., & Summers, L. H. (1988). Mean reversion in stock prices: Evidence and implications. *Journal of Financial Economics, 22*(1), 27-59. https://doi.org/10.1016/0304-405X(88)90021-9

White, H. (2000). A reality check for data snooping. *Econometrica, 68*(5), 1097-1126. https://doi.org/10.1111/1468-0262.00152

## Appendix A: Principal Result Files

The main live result files supporting this paper are:

1. `Data_Clean/*_full_comparison.csv`
2. `Data_Clean/*_monte_carlo_summary.csv`
3. `Data_Clean/*_monte_carlo_robustness_summary.csv`
4. `Data_Clean/*_rcsi.csv`
5. `Data_Clean/*_regime_analysis.csv`
6. `Data_Clean/cross_ticker_fdr_adjusted.csv`
7. `Data_Clean/multi_asset_walk_forward_panel_summary.csv`
8. `Data_Clean/multi_asset_walk_forward_agent_summary.csv`

## Appendix B: Current Pipeline Scope

The active single-ticker entrypoint is `Code/AAAmain.py`. In its current synchronized form, the single-ticker pipeline performs:

1. feature generation
2. forward-only regime labeling
3. five active strategy runs
4. buy-and-hold benchmark generation
5. metric generation
6. matched random-timing Monte Carlo analysis
7. repeated-run robustness analysis
8. RCSI and standardized-RCSI reporting
9. FDR-adjusted comparison-table generation
10. chart generation, including placeholder handling for `No Trades`

The out-of-sample generalization entrypoint is `Code/multi_asset_walk_forward.py`, which reruns each fold from a fresh capital state and produces the walk-forward results summarized in this paper.
