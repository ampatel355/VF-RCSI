# Quantitative Research Audit Report

> Archival note: this report documents the original audit findings that triggered the current refactor. Some issues described here, including the old null design and earlier regime implementation, have since been fixed in the live code. A later same-day update also added explicit no-trade handling so sparse-activity tickers no longer crash the pipeline or receive misleading skill/luck labels. For the synchronized post-fix interpretation, use [README.md](/Users/aryanpatel/Downloads/Virtu_Fortuna_Project/README.md) and [paper.md](/Users/aryanpatel/Downloads/Virtu_Fortuna_Project/Notes/paper.md).

## March 30, 2026 Addendum

One further reliability fix was added after the main audit documented here. The live system now supports ticker-strategy pairs with zero completed trades.

That update changed the project in four important ways:

- header-only trade files are treated as valid zero-trade outputs rather than as pipeline errors
- the verdict engine assigns `No Trades` instead of forcing a skill/luck classification
- the active comparison layer suppresses inferential statistics such as p-value, percentile, and `RCSI_z` for no-trade strategies
- the chart layer produces placeholder figures rather than crashing when no inferential data exist

This matters for sparse-activity instruments and validation runs outside the main universe, including the explicit `EURUSD=X` test performed on March 30, 2026.

## Scope

This audit evaluates the current trading strategy research pipeline with respect to conceptual validity, implementation correctness, statistical inference, and robustness. It also documents the code-level corrections applied during the audit.

## A. Critical Issues

### 1. The original Monte Carlo null model was conceptually invalid

The prior active Monte Carlo design resampled each strategy from its own realized trade-return distribution while preserving the same number of trades. That null does not test skill. It tests whether the realized sample differs from bootstrap draws taken from the empirical distribution defined by the realized sample itself. Because the expected sum of bootstrapped log returns equals the realized sum of empirical log returns, the realized cumulative return is mechanically pulled toward the center of the simulated distribution. This explains why the original study produced p-values near 0.50 and RCSI values near zero for nearly every strategy.

This issue materially invalidated the original inferential conclusion. The old Monte Carlo result did not show that the strategies lacked skill. It showed that the null model was anchored to the observed returns by construction.

### 2. The original main conclusion exceeded the evidence

The prior study concluded that none of the strategies demonstrated meaningful skill. That conclusion depended on the invalid self-bootstrap null described above. Once the null model was corrected, the result changed materially: the mean-reversion strategy showed strong outperformance relative to a matched random-timing null on the synchronized SPY sample, while the trend strategy underperformed that null. The original conclusion was therefore not defensible.

### 3. The mean-reversion signal contains a dimensional inconsistency

The implemented buy rule compares price to:

`ma_20 - 2 * std_20`

where `ma_20` is in price units and `std_20` is the rolling standard deviation of percentage returns. Those units are not commensurate. The rule is therefore heuristic rather than dimensionally correct. This does not make the code unusable, but it weakens the economic interpretation of the signal. The study should not describe this rule as a strict statistical band or a proper z-score trigger.

### 4. The regime labels use full-sample information

The regime script classifies volatility states with `qcut` over the full sample. That means regime thresholds are estimated using future observations relative to earlier dates. The strategies themselves do not trade on regime labels, so this is not a direct trade-execution leak. However, it does create look-ahead contamination in the regime-analysis layer. Any claim that a strategy performs well in a given regime must therefore be treated as ex post descriptive analysis rather than a forward-usable classification.

### 5. The study remains entirely in-sample

The pipeline uses the full historical sample to generate features, construct strategies, evaluate performance, and conduct inference. This makes the project a strong exploratory framework, but not a complete validation framework. Without walk-forward testing, rolling re-estimation, or out-of-sample evaluation, the study cannot support broad claims about deployable edge.

## B. Minor Issues

### 1. Reproducibility was previously too weak for formal research use

The random agent originally defaulted to non-reproducible stochastic behavior, and the single-run Monte Carlo engine originally used one evolving random-number generator across all agents. Both design choices made the pipeline less suitable for formal research documentation.

### 2. The random strategy is a comparator, not a sufficient null by itself

The random agent remains useful as a control strategy, but it is not by itself a valid test of timing skill. It has its own holding-period distribution, exposure path, and stochastic trajectory. A proper null must hold those structural properties approximately constant while randomizing the timing mechanism.

### 3. The p-values are conditional on the chosen null, not universal evidence

The corrected p-values are meaningful only relative to the matched random-timing null. They do not prove structural alpha in a stronger sense, and they do not substitute for out-of-sample validation, White-style data-snooping correction, or cross-asset replication.

### 4. Slippage and market frictions remain simplified

The project includes a fixed 0.1% transaction cost per trade but does not model variable slippage, spread widening, liquidity constraints, or market impact. This is acceptable for a research prototype but should be stated explicitly whenever results are presented.

## C. Corrected Method / Logic

### Corrections applied in code

The following corrections were implemented during the audit:

1. `Code/monte_carlo.py`
   - Replaced the self-bootstrap null with a matched random-timing null.
   - The new simulation preserves trade count and holding-period durations but randomizes where trades occur in the market history.
   - Added explicit `null_model = random_timing_matched_duration` to the output summary.
   - Made single-run Monte Carlo reproducible by default.
   - Standardized agent-level random seeding.

2. `Code/monte_carlo_robustness.py`
   - Updated the robustness engine to use the same matched random-timing null.
   - Preserved the existing summary schema while aligning robustness inference with the corrected null model.

3. `Code/random_agent.py`
   - Changed the default to reproducible stochastic behavior for formal research consistency.

### Correct null-model logic

The corrected simulation proceeds as follows:

1. Load the realized trades for a strategy.
2. Map each realized trade onto the market open-price history.
3. Convert each trade into a holding period measured in open-to-open bars.
4. For each simulation:
   - randomly permute the observed holding periods,
   - distribute the unused bars across pre-trade, between-trade, and post-trade gaps,
   - enforce non-overlapping trade intervals,
   - compute simulated open-to-open trade returns from the actual market path,
   - subtract transaction cost,
   - compound the simulated trade sequence in log-return space.
5. Compare the actual cumulative return to the resulting distribution.

### Corrected pseudocode

```text
for each strategy:
    load realized trades
    compute actual cumulative return from realized trade returns
    map realized trades to holding durations in market bars

    for sim in 1..N:
        shuffle durations
        randomly allocate slack bars across trade gaps
        construct non-overlapping entry/exit schedule
        compute trade returns from market open prices
        subtract transaction cost
        compound simulated returns

    p_value = proportion(simulated >= actual)
    percentile = proportion(simulated <= actual)
    RCSI = actual - median(simulated)
```

### Stronger next upgrades

The following improvements are still recommended but were not fully implemented in this audit pass:

1. Replace full-sample regime thresholds with expanding or rolling thresholds.
2. Replace the current mean-reversion trigger with a dimensionally correct band such as:
   - `Close < MA20 * (1 - k * sigma_return)`, or
   - `z = (Close - MA20) / rolling_price_std`.
3. Add out-of-sample or walk-forward validation.
4. Add parameter sensitivity analysis over moving-average windows and threshold sizes.
5. Add block bootstrap or regime-preserving resampling if the null is extended further.

## D. Revised Conclusion

The original conclusion should be replaced with the following:

> Under the corrected matched-duration random-timing null, the synchronized SPY evidence does not support the prior claim that all tested strategies behave like randomness. Instead, the results are differentiated. The trend strategy underperforms the random-timing baseline materially, with an actual percentile of 5.04, a one-sided p-value of 0.9496, and RCSI of -3.7866. The mean-reversion strategy outperforms the corrected null materially, with an actual percentile of 96.34, a one-sided p-value of 0.0366, and RCSI of 2.5719. This result remains stable across 30 seed-based robustness runs, where the mean p-value is 0.0356 and mean RCSI is 2.5595. The random strategy shows positive relative performance but does not cross conventional significance thresholds, with a p-value of 0.1266. Accordingly, the evidence supports a qualified finding of timing skill for the implemented mean-reversion rule on the current SPY sample, rejects skill for the trend rule on that same sample, and remains inconclusive for the random control. Because the study is still in-sample and the mean-reversion trigger is heuristic, this should be treated as suggestive evidence rather than definitive proof of deployable alpha.

## E. Final Verdict

**Partially valid**

The project is structurally strong as a research pipeline and avoids some common implementation errors, including same-bar look-ahead in trade execution. However, its original inferential layer was invalid because the self-bootstrap Monte Carlo null was anchored to the observed returns. After replacing that null with a matched random-timing benchmark, the study becomes materially more defensible, but unresolved issues, especially full-sample evaluation, ex post regime thresholds, and the mean-reversion specification, prevent a full `Valid` rating.
