QC report — field guide for the operator

The function `report(df, target)` produces a single per-column diagnostic table. Columns are the features, with the target appended when supplied. Rows are metrics in a deliberate order: distributional health, stationarity, persistence, decorrelation length, pairwise redundancy, target edge, and regime stability. Each cell answers one question about one column. The panel is read column-wise to decide what becomes of the feature — kept as it stands, transformed, clustered with neighbours, gated by regime, or dropped.

The thresholds in the example output's `ref` column are operator guidance, not constants in the source. They mark the values at which a column changes status — usable, borderline, suspect — and reflect conventions appropriate to hourly bar data with a return-style target. The same panel against a different target or sampling cadence requires different thresholds; the metric structure is invariant, the interpretive scale changes with the target and sampling frequency.

## Read the panel as a funnel

The six layers of the report form a sequence, not a checklist. Reject unusable data first. Stabilise drifting series next. Account for persistence. Collapse redundant features. Test the target edge. Check regime stability. A feature is not attractive because it has one good row; it is attractive because it survives the sequence. The order matters because each layer's metrics are only meaningful once the prior layers have passed — a strong Spearman on a non-stationary feature is co-trending until proven otherwise, and a confident bootstrap confidence interval (CI) on a feature riddled with NaNs measures a subsample whose representativeness has not been established.

## Operating assumptions

The defaults are calibrated to hourly bar data, a return-style target, uniform sampling across the index, and a minimum of 100 paired non-NaN rows per feature-target pair for the edge metrics. Gaps and irregular spacing degrade every lag-based metric. Targets that are not returns — classification labels, fill ratios, hold times — preserve the metric structure but invalidate the example thresholds. The persistence lags (1, 5, 24, 168) assume hours; on a 5-minute cadence those lags name different horizons and the daily and weekly thresholds shift accordingly.

## Decision table

| Layer | Red flag | Action |
|---|---|---|
| Missingness | `missing_rate` above tolerance | repair or drop |
| Cardinality | very low `unique_fraction` | treat as discrete; ignore decile blanks |
| Stationarity | Augmented Dickey-Fuller (ADF) does not reject and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) rejects | difference or detrend before edge metrics |
| Persistence | high autocorrelation function (ACF), long `block_size` | treat downstream cross-validation (CV) and bootstrap as dependent |
| Redundancy | `max_pairwise_spearman` ≥ 0.7 | cluster; keep one representative |
| Edge | Spearman CI crosses zero | do not treat the sign as stable |
| Deciles | low `monotonic_fraction` | inspect for non-linearity or regime effects |
| Regime | sign agreement below 4/4 | gate the feature or use it conditionally |

The table is the operator's first read. The sections below are the prose behind each row.

## How the panel marks the absent

A cell holds `None` when the metric is undefined for that column, not zero. The conditions are specific: an exception caught inside the safe-evaluation wrapper, a NaN or Inf result, a sample below the per-metric minimum, zero variance in the input, or, in the case of decile spread, an empty quantile bin. A numeric zero is a real measurement; **`None` is not**. A column showing `None` for `decile_spread` and `monotonic_fraction` while showing a finite Spearman has not failed; its rank correlation succeeded and its decile decomposition failed because the column is discrete and several of its deciles fall on the same value. The example's `is_weekend` row demonstrates this exactly.

The target's own column under the pairwise and edge rows is blank by construction. The target does not compete with itself in either bucket.

## Distributional health

The first rows establish whether the column is usable data. `missing_rate` is the null fraction over the row count. Below one percent is operationally invisible. Above a few percent the column should be repaired or excluded before any downstream metric is trusted, because every subsequent row drops NaNs before computing — a 30 percent missing rate means stationarity, correlation, and edge all rest on a 70 percent subsample whose representativeness is not established. The pairwise metrics are **more vulnerable still**: they require rows where every feature is simultaneously non-NaN, so a single column with heavy missingness reduces the effective sample for all of them.

`unique_fraction` counts distinct values over rows. Continuous features sit near one. Discrete or low-cardinality features sit near zero. The reading interacts with later metrics: a feature with `unique_fraction` near zero may return `None` for decile spread because the deciles collapse onto each other. Read the unique fraction first, and the blanks below stop being surprising.

The five distributional markers — `min`, `max`, `median`, `p01`, `p99` — describe the spread on non-NaN values. The percentile pair is the working description of the tails. Compare each against the column's expected physical range. A z-score that drifts to ±8 has either tail mass that ordinary scaling cannot tame or a clipping bound that has come loose. The example shows `parkinson_volatility_1` peaking at 726 bps per bar against a p99 of 165 bps: a single extreme bar that the p99 protects most metrics from but that the raw maximum reveals.

`skew` is the bias-corrected sample skewness. Symmetric distributions sit near zero. Above roughly two, the column has a tail that will dominate ordinary regression and bias any metric that relies on means. A log or signed-log transform is the standard remedy before downstream use. In the example, the Parkinson volatilities, the absolute z-scores, and the absolute normalised return all show skew above 2.5, which is the expected signature of those constructions.

## Stationarity, read jointly

`adf_p` is the Augmented Dickey-Fuller p-value. The null hypothesis is that the series has a unit root — that its level wanders without a stable mean to return to. A p-value below 0.05 rejects the wandering and concludes the series is stationary. The implementation fixes `maxlag = 10` and disables automatic lag selection, a deliberate trade of statistical power for runtime determinism. Series longer than 5,000 observations are stride-subsampled to that length, so the test for long series is indicative rather than authoritative.

`kpss_p` is the KPSS p-value with the inverse null: stationarity around a constant. A p-value above 0.05 fails to reject stationarity. The same subsampling and minimum-sample rule applies. The KPSS implementation reports a floor at 0.01 and a ceiling at 0.10; the values `0.01` and `0.10` are the test's expressive limits, not precise probabilities.

The two tests carry their weight together, not separately. ADF rejecting and KPSS not rejecting is the clean stationary case under these test settings. Both rejecting indicates trend-stationarity or a borderline case in which level moves but the moves are bounded. ADF not rejecting and KPSS rejecting indicates a genuinely drifting series, and every metric below — Spearman, mutual information, decile spread — should be treated as describing a wandering relationship rather than a stable one. A strong correlation between two non-stationary series is, by default, **a co-trending artifact** until proven otherwise.

## Persistence and decorrelation

The autocorrelation rows — `acf_lag_1`, `acf_lag_5`, `acf_lag_24`, `acf_lag_168` — describe how the column's values persist across hourly bars. The lags span the operational horizons of hourly market data: the next bar, the next five hours, the next day, the next week. Above 0.5 at lag 1 is a persistent state variable. Above 0.3 at lag 5 means the persistence still has structure at the medium-term horizon. Above 0.2 at lag 24 means the column carries a daily cycle. Above 0.2 at lag 168 means it carries a weekly cycle.

The `is_weekend` column in the example shows the canonical signature of a weekly-cycle dummy: high persistence at every lag, with 0.99 at lag 168. The `nth_hour` column shows the canonical signature of a daily cycle: near-zero at lag 5, 0.998 at lag 24, 0.99 at lag 168. A column whose ACF sits below 0.2 at every lag tested is persistence-light at the tested horizons; that does not preclude contemporaneous target edge, only persistence-based modelling.

`block_size` is the decorrelation length, computed by walking the absolute autocorrelation from lag 1 to lag 200 and returning the first lag where it falls below 0.05. A returned value of 24 may be **a fallback, not an estimate**: when the ACF does not clear the 0.05 threshold within 200 lags, or when the ACF computation itself fails, the function substitutes 24 as an operational default. The current panel does not distinguish these cases on the face of the table — a `block_size` of 24 should be read as "decorrelation length unavailable, using operational default" rather than "feature decorrelates after one day" unless the ACF rows above make the latter plausible.

This value is the bridge between the column's persistence and its bootstrap confidence interval. The bootstrap resamples blocks of this length to preserve local dependence; setting the block too short returns intervals that resemble independent and identically distributed (iid) intervals and understate uncertainty, setting it too long wastes effective sample size. The example values range from 5 bars for the hour-of-day index, which decorrelates by construction every five hours, to 200 bars for `parkinson_volatility_4`, which carries strong persistence and demands long blocks to capture it.

## Pairwise redundancy

Four rows describe how the column relates to the other features in the set. They are computed on rows where every feature is simultaneously non-NaN, which is the strictest possible filter — a single column with heavy missingness reduces the effective sample for all pairwise metrics. Inputs are stride-subsampled to 20,000 rows for Spearman and 5,000 for mutual information. When fewer than 30 fully clean rows survive the joint filter, all four pairwise values for every column return `None`.

`max_pairwise_spearman` is the largest absolute rank correlation against any other feature. Below roughly 0.3, the column is unlikely to have a close rank-monotonic twin in this feature set. At or above 0.7, the column has a near-twin and the pair should be collapsed by selecting one representative. `mean_pairwise_spearman` is the same quantity averaged over all other features. It distinguishes a column with one near-twin (high max, low mean) from one broadly correlated with the whole set (high mean).

`redundancy_penalty` is one minus `max_pairwise_spearman`. It is the same information as the max, inverted for ranking purposes: higher values mean more unique. Above 0.7 indicates a column that no other column captures.

`max_pairwise_mi` is the maximum mutual information against any other feature, via `mutual_info_regression`. It catches the non-linear redundancy that Spearman misses: two features may be rank-uncorrelated yet jointly determined by a common non-monotonic driver. Below 0.1 indicates near-independence in the information-theoretic sense.

In the example, the two Parkinson volatilities at lags 1 and 4 share a `max_pairwise_spearman` of 0.82 and a `max_pairwise_mi` of 0.63, the expected signature of a feature pair representing the same quantity at different smoothings. Usually keep one unless different horizons are intentionally modelled. The `nth_hour` column at 0.20 and 0.07 is genuinely orthogonal to everything else in the set.

## Edge against the target

The next six rows are populated only when a target is supplied and present in the DataFrame. Each is computed on rows where the feature and the target are simultaneously non-NaN. The per-column minimum is 100 paired rows; below that threshold all six edge values for the column return `None`. The target's own column is blank by construction.

`spearman` is the rank correlation of the feature with the target. The interpretive threshold of 0.1 reflects the noise floor of hourly return prediction. A Spearman of 0.05 against a return target is at the limit of what survives in production after costs. A Spearman of 0.3 against a return target is unusual and warrants examination for leakage before being treated as edge. The sign carries direction; the magnitude carries strength.

The pair `spearman_ci_lo` and `spearman_ci_hi` give the 95 percent confidence interval from a moving-block bootstrap with 200 iterations. The block length is taken from the column's `block_size` row. Inputs above 20,000 rows are stride-subsampled first. The bootstrap exists to account for serial dependence: an iid resampling would treat adjacent observations as independent when they are not, underestimate the variance of the Spearman estimator, and produce intervals too narrow to be trusted — declaring features significant that are not. The block bootstrap preserves the local correlation structure by resampling chunks of length equal to the decorrelation distance, and the interval that results is the honest one. The diagnostic reading is **whether the interval crosses zero**. A CI of [0.21, 0.25] is a confidently non-zero edge given the dependence. A CI of [-0.05, 0.08] is a feature whose sign is not determined by the sample.

`mutual_information` is the feature-target mutual information (MI), computed with `mutual_info_classif` when the target has 10 or fewer unique values and `mutual_info_regression` otherwise. The threshold of "greater than zero" in the example is a placeholder; MI magnitudes are not interpretable in absolute terms across targets of different entropy. The estimate is also sensitive to sample size, the estimator's nearest-neighbour parameter, the prevalence of repeated values in the feature, preprocessing and scaling, and the target's type. Small differences in MI between two features at the same sample size should not be treated as a stable ranking — they sit inside the estimator's noise. Two uses dominate. Compare MI across features on the same target to rank them by information content. Identify features whose Spearman is near zero but whose MI is positive — the signature of a non-monotonic relationship the rank correlation cannot see.

`decile_spread` is the mean target value in the top feature decile minus the mean target value in the bottom feature decile. The interpretation is direct: a strategy trading only on the top and bottom deciles of the feature would harvest this per-bar mean target spread in-sample, before costs. Above 0.1 in absolute terms is operationally useful for a return target. The metric returns `None` when any decile bin is empty, which is the situation for discrete features. The example shows `is_weekend` blank in this row because nine of its deciles fall on the same value.

`monotonic_fraction` is the fraction of decile-to-decile transitions whose sign matches the overall spread direction. A value of 1.0 means the deciles ascend or descend strictly. Above 0.7 indicates a monotone relationship that the decile spread accurately describes. Below 0.5 indicates a non-monotonic relationship — a U-shape, a threshold effect, or a regime split — for which the spread value understates or misrepresents the information content. A feature with `decile_spread` of 0.05 and `monotonic_fraction` of 0.4 has more usable structure than the spread alone suggests and should be examined through MI before being discarded.

## Target leakage

Spearman or MI substantially above the typical noise floor for the chosen target is the first sign of leakage. The panel cannot diagnose leakage — it can only surface the candidates. Common sources, in declining order of how often they appear in practice:

- **Forward-looking rolling windows.** A rolling statistic centred on the current bar, or a window that extends past the bar's timestamp, includes information unavailable at the decision point.
- **Target-aligned normalisation.** Scaling a feature by a statistic computed on rows that overlap the target window — z-scores computed over the same window the target predicts, percentile ranks computed against future quantiles.
- **Features built from future bars.** Lookback indices that off-by-one into the future; resampling that bleeds the next bar's open into the current bar's close.
- **Calendar dummies encoding dataset boundaries.** Day-of-week or holiday features that happen to align with how the historical data was assembled, gapped, or filtered.
- **Fill methods.** Forward-fill is generally safe; backward-fill or interpolation across the target window leaks future information.
- **Joins on incomplete keys.** Merging external data without lagging it to its publication time; using values that are revised after publication without preserving the original.

Any feature whose edge magnitude exceeds the typical noise floor by a factor of three or more should be audited against this list before being deployed.

## Regime stability

The final two rows partition the time index into four contiguous chunks and recompute the feature-target Spearman within each. A chunk is skipped when fewer than 10 valid pairs survive after dropping NaNs. Both regime metrics return `None` when fewer than two chunks survive.

`regime_sign_agreement` is rendered as a fraction string: the count of chunks whose Spearman has the majority sign, over the number of valid chunks. Four out of four is the stable case — the feature points the same direction in every quarter of the sample. Three out of four signals one regime in which the direction flips, the most operationally consequential pattern to investigate, because it identifies features whose edge is conditional and whose deployment requires a regime gate or an interaction term rather than a standalone position. Two out of four is direction noise.

`regime_spearman_std` is the standard deviation of the per-chunk Spearman values. Below 0.05 means the strength of the edge is stable, not only its sign. A column with 4/4 agreement and a standard deviation of 0.15 carries a consistent direction but with substantial magnitude variation across regimes — the trade exists in every period but its expected return per bar varies by a factor of several. The two rows together distinguish three operational states: stable feature (4/4, low standard deviation), direction-stable but strength-variable feature (4/4, high standard deviation), and regime-conditional feature (less than 4/4 in either direction).

The four-chunk split is fixed at quartiles of the index. It cannot detect a regime change whose timing falls inside a chunk, and it does not find regime structure that does not align with the quartile boundaries. For sub-4/4 features or suspected intra-chunk breaks, the next diagnostic step is a **rolling Spearman or a formal change-point procedure** (for example, a `ruptures`-style break detector on the per-bar correlation series). The panel's regime rows are a coarse stability check, not a regime-detection algorithm.

## Cross-target comparability

Spearman, mutual information, decile spread, and the regime metrics are all target-dependent and horizon-dependent. They are comparable across features when the target and sampling cadence are held fixed. They are not comparable across targets: a Spearman of 0.15 against a 1-bar return and a Spearman of 0.15 against a 24-bar return measure different things, and the same is true of MI across targets of different entropy. The example's `ref` column reflects one target type at one cadence; transplanting it without adjustment is a common source of misreading.

## Performance and determinism

Bootstrap intervals are **reproducible after a fresh import** provided columns are processed in the same order. Mutual information is independently reproducible because each call passes `random_state=0`. The mechanics below matter only when debugging a reproduction failure.

Subsampling. ADF and KPSS subsample to 5,000 rows via uniform stride. Pairwise Spearman uses 20,000 rows by stride; pairwise MI uses 5,000. Edge mutual information uses 20,000. The bootstrap caps its input at 20,000 and runs 200 iterations. Stride subsampling preserves time order but compresses local lag structure, which is the reason ADF and KPSS for long series are treated as indicative rather than authoritative.

Random number generator (RNG) state. The bootstrap RNG is module-scoped: `_RNG = np.random.default_rng(0)` is constructed once at import and advanced by each draw. Identical columns processed in identical order within a single Python process produce identical confidence intervals. The first call after a fresh import reproduces across processes. Subsequent calls reproduce only when the full call sequence is identical, because the RNG state has advanced between them. Reimporting the module resets the state. MI calls use `random_state=0` per call and reproduce independently of the bootstrap RNG.

## Reading the panel — applied

The reading order applied to the example clarifies the metrics. Take `parkinson_volatility_1_bps_per_bar` first. Health: clean, with a heavy tail (skew 3.6, max 726 bps against a p99 of 165). Stationarity: ADF rejects, KPSS rejects — bounded with movement, not a wanderer. Persistence: 0.57 at lag 1 dropping to 0.28 at lag 168 — a persistent volatility process with weak weekly residue. Pairwise: 0.82 against another feature (its sibling at lag 4), 0.63 in MI — redundant. Edge: Spearman 0.23 with CI [0.21, 0.25] not crossing zero, decile spread 0.58, monotonic fraction 1.0 — a clean, monotonic, confidently non-zero edge. Regime: 4/4, std 0.014 — stable. The reading: usable as a feature, but only one of the two volatility columns survives the pairwise step.

`is_weekend` next. Health: discrete, `unique_fraction` near zero. Persistence: 0.97 at lag 1, 0.99 at lag 168 — the canonical weekly-cycle signature. Pairwise: low (max 0.49, mean 0.27) — informative against the rest of the set. Edge: Spearman -0.20 with CI not crossing zero. Decile spread and monotonic fraction blank because nine of ten deciles collapse onto a single value. Regime: 4/4, stable. The blank decile cells are not a failure; they are the only correct response for a binary column. Read the Spearman and MI and ignore the decile rows.

`nth_hour` is the cautionary case. Persistence: 0.998 at lag 24, 0.99 at lag 168 — perfect daily and weekly cycles. ADF does not reject (0.96), KPSS does not reject (0.10) — the test settings cannot decide, and given the construction the underlying answer is bounded-with-cycle rather than wandering. Pairwise: 0.20 max, 0.07 max MI — orthogonal to everything. Edge: Spearman 0.06 with CI [0.05, 0.08] not crossing zero, but `decile_spread` is -0.05 and `monotonic_fraction` is 0.67. The edge is real but small, non-monotonic, and almost certainly a calendar effect rather than a tradable signal. The feature is a candidate for an interaction term, not a standalone.

`liquidity_z_96` is the strongest single row. Spearman 0.29 with CI [0.27, 0.30], decile spread 0.71 with monotonic fraction 1.0, 4/4 regime agreement with std 0.011. Health and stationarity are clean. The redundancy row is the only caveat: `max_pairwise_spearman` 0.73 against another feature. The action is to identify the twin and select the representative before deployment.

## The candidate deployable set

The features that pass all six layers form the **candidate** deployable set. They still require out-of-sample validation, leakage review, cost-aware testing, and live monitoring before production use. Those that fail the regime layer but pass the rest are candidates for conditional use through a regime gate or interaction term. Those that pass health and pairwise but fail edge are noise against the chosen target — not against every target. Those that fail health are not features.

## Known limitations

The panel can mislead when its assumptions break. Uniform sampling is assumed throughout: gaps and irregular spacing degrade every lag-based metric, because the ACF then measures persistence across bars of varying real-time duration. The ADF implementation trades power for determinism — borderline columns warrant an external check with `autolag` enabled and a longer maxlag. The moving-block bootstrap captures dependence up to the block length and no further; dependence at lags above the block leaks into the confidence interval as artificial narrowness.

Edge metrics are univariate. The panel does not control for confounders, does not condition on other features, and does not detect interactions. A feature whose edge is entirely explained by another feature in the set will still display a strong Spearman against the target; the pairwise redundancy rows are the only signal that this is the case, and they are necessary but not sufficient. Mutual information carries finite-sample bias that depends on sample size and discretisation; small differences across columns at the same n should not be over-interpreted.

The regime split is fixed at quartiles of the index. Real regime structure that does not align — a single break two-thirds through the sample, for instance, or a smooth drift — is invisible to the metric. The two regime rows answer the question they ask, which is narrower than the question the operator may want answered.