QC report — field guide for the operator

`qc.report(df, target)` produces a single per-column diagnostic table. Columns are the features, with the target appended when supplied. Rows are metrics organised into five groups in deliberate order: distributional **HEALTH**, **STATIONARITY**, **REDUNDANCY**, target **EDGE**, and **REGIME** stability. Each cell answers one question about one column. The panel is read column-wise to decide what becomes of the feature — kept as it stands, transformed, clustered with neighbours, gated by regime, or dropped.

Reference ranges are not in the report payload. They live in the UI as per-row tooltips (`QC_META.tip` in `index.html`) and as the grading thresholds in `grading_rules.py`. The metric structure is invariant; the interpretive scale (and the grading ladder bound to it) is calibrated for hourly bar data with a return-style target. A different target or sampling cadence requires recalibrating the ladders.

## Read the panel as a funnel

The five groups form a sequence, not a checklist. Reject unusable data first (HEALTH). Stabilise drifting series next (STATIONARITY). Collapse redundant features (REDUNDANCY). Test the target edge (EDGE). Check regime stability (REGIME). A feature is not attractive because it has one good row; it is attractive because it survives the sequence. The order matters because each group's metrics are only meaningful once the prior groups have passed — a strong Spearman on a non-stationary feature is co-trending until proven otherwise, and a confident bootstrap confidence interval (CI) on a feature riddled with NaNs measures a subsample whose representativeness has not been established.

## Operating assumptions

The defaults are calibrated to hourly bar data, a return-style target, uniform sampling across the index, and a minimum of 100 paired non-NaN rows per feature-target pair for the edge metrics. Gaps and irregular spacing degrade every lag-based metric. Targets that are not returns — classification labels, fill ratios, hold times — preserve the metric structure but invalidate the example thresholds.

## Grades

The `/qc` endpoint ships a `grades` map alongside the metric rows. Each per-feature grade is one of `A` / `B` / `D` / `F` (no `C` — `D` is the middling grade and `B` is the next step up). The thresholds live in `grading_rules.py`; the composite verdicts (`health`, `edge`, `selection`) live in `grading.py`. The UI renders a per-cell overlay using the row→grade mapping in `grading_rules.GRADE_TO_METRIC_ROWS`. Grades are an aid for fast scanning; the metric values remain the primary evidence.

## Decision table

| Group | Red flag | Action |
|---|---|---|
| HEALTH       | `missing_rate` above tolerance | repair or drop |
| HEALTH       | very low `unique_fraction` or `n_unique = 1` | treat as discrete or drop |
| STATIONARITY | ADF does not reject and KPSS rejects | difference or detrend before edge metrics |
| REDUNDANCY   | `max_pairwise_spearman` ≥ 0.7 | cluster; keep one representative |
| EDGE         | Spearman CI crosses zero | do not treat the sign as stable |
| EDGE         | low `monotonic_fraction` | inspect for non-linearity or regime effects |
| REGIME       | sign agreement below 4/4 | gate the feature or use it conditionally |

The table is the operator's first read. The sections below are the prose behind each row.

## How the panel marks the absent

A cell holds `None` when the metric is undefined for that column, not zero. The conditions are specific: an exception caught inside the safe-evaluation wrapper, a NaN or Inf result, a sample below the per-metric minimum, zero variance in the input, or, in the case of decile spread, an empty quantile bin. A numeric zero is a real measurement; **`None` is not**. A column showing `None` for `decile_spread` and `monotonic_fraction` while showing a finite Spearman has not failed; its rank correlation succeeded and its decile decomposition failed because the column is discrete and several of its deciles fall on the same value.

The target's own column under the pairwise and edge rows is blank by construction. The target does not compete with itself in either bucket.

## HEALTH

`missing_rate` is the fraction of bar values missing **after** a Float64 cast — Polars nulls plus any value that becomes NaN through casting. Below one percent is operationally invisible. Above a few percent the column should be repaired or excluded before any downstream metric is trusted, because every subsequent row drops NaNs before computing — a 30 percent missing rate means stationarity, correlation, and edge all rest on a 70 percent subsample whose representativeness is not established. The pairwise metrics are **more vulnerable still**: they require rows where every feature is simultaneously non-NaN, so a single column with heavy missingness reduces the effective sample for all of them.

`n_unique` is the count of distinct non-NaN values; it feeds the cardinality grade alongside the report's top-level `n_rows`. `unique_fraction` is `n_unique / n_rows`. Continuous features sit near 1.0; discrete or low-cardinality features sit near 0. The reading interacts with later metrics: a feature with `unique_fraction` near zero may return `None` for decile spread because the deciles collapse onto each other.

The five distributional markers — `min`, `max`, `median`, `p01`, `p99` — describe the spread on non-NaN values. The percentile pair is the working description of the tails. Compare each against the column's expected physical range. A z-score that drifts to ±8 has either tail mass that ordinary scaling cannot tame or a clipping bound that has come loose.

`skew` is the bias-corrected sample skewness. Symmetric distributions sit near zero. Above roughly two, the column has a tail that will dominate ordinary regression and bias any metric that relies on means. A log or signed-log transform is the standard remedy before downstream use. Parkinson volatilities, absolute z-scores, and absolute normalised return all show skew above 2.5; that is the expected signature of those constructions.

## STATIONARITY

`adf_p` is the Augmented Dickey-Fuller p-value. The null hypothesis is that the series has a unit root — that its level wanders without a stable mean to return to. A p-value below 0.05 rejects the wandering and concludes the series is stationary. The implementation fixes `maxlag = 10` and disables automatic lag selection, a deliberate trade of statistical power for runtime determinism. Series longer than 5,000 observations are stride-subsampled to that length, so the test for long series is indicative rather than authoritative.

`kpss_p` is the KPSS p-value with the inverse null: stationarity around a constant. A p-value above 0.05 fails to reject stationarity. The same subsampling and minimum-sample rule applies. The KPSS implementation reports a floor at 0.01 and a ceiling at 0.10; those values are the test's expressive limits, not precise probabilities.

The two tests carry their weight together, not separately. ADF rejecting and KPSS not rejecting is the clean stationary case under these test settings. Both rejecting indicates trend-stationarity or a borderline case in which level moves but the moves are bounded. ADF not rejecting and KPSS rejecting indicates a genuinely drifting series, and every metric below — Spearman, mutual information, decile spread — should be treated as describing a wandering relationship rather than a stable one. A strong correlation between two non-stationary series is, by default, **a co-trending artifact** until proven otherwise. The grading layer (`grading.STATIONARITY_TABLE`) encodes the cross of the two sub-grades; the special case ADF=A AND KPSS=F maps to D with the note `difference-stationary, possibly trending`.

A bootstrap block length is computed internally from the column's autocorrelation and used to drive the Spearman CI; it is not surfaced as a row in the report.

## REDUNDANCY

Four rows describe how the column relates to the other features in the set. They are computed on rows where every feature is simultaneously non-NaN, which is the strictest possible filter — a single column with heavy missingness reduces the effective sample for all pairwise metrics. Inputs are stride-subsampled to 20,000 rows for Spearman and 5,000 for mutual information. When fewer than 30 fully clean rows survive the joint filter, all four pairwise values for every column return `None`.

`max_pairwise_spearman` is the largest absolute rank correlation against any other feature. Below roughly 0.3, the column is unlikely to have a close rank-monotonic twin in this feature set. At or above 0.7, the column has a near-twin and the pair should be collapsed by selecting one representative. `mean_pairwise_spearman` is the same quantity averaged over all other features. It distinguishes a column with one near-twin (high max, low mean) from one broadly correlated with the whole set (high mean).

`redundancy_penalty` is `1 − max_pairwise_spearman`. It is the same information as the max, inverted for ranking purposes: higher values mean more unique. Above 0.7 indicates a column that no other column captures.

`max_pairwise_mi` is the maximum mutual information against any other feature, via `mutual_info_regression`. It catches non-linear redundancy that Spearman misses: two features may be rank-uncorrelated yet jointly determined by a common non-monotonic driver. Below 0.1 indicates near-independence in the information-theoretic sense.

A single-feature view has no peers, so all four redundancy rows return `None` for it. The grading layer treats a missing redundancy grade as a non-issue (no evidence of overlap) rather than a failure.

## EDGE against the target

The next six rows are populated only when a target is supplied and present in the DataFrame. Each is computed on rows where the feature and the target are simultaneously non-NaN. The per-column minimum is 100 paired rows; below that threshold all six edge values for the column return `None`. The target's own column is blank by construction.

`spearman` is the rank correlation of the feature with the target. The interpretive threshold of 0.1 reflects the noise floor of hourly return prediction. A Spearman of 0.05 against a return target is at the limit of what survives in production after costs. A Spearman of 0.3 against a return target is unusual and warrants examination for leakage before being treated as edge. The sign carries direction; the magnitude carries strength.

The pair `spearman_ci_lo` and `spearman_ci_hi` give the 95 percent confidence interval from a moving-block bootstrap with 200 iterations. The block length is resolved internally from the column's autocorrelation (capped at 200 lags; falls back to 24 when the ACF does not clear the threshold or computation fails). Inputs above 20,000 rows are stride-subsampled first. The bootstrap exists to account for serial dependence: an iid resampling would treat adjacent observations as independent when they are not, underestimate the variance of the Spearman estimator, and produce intervals too narrow to be trusted. The block bootstrap preserves the local correlation structure by resampling chunks of length equal to the decorrelation distance, and the interval that results is the honest one. The diagnostic reading is **whether the interval crosses zero**. A CI of [0.21, 0.25] is a confidently non-zero edge. A CI of [-0.05, 0.08] is a feature whose sign is not determined by the sample.

`mutual_information` is the feature-target mutual information (MI), computed with `mutual_info_classif` when the target has 10 or fewer unique values and `mutual_info_regression` otherwise. MI magnitudes are not interpretable in absolute terms across targets of different entropy. The estimate is also sensitive to sample size, the estimator's nearest-neighbour parameter, the prevalence of repeated values in the feature, preprocessing and scaling, and the target's type. Small differences in MI between two features at the same sample size should not be treated as a stable ranking — they sit inside the estimator's noise. Two uses dominate. Compare MI across features on the same target to rank them by information content. Identify features whose Spearman is near zero but whose MI is positive — the signature of a non-monotonic relationship the rank correlation cannot see.

`decile_spread` is the mean target value in the top feature decile minus the mean target value in the bottom feature decile. The interpretation is direct: a strategy trading only on the top and bottom deciles of the feature would harvest this per-bar mean target spread in-sample, before costs. Above 0.1 in absolute terms is operationally useful for a return target. The metric returns `None` when any decile bin is empty, which is the situation for discrete features.

`monotonic_fraction` is the fraction of decile-to-decile transitions whose sign matches the overall spread direction (range `[0, 1]`). A value of 1.0 means the deciles ascend or descend strictly. Above 0.7 indicates a monotone relationship that the decile spread accurately describes. Below 0.5 indicates a non-monotonic relationship — a U-shape, a threshold effect, or a regime split — for which the spread value understates or misrepresents the information content. The grading ladder is monotone in `frac` (higher is better); there is no symmetric "extremes-good" interpretation.

## Target leakage

Spearman or MI substantially above the typical noise floor for the chosen target is the first sign of leakage. The panel cannot diagnose leakage — it can only surface the candidates. Common sources, in declining order of how often they appear in practice:

- **Forward-looking rolling windows.** A rolling statistic centred on the current bar, or a window that extends past the bar's timestamp, includes information unavailable at the decision point.
- **Target-aligned normalisation.** Scaling a feature by a statistic computed on rows that overlap the target window — z-scores computed over the same window the target predicts, percentile ranks computed against future quantiles.
- **Features built from future bars.** Lookback indices that off-by-one into the future; resampling that bleeds the next bar's open into the current bar's close.
- **Calendar dummies encoding dataset boundaries.** Day-of-week or holiday features that happen to align with how the historical data was assembled, gapped, or filtered.
- **Fill methods.** Forward-fill is generally safe; backward-fill or interpolation across the target window leaks future information.
- **Joins on incomplete keys.** Merging external data without lagging it to its publication time; using values that are revised after publication without preserving the original.

Any feature whose edge magnitude exceeds the typical noise floor by a factor of three or more should be audited against this list before being deployed.

## REGIME stability

The final two rows partition the time index into four contiguous chronological chunks (the input is sorted by `datetime` at the top of `report()` so chunks are always in order regardless of the caller's row order) and recompute the feature-target Spearman within each. A chunk is skipped when fewer than 10 valid pairs survive after dropping NaNs. Both regime metrics return `None` when fewer than two chunks survive.

`regime_sign_agreement` is rendered as a fraction string: the count of chunks whose Spearman has the majority sign, over the number of valid chunks. Four out of four is the stable case — the feature points the same direction in every quarter of the sample. Three out of four signals one regime in which the direction flips, the most operationally consequential pattern to investigate, because it identifies features whose edge is conditional and whose deployment requires a regime gate or an interaction term rather than a standalone position. Two out of four is direction noise.

`regime_spearman_std` is the standard deviation of the per-chunk Spearman values. Below 0.05 means the strength of the edge is stable, not only its sign. A column with 4/4 agreement and a standard deviation of 0.15 carries a consistent direction but with substantial magnitude variation across regimes — the trade exists in every period but its expected return per bar varies by a factor of several. The two rows together distinguish three operational states: stable feature (4/4, low standard deviation), direction-stable but strength-variable feature (4/4, high standard deviation), and regime-conditional feature (less than 4/4 in either direction).

The four-chunk split is fixed at quartiles of the index. It cannot detect a regime change whose timing falls inside a chunk, and it does not find regime structure that does not align with the quartile boundaries. For sub-4/4 features or suspected intra-chunk breaks, the next diagnostic step is a **rolling Spearman or a formal change-point procedure** (for example, a `ruptures`-style break detector on the per-bar correlation series). The panel's regime rows are a coarse stability check, not a regime-detection algorithm.

## Cross-target comparability

Spearman, mutual information, decile spread, and the regime metrics are all target-dependent and horizon-dependent. They are comparable across features when the target and sampling cadence are held fixed. They are not comparable across targets: a Spearman of 0.15 against a 1-bar return and a Spearman of 0.15 against a 24-bar return measure different things, and the same is true of MI across targets of different entropy. Recalibrate the grading ladders before transplanting them to a different target/cadence.

## Performance and determinism

Bootstrap intervals are deterministic per column. `_bootstrap_spearman_ci` builds a fresh `np.random.default_rng` per call seeded from a stable hash of the column name (`zlib.crc32`), so the CI for a given column is reproducible regardless of how many other columns were processed before it, the order they were processed in, or whether the module has been reimported. Mutual information is independently reproducible because each call passes `random_state=0`.

Subsampling. ADF and KPSS subsample to 5,000 rows via uniform stride. Pairwise Spearman uses 20,000 rows by stride; pairwise MI uses 5,000. Edge mutual information uses 20,000. The bootstrap caps its input at 20,000 and runs 200 iterations. Stride subsampling preserves time order but compresses local lag structure, which is the reason ADF and KPSS for long series are treated as indicative rather than authoritative.

## The candidate deployable set

Features that pass all five groups form the **candidate** deployable set. They still require out-of-sample validation, leakage review, cost-aware testing, and live monitoring before production use. Those that fail REGIME but pass the rest are candidates for conditional use through a regime gate or interaction term. Those that pass HEALTH and REDUNDANCY but fail EDGE are noise against the chosen target — not against every target. Those that fail HEALTH are not features.

## Known limitations

The panel can mislead when its assumptions break. Uniform sampling is assumed throughout: gaps and irregular spacing degrade every lag-based metric, because the ACF then measures persistence across bars of varying real-time duration. The ADF implementation trades power for determinism — borderline columns warrant an external check with `autolag` enabled and a longer maxlag. The moving-block bootstrap captures dependence up to the block length and no further; dependence at lags above the block leaks into the confidence interval as artificial narrowness.

Edge metrics are univariate. The panel does not control for confounders, does not condition on other features, and does not detect interactions. A feature whose edge is entirely explained by another feature in the set will still display a strong Spearman against the target; the redundancy rows are the only signal that this is the case, and they are necessary but not sufficient. Mutual information carries finite-sample bias that depends on sample size and discretisation; small differences across columns at the same n should not be over-interpreted.

The regime split is fixed at quartiles of the index. Real regime structure that does not align — a single break two-thirds through the sample, for instance, or a smooth drift — is invisible to the metric. The two regime rows answer the question they ask, which is narrower than the question the operator may want answered.
