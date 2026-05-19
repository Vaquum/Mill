# QC report — operator manual

**Purpose.** Per-feature diagnostic for time-series feature sets. Reports column health, pairwise redundancy, edge against an optional target, and stability across time. One DataFrame in, one table out. Read column-wise to decide whether a feature is kept, transformed, clustered with neighbours, or dropped.

**Invocation.** `report(df, target)`. `target` may be `None`; edge and regime rows are populated only when a target is given and present in `df`.

## Input

- `df`: polars DataFrame, time-ordered, hourly bars assumed by the ACF lag set (1, 5, 24, 168). A `datetime` column is excluded from analysis.
- `target`: column name (string) or `None`.

Non-Float columns are cast to Float64. Cast failures become NaN.

## Output

`{"columns": [...], "rows": [...]}`. Columns are feature names, followed by the target column when present. Rows are metric records in fixed order: **health → block_size → pairwise → edge → regime**. A cell is `None` when the metric is undefined for that column (see *Cell conventions*).

## Health — per column

Computed on the column's non-NaN values.

`missing_rate` — null fraction. **<0.01 acceptable**. Above a few percent the column should be repaired or dropped before downstream use.

`unique_fraction` — distinct-value count over n. **>0.9 continuous**. Near zero indicates a constant, near-constant, or low-cardinality categorical.

`min`, `max`, `median`, `p01`, `p99` — distribution markers from non-NaN values. p01/p99 give tail behaviour without the noise of extrema. Check against the column's expected physical range.

`skew` — bias-corrected sample skewness. **|skew| < 1 symmetric**. Above 2, the column has a heavy tail — apply log or signed-log before modelling.

`adf_p` — Augmented Dickey-Fuller p-value. Tests the null of a unit root. Fixed `maxlag=10`, `autolag=None`. **<0.05 rejects unit root** → stationary. Subsampled to 5,000 via stride above that size; treat values for long series as indicative. Returns `None` when n < 50.

`kpss_p` — KPSS p-value. Tests the null of stationarity around a constant. `regression="c"`, `nlags=10`. **>0.05 fails to reject stationarity**. Same subsampling and n<50 rule as ADF.

**Joint reading.** ADF rejects + KPSS does not = stationary. Both reject = trend-stationary or borderline. ADF does not reject + KPSS rejects = non-stationary; difference or detrend before later metrics are trusted.

`acf_lag_{1, 5, 24, 168}` — autocorrelation at fixed lags via `statsmodels.acf(fft=True)`. Returns `None` when n ≤ lag+1.

- lag 1: **>0.5 persistent**. Below 0.2 = noise-like.
- lag 5: **>0.3 persistent** at the 5-hour horizon.
- lag 24: **>0.2 daily cycle**.
- lag 168: **>0.2 weekly cycle**.

`block_size` — decorrelation length estimate. Walks |ACF| from lag 1 to lag 200 and returns the first lag where |ACF| < 0.05. Falls back to 24 if ACF fails or no lag clears the threshold. Used to size the moving-block bootstrap. Range 1–200 bars.

## Pairwise — feature against other features

Computed on rows where **all features** are simultaneously non-NaN. Spearman subsampled to 20,000 rows by linspace stride; MI to 5,000. All four metrics return `None` when fewer than 30 clean rows survive.

`max_pairwise_spearman` — max |Spearman ρ| against any other feature. **<0.3 unique**. ≥0.7 = redundant; collapse the cluster and keep one representative.

`mean_pairwise_spearman` — mean |Spearman ρ| across the other features. Captures whether the feature is broadly correlated with the set or only locally.

`redundancy_penalty` — `1 − max_pairwise_spearman`. **>0.7 unique**. The inverse view of the max — useful when ranking features for inclusion.

`max_pairwise_mi` — max mutual information against any other feature, via `mutual_info_regression`. Catches non-linear redundancy Spearman misses. **<0.1 independent**.

## Edge — feature against target

Populated only when target is provided. Per-column edge requires ≥100 rows where both feature and target are non-NaN; below that threshold all six edge values for the column are `None`. Empty in the target's own column by construction.

`spearman` — rank correlation with target. **|ρ| > 0.1 strong** in this context. Magnitudes much above 0.3 in raw single-feature edge against a return target warrant a look for leakage.

`spearman_ci_lo`, `spearman_ci_hi` — 95 % CI from moving-block bootstrap. Block size taken from the column's `block_size`. 200 iterations. Input subsampled to 20,000 first. Seeded from `np.random.default_rng(0)` at module import. **CI not crossing zero** = edge is meaningfully nonzero given the dependence structure.

`mutual_information` — feature ↔ target MI. Auto-detects target type: ≤10 unique target values → `mutual_info_classif`, otherwise `mutual_info_regression`. Subsampled to 20,000. Seeded (`random_state=0`). Magnitude depends on target entropy — **compare across features, not against an absolute threshold**.

`decile_spread` — mean(target | top decile) − mean(target | bottom decile). **|d| > 0.1 useful**. Returns `None` when any decile bin is empty — discrete or binary features collapse here. In the example, `is_weekend` is blank for this reason.

`monotonic_fraction` — fraction of the 9 decile-to-decile transitions whose sign matches the overall spread sign. **>0.7 monotone**. 1.0 = strictly monotonic. Below 0.5 indicates a U-shape or noisy non-monotonic relationship; the spread metric overstates the edge in that case.

## Regime — stability across time

Data partitioned into 4 contiguous chunks by index. Spearman against target is computed per chunk on non-NaN pairs. A chunk is skipped when fewer than 10 valid pairs survive; both regime metrics fall back to `None` when fewer than 2 chunks survive.

`regime_sign_agreement` — `max(positive_chunks, negative_chunks) / valid_chunks`, rendered as a fraction string. **4/4 stable**. 3/4 = one regime flip — investigate before deployment. 2/4 = no consistent direction.

`regime_spearman_std` — standard deviation of the per-chunk Spearman values. **<0.05 stable**. 4/4 agreement with high std = direction holds, strength fluctuates.

## Cell conventions

`None` indicates: an exception caught by `_safe`, a NaN/Inf result, sample size below threshold, zero-variance input, or an empty decile bin. **`None` is not zero**. A numeric 0 is a real measurement.

Blanks in the target column under pairwise and edge rows are by construction.

## Performance and reproducibility

| operation | cap | sampling | seeded |
|---|---|---|---|
| ADF / KPSS | 5,000 | linspace stride | deterministic |
| Pairwise Spearman | 20,000 | linspace stride | deterministic |
| Pairwise MI | 5,000 | linspace stride | `random_state=0` |
| Bootstrap CI | 20,000 input, 200 iter | linspace stride + block resample | `default_rng(0)` |
| Edge MI | 20,000 | linspace stride | `random_state=0` |

Stride subsampling preserves time order but breaks local lag structure at scales below the stride; ADF/KPSS for very long series are indicative.

**Bootstrap RNG is module-scoped.** A single `_RNG = np.random.default_rng(0)` is created at import and advanced by each draw. Identical input columns processed in identical order within a single Python process produce identical CIs. Across processes the first call reproduces exactly; later calls reproduce only if the call sequence is identical. Reimport the module to reset.

## Workflow

1. **Health pass.** Drop columns with missing_rate above tolerance, unique_fraction near zero, or p01 = p99.
2. **Stationarity pass.** Read ADF/KPSS jointly. Difference or detrend non-stationary columns before edge metrics carry signal.
3. **Persistence pass.** The ACF profile is the feature's character. Match against the model horizon — short-memory, daily, weekly.
4. **Redundancy pass.** Cluster features with `max_pairwise_spearman ≥ 0.7`; within each cluster keep the one with the strongest edge.
5. **Edge pass.** Require `spearman_ci_lo` and `spearman_ci_hi` of the **same sign**, `|decile_spread| > 0.1`, and `monotonic_fraction > 0.7`. MI is the cross-check for non-linear edge the rank metrics miss.
6. **Stability pass.** Require `regime_sign_agreement = 4/4` for any feature intended to stand alone. Anything weaker is regime-conditional — gate it or interact it.

## Known limitations

- ACF and lag-based metrics assume uniform sampling. Gaps and irregular spacing degrade lag interpretation.
- ADF with `autolag=None, maxlag=10` is a deliberate speed/consistency choice, not the most powerful test. Borderline columns warrant an external check.
- Moving-block bootstrap captures dependence up to block size only. Dependence at lags greater than block size leaks into the CI as artificial narrowness.
- Edge metrics are univariate. No control for confounders, no joint conditioning.
- The 4-chunk regime split is fixed by index, not detected. Real regime structure that does not align with the quartile boundaries is missed.