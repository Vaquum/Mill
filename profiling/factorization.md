# Mill factorization: measured results

Reference: `847cd481e82b1000496db082e52acb4d9794c207`. Exact engine and driver fingerprints, raw observations, CPU times and output digests are in [factorization.json](factorization.json) and [the run matrix](results/factorization/jobs.json).

This comparison holds Limen 3.30.0 (`86968e44022cabcae83fc38fe2ed0c7105816fa2`) fixed on both sides to isolate Mill changes.

## Matched timings

Milliseconds; median of three independent process medians. Parenthesized values are the range of those three observations, not confidence intervals. Before/after order alternates. Full source histories contain 20,784 or 83,136 rows; no computation row limit was used.

| Workload | Before ms (range) | After ms (range) | Ratio |
|---|---:|---:|---:|
| Eight views, first build + target + JSON | 924.25 (840.68–1,061.33) | 534.90 (479.54–551.56) | 1.73× |
| Eight views, repeated cold app caches | 1,029.94 (865.02–1,136.91) | 756.19 (498.84–1,017.16) | 1.36× |
| Eight views, cached frames/payloads | 0.17 (0.15–0.22) | 0.16 (0.16–0.50) | 1.01× |
| QC, 31 columns / 465 pairs, 83k rows | 15,441.66 (14,976.47–15,846.67) | 6,032.06 (5,906.05–6,371.50) | 2.56× |
| QC, 31 columns / 465 pairs, 20k rows | 30,992.36 (15,463.96–32,689.86) | 5,900.72 (5,869.09–13,958.37) | 5.25× |
| QC, warm content kernels, 83k rows | 36.19 (35.79–38.16) | 37.32 (37.27–38.22) | 0.97× |
| Discovery, 259 columns / 83k rows, cold search | 7,248.03 (7,037.72–7,552.40) | 3,806.62 (3,637.49–6,138.20) | 1.90× |
| Discovery, 259 columns / 20k rows, cold search | 2,550.93 (1,766.60–2,867.81) | 1,632.75 (1,299.91–1,639.63) | 1.56× |
| Discovery, repeat, 20k rows | 2,316.86 (2,306.59–2,925.43) | 7.31 (7.09–8.68) | 316.80× |
| Discovery, display change, 20k rows | 2,331.76 (1,730.67–2,511.31) | 3.46 (2.88–3.51) | 673.73× |
| Discovery, changed windows, 20k rows | 2,294.53 (1,691.88–2,831.83) | 889.45 (707.03–1,089.92) | 2.58× |
| Synthetic source advance, prebuilt 20k pool | 3,393.90 (2,523.71–7,028.40) | 107.08 (80.29–146.98) | 31.69× |
| Full catalog preparation, 83k rows | 24,778.56 (24,550.75–31,827.71) | 26,728.28 (23,843.83–29,020.39) | 0.93× |
| Persisted catalog mmap hit, 83k rows | 18.95 (13.76–20.11) | 18.12 (14.47–22.71) | 1.05× |
| View research service, cold, 83k rows | 8,313.16 (6,934.38–11,711.30) | 4,644.74 (4,523.85–5,765.16) | 1.79× |
| View research service, repeat, 83k rows | 5,030.90 (4,446.72–5,689.53) | 29.65 (29.00–35.41) | 169.70× |
| View research service, changed windows, 83k rows | 3,572.04 (3,487.46–4,326.04) | 1,140.58 (972.79–1,159.70) | 3.13× |
| Full catalog research service, cold, 20k rows | 17,768.31 (9,089.43–24,104.34) | 10,900.80 (10,347.96–14,104.62) | 1.63× |
| HTTP /data, first request, 83k rows | 653.85 (492.32–701.98) | 510.22 (483.95–794.85) | 1.28× |
| HTTP /data, four identical cold callers | 140.01 (96.63–317.21) | 67.06 (57.27–138.55) | 2.09× |

The eight-view workload includes GetData loading from a private Arrow fixture, feature construction, target joins and JSON output. First-build rows start after imports and fixture setup; they are not total program-launch timings. Warm results already bypassed numerical work in the reference; no universal speedup is claimed for those hits.

Research-service rows include the real research process, pool preparation, IPC and JSON response transport, through the same entry point as /discover. They exclude HTTP socket/authentication. The view workload uses 30 factories; the catalog workload uses every eligible factory. The HTTP workload separately measures actual socket requests and records 1/4/8-client samples plus /data latency during QC.

**Source-advance timings are pure-kernel measurements on an explicitly synthetic, prebuilt advanced pool.** One final feature row is repeated outside the unchanged historical context. Pool extension/setup is recorded separately. This does not measure a new market bar or include rebuilding its feature catalog. Cold catalog preparation remains a major cost.

## Setup

Each entry below is before / after, in milliseconds. These separately measured phases are not presented as an end-to-end sum. Existing on-disk compiled caches are present; compiled-bootstrap setup includes loading/warmup rather than a controlled first-install compilation.

| Workload | Imports ms | Source decode ms | Fixture preparation ms |
|---|---:|---:|---:|
| workload-83k | 2,988.21 / 2,802.86 | 46.97 / 18.71 | 4.69 / 3.93 |
| qc-20k | 2,507.39 / 2,062.80 | 44.71 / 7.50 | 20.50 / 2.25 |
| qc-83k | 2,482.15 / 2,474.67 | 19.04 / 18.24 | 3.83 / 3.79 |
| discover-full-83k | 1,826.93 / 1,832.19 | 21.50 / 28.21 | 4.34 / 7.00 |
| discover-20k | 2,691.24 / 2,629.42 | 9.98 / 24.32 | 7.85 / 13.23 |
| transition-20k | 3,051.95 / 3,605.18 | 36.51 / 18.15 | 9.03 / 9.61 |
| pool-83k | 3,026.41 / 3,265.58 | 27.08 / 21.63 | 9.06 / 5.53 |
| research-discover-83k | 2,547.30 / 2,348.99 | 19.31 / 20.96 | 4.34 / 5.24 |
| http-83k | 2,370.26 / 3,406.40 | 20.29 / 41.38 | 4.59 / 5.23 |
| catalog-service-20k | 3,576.58 / 2,600.95 | 58.62 / 11.89 | 14.33 / 7.94 |

## CPU and memory

Peak RSS covers the whole worker, including imports, fixtures and output validation; it is not an isolated kernel working-set measurement. Research child peaks are recorded separately and must not be added to parent peaks as simultaneous consumption.

| Workload | Before peak MiB (range) | After peak MiB (range) |
|---|---:|---:|
| workload-83k | 716.42 (712.38–717.55) | 837.91 (803.86–846.06) |
| qc-20k | 482.62 (451.44–483.22) | 534.20 (530.47–534.27) |
| qc-83k | 520.34 (520.14–521.64) | 539.98 (539.88–540.27) |
| discover-full-83k | 1,061.73 (1,040.19–1,075.20) | 801.22 (773.94–804.33) |
| discover-20k | 526.39 (524.22–526.64) | 455.59 (454.86–456.81) |
| transition-20k | 524.28 (519.77–526.08) | 492.12 (487.58–492.14) |
| pool-83k | 504.83 (503.33–505.50) | 522.88 (512.67–523.94) |
| research-discover-83k | 306.80 (306.47–307.78) | 309.66 (305.22–310.06) |
| http-83k | 421.80 (420.05–440.25) | 488.17 (480.80–495.45) |
| catalog-service-20k | 278.58 (278.50–279.75) | 280.02 (278.98–280.34) |

| Cold operation | Before CPU ms | After CPU ms |
|---|---:|---:|
| Eight views, first build + target + JSON | 942.07 | 753.00 |
| Eight views, repeated cold app caches | 910.74 | 805.59 |
| QC, 31 columns / 465 pairs, 83k rows | 15,768.32 | 6,466.36 |
| QC, 31 columns / 465 pairs, 20k rows | 15,559.35 | 5,696.80 |
| QC, warm content kernels, 83k rows | 38.37 | 39.23 |
| Discovery, 259 columns / 83k rows, cold search | 7,785.41 | 4,611.91 |
| Discovery, 259 columns / 20k rows, cold search | 1,852.60 | 1,201.34 |
| Discovery, repeat, 20k rows | 1,818.71 | 6.51 |
| Discovery, display change, 20k rows | 1,814.90 | 2.69 |
| Discovery, changed windows, 20k rows | 1,832.38 | 680.66 |
| Synthetic source advance, prebuilt 20k pool | 1,799.39 | 50.29 |
| Full catalog preparation, 83k rows | 19,214.18 | 19,314.22 |
| Persisted catalog mmap hit, 83k rows | 2.71 | 2.94 |

## Ablations

Each row uses three fresh, adjacent control/disabled process pairs, alternating order. Values are milliseconds. Components interact; their ratios must not be multiplied or summed. QC ablations use nine columns / 36 pairs over the full 20,784-row snapshot and show a second cold computation after JIT/import setup, with the content-kernel cache cleared. First cold computation and setup remain in the raw files. FFT-mask, compact-rank and AUC ablations use all 259 columns and 83,136 rows; rule/reuse/context ablations use the complete 20,784-row pool. An initial 20k null ablation is retained as a diagnostic, separately from the three full-size pairs.

| Disabled component | Measured operation | Optimized control ms | Disabled ms | Disabled/control | Control/disabled CPU ms |
|---|---|---:|---:|---:|---:|
| auc | discover_full | 5,975.53 | 12,607.36 | 2.11× | 4,805.15 / 7,578.34 |
| bootstrap | qc_precompute_second_cold | 1,126.81 | 1,859.80 | 1.65× | 973.32 / 1,229.47 |
| compact | discover_full | 5,866.64 | 5,563.06 | 0.95× | 4,724.24 / 4,928.78 |
| context | discover_source_advance | 99.17 | 1,252.44 | 12.63× | 47.96 / 1,045.78 |
| edges | qc_precompute_second_cold | 992.58 | 1,337.51 | 1.35× | 942.12 / 1,214.44 |
| json | all_views_cold_repeated | 543.27 | 1,156.99 | 2.13× | 751.76 / 884.58 |
| native | all_views_cold_repeated | 722.50 | 1,091.57 | 1.51× | 796.87 / 781.17 |
| null | discover_full | 5,798.77 | 6,002.84 | 1.04× | 4,870.84 / 5,840.19 |
| pairs | qc_precompute_second_cold | 930.48 | 1,334.76 | 1.43× | 914.00 / 1,235.33 |
| reuse | discover_repeat | 13.23 | 1,472.51 | 111.28× | 8.38 / 1,108.62 |
| rules | discover_repeat | 14.49 | 50.63 | 3.49× | 7.02 / 24.09 |

Native expression sharing demonstrably removes duplicate graph nodes, but this view workload shows no clear CPU saving. The encoder reduces Python row/cell materialization while increasing measured whole-worker peak memory. Both tradeoffs are retained explicitly rather than attributed to every workload.

## Removed work and dependency boundaries

- Native producer expressions share identical dependency nodes; opaque operations preserve their eager behavior and errors. Targets are calculated once per trusted source snapshot rather than once per view.
- QC shares exact row-cohort masks, sampling positions, ranks, scaling and role-specific seeded jitter. An eight-column operation-count test requires 8 rankings instead of 56. Pair-specific joint neighborhoods remain necessary; sorted exact marginal counts replace two marginal trees per pair.
- Spearman bootstrap counts both columns together, reuses count buffers as rank tables, and reduces covariance directly with the exact rank mean. Scratch is allocated outside draw loops, with eight fixed lanes.
- Discovery shares null counts for identical validity patterns, operates in 16-column Float64 FFT tiles, and stores exact half-integer ranks in Float32 only through the proven 2**23-row bound.
- Discovery AUC bootstrap preserves the original circular block starts. Difference counts and sparse positive rank groups remove sampled-rank matrices and repeated dense group products. Larger/nonbinary inputs retain the reference path.
- Persisted immutable contexts share ranks and analysis across related queries; source advances require hashing actual context values. Current-snapshot rules, plots and manifests are rebuilt when their own inputs change. Mutable pools never enter reuse.

The [behavior contract](factorization-contract.md) defines zero numerical tolerance, unchanged sampling/seeds, errors, missingness, metadata and ordering. The [dependency map](engine.mmd) shows where bounded caches and research coordination sit.

## Remaining costs and limits

Full catalog construction remains dominated by factory computation and materialization. Its three-pair CPU cost is essentially unchanged; the wall-time result is not a demonstrated improvement. A new source version still needs a current feature pool before historical context analysis can be reused.

Pair-specific joint MI neighborhoods, distinct-label FFT nulls, changed-context ranks, current-snapshot rule fitting, requested output bytes and process/IPC setup remain. Existing sample caps are unchanged. Byte-bounded caches intentionally trade some recomputation for bounded retained memory.

This was a shared M1 Max host with 10 CPUs and 64 GiB; load is recorded in every file and varied substantially. Censored reference trials (QC, related discovery and cold catalog service) are retained separately. Completed pairs cannot characterize the censored tail. The larger cold catalog service did not finish inside the required 49-second worker limit; no projected end-to-end sum is substituted.

Each trial starts a fresh process and private application caches; existing Numba disk caches are present. Imports, snapshot reads, fixtures, compiled-bootstrap setup, cold work, warm work and outputs are separate fields. These results do not characterize first-install JIT compilation or production p95 latency. Research children use the existing four-thread research policy, including its BLAS settings.

The local source snapshots span January 2024 through May 15, 2026. They cover the complete recorded source inputs, not a claim about the current live bar store. No approximation or utility loss was authorized or introduced.
