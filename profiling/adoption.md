# Exact kernel adoption

The released Limen kernels address the full-catalog bottleneck left after Mill factorization. This experiment changes only the dependency: both sides use the same optimized Mill source and the same frozen 226-factory API. The earlier [Mill-only experiment](factorization.md) remains a separate, fixed-Limen comparison; its speedups are not compounded here.

Measured Mill commit: `12cfe767ee299127e5e564a7b134655ddc046f41`. Limen control `86968e4` (3.30.0), adoption `728a6ee` (5.29.2). The pinned reference is verified before and after comparison. The packaged contract retains the old names, parameter schemas and defaults; new upstream factories are not exported. Loaded catalog state participates in persisted pool identities. The optimized upstream release admits Python <3.14; Mill selects it on 3.12–3.13 and retains the existing immutable dependency pin on 3.14+ to preserve interpreter admission. No adoption speedup is measured or claimed on 3.14+.

## Behavior gate

All 2,486 factory/case comparisons and 77 target/case comparisons are exact: 226 factories across empty, one-row, short, constant/tied, missing/NaN, infinite, nondefault-parameter, Float32, integer, near-full-window and overlong-window inputs. All seven existing target bindings also match. Schemas, null masks, finite value bits, signed zero, string values and errors match; NaN payload/sign is canonicalized. No numerical tolerance or approximation is introduced. These finite cases are evidence, not a proof over every possible input. All matched full-history workloads also compare output/metadata/report/JSON digests. The intentional output change is the exported rule metadata field `candidates/*/rule/manifest/metadata/limen_version`: `3.30.0` → `5.29.2`. Independent output witnesses reproduce every original before/after service digest by changing exactly that string field; every other value, decision, schema and output field remains exact. The catalog witness reuses a prepared pool identified by source values, dependency, code and loaded catalog state, then reconstructs the recorded cold `pool.cached=false` metadata; its digests must match all original cold-response hashes. Witness setup/timings are not included in performance comparisons. Replayed catalog verification uses one research thread; original timed services use four. Elapsed-time fields are excluded as in the prior experiment. [Functional evidence](results/adoption/functional.json) and [verified timing evidence](adoption.json).

## Repeated matched timings

Milliseconds; median of three fresh process observations. Brackets give the min–max across those observations. Within-process repetitions remain in the raw data. Alternating order: before/after, after/before, before/after.

| Complete measured boundary | Before ms [range] | After ms [range] | Median ratio |
| --- | ---: | ---: | ---: |
| Eight views, cold, 83k | 845.59 [549.64–1,317.15] | 584.85 [550.43–672.40] | 1.45× |
| Eight views, steady, 83k | 0.15 [0.15–0.16] | 0.17 [0.16–0.77] | 0.90× |
| Full catalog preparation, 83k | 35,287.52 [34,529.25–43,786.78] | 20,670.47 [17,085.19–21,367.09] | 1.71× |
| Full catalog mapped hit, 83k | 25.85 [20.64–27.88] | 20.25 [8.41–24.71] | 1.28× |
| Full catalog discovery service, cold, 20k | 20,893.49 [19,053.29–30,372.02] | 22,995.11 [15,903.43–23,362.94] | 0.91× |
| 30-factory discovery service, cold, 83k | 15,501.38 [9,310.74–22,140.22] | 19,224.55 [10,311.67–20,296.63] | 0.81× |
| 30-factory discovery service, repeated, 83k | 55.94 [55.71–80.12] | 97.18 [94.26–165.42] | 0.58× |
| HTTP data, first, 83k | 1,872.07 [1,448.24–2,097.20] | 460.72 [432.72–477.32] | 4.06× |
| HTTP data, four cold callers, 83k | 387.55 [152.01–426.41] | 162.24 [112.92–164.80] | 2.39× |
| QC, 31 columns / 465 pairs, 83k | 10,421.05 [8,187.53–24,613.68] | 15,681.20 [11,648.59–18,010.20] | 0.66× |
| QC report from precomputed state, 83k | 0.49 [0.45–11.55] | 0.37 [0.36–0.90] | 1.32× |

Full-catalog preparation process CPU: 19,309.67 → 11,904.16 ms; all three per-side CPU samples and ranges are retained in the JSON. Setup/imports, source decoding and fixture preparation are outside the named request boundaries and are reported below. The view workload includes feature/target production, joins and JSON; HTTP includes sockets and payload reads. Catalog discovery service includes cold worker creation, preparation retries, full pool generation, discovery, IPC and JSON; it excludes HTTP/auth. QC setup/JIT is measured separately; cold QC retains no content/kernel-result cache. Existing on-disk compiled artifacts and OS file caches are retained; this is not controlled first-install compilation. A mapped catalog hit reuses a previously completed catalog.

Catalog preparation and first HTTP delivery improve in this experiment. Service-wide latency is not established as faster: QC and discovery-service wall medians are higher, and ranges overlap under substantial contention. These remain latency concerns; the reported ratios are not a blanket engine speedup.

## Setup and peak memory

| Workload | Import ms before → after | Source read ms before → after | Fixture ms before → after | Worker peak MiB before → after | Research child peak MiB before → after |
| --- | ---: | ---: | ---: | ---: | ---: |
| workload-83k | 3,975.17 → 2,744.10 | 47.23 → 21.26 | 8.70 → 4.19 | 825.75 → 823.34 | — |
| pool-83k | 3,931.96 → 3,577.77 | 83.40 → 26.65 | 20.52 → 5.03 | 458.06 → 462.31 | — |
| catalog-service-20k | 5,677.30 → 6,001.85 | 20.68 → 18.34 | 6.11 → 11.15 | 279.52 → 255.88 | 432.58 → 419.95 |
| research-discover-83k | 7,792.77 → 7,737.70 | 99.07 → 66.66 | 47.61 → 26.33 | 296.69 → 285.59 | 406.73 → 411.55 |
| http-83k | 9,332.13 → 9,064.32 | 34.06 → 38.23 | 19.11 → 9.07 | 452.88 → 451.14 | — |
| qc-83k | 4,946.16 → 7,860.77 | 29.92 → 35.09 | 9.08 → 8.50 | 534.75 → 515.75 | — |

Compiled-bootstrap setup/load: 183.74 → 373.32 ms. Total eight-view encoded output: 55.77 → 55.77 MiB. Exact output parity includes decoded JSON and the underlying feature frames.

Observed one-minute host load: 30.3–142.6 on 10 logical CPUs. 7 excluded trials are retained with their actual status (deadline, error, or unpaired observation); none enters completed-pair medians. Completed medians are conditional on finishing within these limits. High contention makes small wall-time changes inconclusive. No production latency or throughput guarantee is inferred. Peak RSS includes imports, source decoding, fixtures, retained caches and transient work; parent and child peaks are separate measurements, not a simultaneous total.

| Excluded trial | Recorded status | Error or incomplete phase |
| --- | --- | --- |
| http-after-83k-2-request-timeout-1.json | error | TimeoutError: timed out |
| pool-before-83k-1-timeout-1.json | timeout_49s | service_catalog_prepare |
| pool-before-83k-3-timeout-1.json | timeout_49s | service_catalog_prepare |
| pool-before-83k-3-timeout-2.json | timeout_49s | service_catalog_prepare |
| workload-after-83k-1-import-error.json | error | ModuleNotFoundError: No module named 'typing_extensions' |
| workload-before-83k-1-timeout-1.json | timeout_49s | all_views_cold_repeated |
| workload-before-83k-1-unpaired.json | complete | unpaired completed observation |

The HTTP client has a 15-second response timeout; the benchmark worker has a 49-second overall deadline. Deadline/error trials are failures under the recorded contention, not zero-cost observations.

The isolated target initially used `--no-deps`; its import-error trial preceded installation of the declared `typing_extensions` dependency. Completed matched pairs use the completed environment. Parent Polars/Numba use four threads and BLAS/OMP one; production research children cap all four libraries at four threads on both sides.

## Attribution and remaining work

The dependency switch is the ablation in this experiment: identical Mill code, catalog contract, fixtures and numerical runtime. The [prior factorization report](factorization.md) separately contains 33 matched ablation pairs for eleven Mill components. QC pair enumeration, exact joint-neighborhood searches, bootstrap draws and discovery null/rule work remain necessary computations. Cache hits still require full relevant source/value identity; new histories still require feature generation and current-snapshot rules. The prior 107 ms synthetic appended-pool search excludes rebuilding the pool and is not a production new-bar latency. Opaque factories provide no stable-prefix or resumable-state contract, so a catalog for a new source snapshot is rebuilt. Serialization and output transfer remain proportional to payload size. Bounded cache retention trades memory against recomputation; it does not bound whole-process RSS. Reducing these remaining semantics, sampling history or changing numeric accuracy requires a separately agreed utility contract.

The following post-adoption eager-factory profile uses the complete 20,784-row source, ordered by process CPU. It is one diagnostic observation, not a speedup experiment; eager calls can repeat work shared by the catalog planner. No extrapolation to 83k rows is made.

| Eager factory diagnostic | Wall ms | Process CPU ms |
| --- | ---: | ---: |
| cdlmathold | 441.34 | 98.10 |
| ht_trendmode | 172.98 | 95.85 |
| cdl3starsinsouth | 215.58 | 92.22 |
| cdlconcealbabyswall | 1,167.40 | 87.27 |
| cdlidentical3crows | 224.58 | 87.14 |
| cdlrisefall3methods | 127.72 | 86.72 |
| ht_sine | 103.10 | 66.32 |
| ultosc | 358.39 | 64.84 |
| ht_dcphase | 71.70 | 62.79 |
| cdl3blackcrows | 180.95 | 60.17 |

Reproduction: the unchanged `engine_sweep.py` and `catalog_service.py` drivers run through the full source with `GetData.from_parquet`; see each raw record for versions, thread environment, source hash, row count, setup, CPU samples, output hashes and memory. The two complete local sources contain 20,784 hourly bars and 83,136 fifteen-minute bars. No date/row truncation or simulated speedup is used.

The [instrumented full-20k pool diagnostic](results/adoption/pool-io-diagnostic-after-20k-1.json) measured 127 disk-retention scans: 186.72 ms CPU (5.01% of 3729.02 ms preparation CPU), 263.10 ms wall. The scan sees changing cache contents at each checkpoint; reducing this cost requires concurrency-safe size accounting. This is a measured lower-priority concern, not an 83k projection. The same diagnostic also matches all 259 pool value/metadata digests from the prior 20k reference. Its dependency imports precede instrumentation, so its import time is not a cold-import measurement. The [executed instrumentation wrapper](results/adoption/pool-io-driver.py.txt) is retained.
