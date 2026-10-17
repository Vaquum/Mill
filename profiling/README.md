# Mill engine performance

**Cache misses dominate.** Built-in views compute quickly; JSON, repeated history work, discovery allocations and statistical QC prevent consistently low latency.

| Path | 20,784 bars | 83,136 bars |
| --- | ---: | ---: |
| Built-in view compute / JSON | 0.10–13.71 / 22.28–67.67 ms | Not measured |
| Eight indicators compute / JSON | 100.05 / 57.94 ms | 663.13 / 253.63 ms |
| Exact result cache hit, in process | 19 µs | 18 µs |
| Catalog build, 259 columns | 4.48 s | 18.60 s |
| Discovery, prebuilt catalog | 2.09 s | 12.41 s |
| Discovery peak process RSS | 1.12 GiB | 3.29 GiB |
| Cold QC, 1 / 8 / 30 indicators | 1.32 / 3.15 / 19.14 s | Not measured |

[Engine map](engine.mmd), [16 findings and optimization sequence](audit.json), [raw samples](results/2026-10-17), [import attribution](import_costs.json). Measurements are local snapshots on M1 Max, Python 3.13, pinned Polars/Limen; production uses Linux/Python 3.12. Medians describe repeated timings; cold measurements are single samples. RSS includes imports, full input decoding, fixtures and transient work; it is neither incremental allocation nor steady-state memory. No production code changed. Prioritize lazy JSON, factory-level reuse, bounded discovery memory/workers, and a live Arrow path that avoids Parquet round-trips. Preserve complete history, closed-bar freshness, feature values, statistical decisions and existing API responses. Proposed acceptance gates are in the audit; speedups are not yet demonstrated.

Activate venv `mill`; run `python profiling/engine_sweep.py --case features --parquet /absolute/source.parquet --output /tmp/features.json --k 8 --reps 5`. Cases: startup, data, features, factories, views, qc, discover, http, memory. Add `--bar 15m_time_bars` for 15m input; `--catalog` for full discovery. Each invocation has a 49-second worker alarm and 50-second supervisor limit; fixtures are temporary. `--rows` limits compute after complete source decoding and is rejected for memory measurements. Run cases sequentially. Ordinary requests do not have these profiler deadlines.

Existing QC tools remain available: `qc_profile.py` attributes stages; `qc_validate.py --dump /tmp/before.json` then `--check /tmp/before.json` checks exact QC outputs across revisions; `qc_memo_demo.py` measures reuse; `qc_sweep_profile.py` measures subset aggregation. The CLI safety boundary and recorded-sample consistency run in pytest; performance timings stay manual.
