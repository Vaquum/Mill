# Mill engine performance

**Reuse and bounded research reduce cold work.** [Original audit](audit.json), [current engine map](engine.mmd), [comparison evidence](optimization.json), [raw measurements](results/optimization).

| Measured path | Before | After |
| --- | ---: | ---: |
| Add one feature, 83,136 bars | 895 ms | 387 ms |
| JSON encode, 83,136 bars | 307 ms | 173 ms |
| GetData disk hit, 83,136 bars | 19.22 ms | 0.64 ms |
| Closed-history refresh, unchanged source values | 49.81 ms | 16.59 ms |
| Cold QC, 8 / 30 factories, 20,784 bars | 5.11 / 19.07 s | 2.40 / 14.51 s |
| Discovery, prebuilt 259-column pool, 83,136 bars | 13.39 s | 9.69 s |
| Discovery process peak RSS, same input pool | 2.88 GiB | 1.03 GiB |
| Data p95 during QC, latest short 20k run | 19.85 ms | 22.71 ms |

Feature, JSON, QC, pool and discovery digests match. Mixed HTTP QC also matches. Local M1 Max/Python 3.13 snapshots; Polars/Numba use four threads and BLAS one. Shared-host load varied substantially; these are measured samples, not production latency bounds. Cold operations and 83k search have one sample; repeated timings use medians. RSS covers the complete worker, imports, input, fixtures and transient work. Catalog construction remains dominated by [upstream indicators](https://github.com/Vaquum/Limen/issues/930); exact pairwise QC, ranking/null work and JSON boxing remain costs. No stable mixed-latency improvement is established.

RAM budgets use `MILL_{RAW,SERVED_RAW,PRODUCERS,RESULTS,QC,KERNELS,POOLS}_MIB` (defaults 64/64/256/128/16/32/256). They bound accounted cache data, not total RSS; zero disables a RAM cache. HTTP research uses memory-mapped pools and a 2,048 MiB disk LRU (`MILL_POOL_DISK_MIB`). Catalogs checkpoint between roughly five-second steps and survive worker restarts. Preparation/overload/deadlines return 503 with Retry-After. One research job runs, one queues; `MILL_RESEARCH_THREADS` accepts 1–8 (default four), execution/queue deadline is 45 s; HTTP admits 16 requests, at most two research requests, with a 10 s socket inactivity timeout. Disk eviction can evict catalogs; increase its budget to retain more histories. Complete history and statistical decisions are preserved.

Activate venv `mill`. Run `python profiling/engine_sweep.py --case features --parquet /absolute/source.parquet --output /tmp/features.json --k 8 --reps 3`; cases include data, qc, discover, pool, http and memory. Each invocation has a 49 s worker and 50 s supervisor limit. For large discovery, use `--discover-phase pool --pool-artifact /tmp/pool`, then `--discover-phase search` with the same artifact and inputs. `--portable-pool` explicitly compares different engine kernels on one recorded immutable pool. Failed trials remain under `results/optimization/trials`. The original audit and samples remain historical evidence.
