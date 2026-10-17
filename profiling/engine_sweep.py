"""Bounded, isolated engine measurements. Run one case per process; see README."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import statistics
import subprocess
import sys
import tempfile
import time


CASES = ("startup", "data", "features", "factories", "views", "qc", "discover", "pool", "http", "memory", "workload", "research-discover")
FACTORIES = ("rsi", "mom", "roc", "rocp", "cci", "cmo", "willr", "atr",
             "natr", "trix", "apo", "ppo", "bop", "mfi", "kama", "tema",
             "dema", "wma", "sma", "ema", "midpoint", "linearreg", "tsf",
             "ad", "obv", "trange", "avgprice", "medprice", "typprice", "wclprice")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=0, help="0 means all source rows")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--bar", choices=("1h_time_bars", "15m_time_bars"), default="1h_time_bars")
    parser.add_argument("--catalog", action="store_true", help="use all eligible factories for discovery")
    parser.add_argument("--discover-phase", choices=("combined", "pool", "search"), default="combined")
    parser.add_argument("--pool-artifact", type=Path, help="private directory for a split discovery measurement")
    parser.add_argument("--portable-pool", action="store_true", help="compare kernels on one pool built by another recorded engine revision")
    parser.add_argument("--ablate", choices=("none", "native", "pairs", "edges", "null", "bootstrap", "reuse", "rules", "json", "context"), default="none")
    parser.add_argument("--reference-source", type=Path, help="frozen reference checkout for the null ablation")
    parser.add_argument("--related-searches", action="store_true", help="also measure repeated, display-only and changed-window searches")
    parser.add_argument("--transition-searches", action="store_true", help="measure an explicitly synthetic appended pool row outside the original context")
    parser.add_argument("--warm-kernels", action="store_true", help="measure compiled-kernel setup separately before cold QC")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--scratch", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rows < 0 or not 1 <= args.k <= len(FACTORIES) or not 1 <= args.reps <= 20:
        parser.error("rows >= 0, 1 <= k <= 30, and 1 <= reps <= 20 required")
    if not args.parquet.is_file():
        parser.error("parquet source must exist")
    if args.output.resolve() == args.parquet.resolve():
        parser.error("output cannot overwrite the input parquet")
    if args.case == "memory" and args.rows:
        parser.error("memory cases require full input; --rows limits compute, not source decoding")
    if args.discover_phase != "combined" and (args.case != "discover" or args.pool_artifact is None):
        parser.error("split discovery requires --case discover and --pool-artifact")
    if args.ablate in ("null", "bootstrap") and not args.reference_source:
        parser.error("null/bootstrap ablation requires --reference-source")
    if args.ablate == "bootstrap" and not args.warm_kernels:
        parser.error("bootstrap ablation requires --warm-kernels to compare execution separately from compilation")
    allowed = {"native": ("workload", "features", "views", "discover", "pool"),
               "pairs": ("qc",), "edges": ("qc",), "bootstrap": ("qc",),
               "null": ("discover",), "reuse": ("discover",), "rules": ("discover",),
               "json": ("workload", "features", "http"), "context": ("discover",)}
    if args.ablate != "none" and args.case not in allowed[args.ablate]:
        parser.error("ablation does not apply to this workload")
    if args.ablate == "native" and args.case == "discover" and args.discover_phase == "search":
        parser.error("native ablation requires building a pool")
    if args.related_searches and args.transition_searches:
        parser.error("related and transition searches use separate bounded invocations")
    return args


def peak_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def write_result(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".mill-sweep-", suffix=".json", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(doc, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class Recorder:
    def __init__(self, args):
        self.args = args
        self.doc = {"schema": 1, "case": args.case, "status": "running", "worker_pid": os.getpid(),
                    "peak_rss_scope": "whole worker, including imports, complete source decoding and fixture setup", "measurements": []}
        self.save()

    def save(self):
        self.doc["peak_rss_bytes"] = peak_bytes()
        write_result(self.args.output, self.doc)

    def measure(self, name, fn, reps=None, **metadata):
        self.doc["active_measurement"] = name
        self.save()
        samples, cpu_samples, result = [], [], None
        for _ in range(self.args.reps if reps is None else reps):
            began = time.perf_counter_ns()
            cpu_began = time.process_time_ns()
            result = fn()
            samples.append((time.perf_counter_ns() - began) / 1e6)
            cpu_samples.append((time.process_time_ns() - cpu_began) / 1e6)
        self.doc["measurements"].append({"name": name, "samples_ms": samples,
                                         "median_ms": statistics.median(samples),
                                         "process_cpu_samples_ms": cpu_samples, **metadata})
        self.doc.pop("active_measurement", None)
        self.save()
        return result


def environment():
    from importlib.metadata import distribution, version
    import numba
    import polars as pl
    try:
        import mill
        checkout = Path(mill.__file__).resolve().parents[2]
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True, timeout=5,
                                      stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        sha = None
    return {"python": sys.version, "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "host_load_1m": os.getloadavg()[0],
            "polars_threads": pl.thread_pool_size(),
            "numba_threads": numba.get_num_threads(), "numba_layer": numba.threading_layer(),
            "versions": {name: version(name) for name in
                         ("polars", "numpy", "numba", "scipy", "scikit-learn", "statsmodels", "vaquum_limen")},
            "limen_origin": json.loads(distribution("vaquum_limen").read_text("direct_url.json") or "null"),
            "thread_environment": {k: os.environ.get(k) for k in
                                   ("POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
            "engine_sha": sha,
            "source_tree_sha256": content_digest({
                str(p.relative_to(Path(mill.__file__).parent)): source_digest(p)
                for p in sorted(Path(mill.__file__).parent.rglob("*.py"))})}


def source_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def matrix_digest(values):
    import numpy as np
    digest = hashlib.sha256(str(values.shape).encode())
    for column in values.T:
        digest.update(np.ascontiguousarray(column).tobytes())
    return digest.hexdigest()


def frame_digest(frame):
    digest = hashlib.sha256(content_digest(dict(frame.schema)).encode())
    for column in frame:
        digest.update(column.is_null().to_numpy().tobytes())
        values = column.to_physical()
        if values.null_count():
            values = values.fill_null(False if values.dtype == bool else 0)
        digest.update(values.to_numpy().tobytes())
    return digest.hexdigest()


def worker(args):
    record = Recorder(args)
    record.doc["profiler_sha256"] = source_digest(Path(__file__))
    # Set before Mill imports; all cache/store/config mutations stay in this directory.
    scratch = nullcontext(str(args.scratch)) if args.scratch else tempfile.TemporaryDirectory(prefix="mill_engine_sweep_")
    with scratch as directory:
        os.environ["MILL_RESEARCH_TMPDIR"] = directory
        for key, name in (("MILL_ARROW_DIR", "arrow"), ("MILL_CACHE_DIR", "cache"), ("MILL_CONFIG_DIR", "config")):
            os.environ[key] = str(Path(directory) / name)
        began = time.perf_counter_ns()
        if args.case == "discover":
            from mill import discover, feature_registry
            serve = qc = targets = None
            record.doc["import_scope"] = "discovery dependencies only; no frontend or QC warmup"
        else:
            from mill import serve
            record.doc["serve_import_ms"] = (time.perf_counter_ns() - began) / 1e6
            from mill import qc, discover, targets, feature_registry
        import numpy as np
        import polars as pl
        from mill.get_data import GetData
        from mill.table import build_from_spec
        from mill.settings import arrow_latest_path
        from mill.bars import arrow_series, time_bar_seconds
        record.doc["ablation"] = args.ablate
        if args.ablate == "native":
            from mill import expressions, table
            def eager(raw, requests, *, errors=False, publish=None):
                results = []
                for producer, params in requests:
                    try:
                        results.append(producer(raw, **params))
                    except Exception as exc:
                        if not errors:
                            raise
                        results.append(exc)
                    if publish is not None:
                        publish({len(results)-1: results[-1]})
                return results
            expressions.evaluate = table.evaluate = discover.evaluate = eager
        elif args.ablate == "rules":
            discover.discovery_reuse.rule = lambda key, build: build()
        elif args.ablate == "json":
            serve.encode_rows = lambda frame: serve.orjson.dumps({"columns": frame.columns, "rows": frame.rows()})
        elif args.ablate == "context":
            discover.discovery_reuse.context_key = lambda source_key, row_bits, pool, index: (source_key, row_bits)
        elif args.ablate == "pairs":
            def original_pairs(arrays, cols):
                return {(left, right): qc._pair_metrics(arrays[left], arrays[right])
                        for i, left in enumerate(cols) for right in cols[i+1:]}
            qc._pair_map = original_pairs
        elif args.ablate == "edges":
            def original_edges(arrays, cols, target, blocks, label):
                return {col: qc._edge_metrics(arrays[col], target, qc._joint_block(blocks, col, label),
                                              qc._seed_for(col)) for col in cols}
            qc._edge_map = original_edges
        elif args.ablate in ("null", "bootstrap"):
            import ast
            module, name = (discover, "shift_null") if args.ablate == "null" else (qc, "_bootstrap_spearman_estimates")
            source = args.reference_source / ("src/mill/discover.py" if args.ablate == "null" else "src/mill/qc.py")
            tree = ast.parse(source.read_text())
            node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            if args.ablate == "bootstrap":
                # Compile the archived exact kernel without changing either
                # checkout's disk JIT cache; setup is measured explicitly.
                for decorator in node.decorator_list:
                    for keyword in getattr(decorator, "keywords", []):
                        if keyword.arg == "cache":
                            keyword.value = ast.Constant(False)
                ast.fix_missing_locations(node)
            namespace = dict(module.__dict__)
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
            setattr(module, name, namespace[name])
            record.doc["ablation_reference_sha256"] = source_digest(source)
        record.doc["import_ms"] = (time.perf_counter_ns() - began) / 1e6
        record.doc["environment"] = environment()
        if args.case == "startup":
            record.doc["peak_rss_scope"] = "imports only; source parquet is not opened"
            record.doc["status"] = "complete"
            record.save()
            return
        record.doc["source"] = {"file": args.parquet.name, "sha256": source_digest(args.parquet)}
        raw = record.measure("source_read", lambda: GetData().from_parquet(args.parquet), reps=1)
        source_rows = raw.height
        if args.rows:
            raw = raw.tail(args.rows)
        record.doc["input"] = {"source_rows": source_rows, "rows": raw.height,
                               "columns": raw.width, "frame_bytes": raw.estimated_size(),
                               "start": str(raw["datetime"][0]), "end": str(raw["datetime"][-1]),
                               "bar": args.bar, "k": args.k, "reps": args.reps}
        fixture_began = time.perf_counter_ns()
        bar = args.bar
        series = arrow_series(bar)
        path = arrow_latest_path(series)
        path.parent.mkdir(parents=True)
        raw.with_columns(pl.col("datetime").dt.epoch("ns").alias("ts")).drop("datetime").write_ipc(
            path, compression="uncompressed", record_batch_size=max(1, raw.height))
        # Make every fixture bar closed, including sources with future timestamps.
        stamp = max(time.time_ns(), int(raw["datetime"].dt.epoch("ns")[-1]) + time_bar_seconds(bar) * 1_000_000_000)
        os.utime(path, ns=(stamp, stamp))
        record.doc["fixture_prepare_ms"] = (time.perf_counter_ns() - fixture_began) / 1e6
        spec = [{"factory": name, "params": {}} for name in FACTORIES[:args.k]]
        measure = record.measure
        if args.case == "workload":
            from mill import get_data, table
            from mill.views import ALL
            def batch(cold=False):
                if cold:
                    for cache in (serve._RAW_CACHE, serve._RESULTS, table._PRODUCERS, get_data._DATA):
                        cache.clear()
                    if hasattr(targets, "_RESULTS"):
                        targets._RESULTS.clear()
                return [(view.NAME, *serve._ensure(bar, getattr(view, "TARGET", "next_return"), list(view.SPEC)))
                        for view in ALL]
            result = measure("all_views_cold_first", lambda: batch(True), reps=1)
            measure("all_views_cold_repeated", lambda: batch(True))
            measure("all_views_steady", batch)
            record.doc["parity_workload"] = {name: {"frame": frame_digest(frame),
                "payload": content_digest(json.loads(payload))} for name, payload, frame in result}
            record.doc["payload_digest_scope"] = "canonical parsed JSON; numeric values and types are preserved"
            record.doc["output_bytes"] = sum(len(payload) for _, payload, _ in result)
            record.doc["workload_views"] = [view.NAME for view in ALL]
        elif args.case == "data":
            loader = GetData()
            measure("arrow_read_normalize_select", lambda: loader._load_arrow(series, None, None))
            data = measure("get_data_cold_cache", lambda: loader.get_time_bars(bar), reps=1)
            record.doc["parity_data"] = frame_digest(data)
            measure("get_data_memory_hit", lambda: loader.get_time_bars(bar))
            def disk_hit():
                from mill import get_data
                if hasattr(get_data, "_DATA"):
                    get_data._DATA.clear()
                return loader.get_time_bars(bar)
            measure("get_data_disk_cache_hit", disk_hit)
            measure("get_data_tail_100_cold", lambda: loader.get_time_bars(bar, n_rows=100), reps=1)
            measure("raw_for_first", lambda: serve._raw_for(bar), reps=1)
            measure("raw_for_memory_hit", lambda: serve._raw_for(bar))
            def rewrite():
                current = path.stat().st_mtime_ns + 1_000_000
                os.utime(path, ns=(current, current))
                return serve._raw_for(bar)
            measure("raw_for_unchanged_rewrite", rewrite)
            first_version = serve._raw_for(bar)[1]
            changed = raw.with_columns(pl.when(pl.int_range(pl.len()) == raw.height - 1)
                                      .then(pl.col("close") + 1).otherwise(pl.col("close")).alias("close"))
            changed.with_columns(pl.col("datetime").dt.epoch("ns").alias("ts")).drop("datetime").write_ipc(path)
            os.utime(path, ns=(stamp + 1_000_000_000, stamp + 1_000_000_000))
            measure("raw_for_changed_closed_bar", lambda: serve._raw_for(bar), reps=1)
            record.doc["version_changed"] = serve._raw_for(bar)[1] != first_version
            from concurrent.futures import ThreadPoolExecutor
            from unittest.mock import patch
            serve._RAW_CACHE.clear()
            with patch.object(serve, "klines", wraps=serve.klines) as load:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    measure("raw_for_concurrent_first_c4", lambda: list(executor.map(serve._raw_for, [bar] * 4)), reps=1)
                record.doc["concurrent_raw_load_calls"] = load.call_count
        elif args.case in {"features", "memory"}:
            frame = measure("build_features", lambda: build_from_spec(raw, spec))
            target = measure("target_next_return", lambda: targets.apply(raw, "next_return").select("datetime", "next_return"))
            joined = measure("join_target", lambda: frame.join(target, on="datetime", how="left"))
            payload = measure("serialize_json", lambda: serve._serialize(joined))
            record.doc["parity_frame"] = frame_digest(joined)
            record.doc["parity_payload"] = content_digest(json.loads(payload))
            record.doc["result"] = {"columns": joined.width, "frame_bytes": joined.estimated_size(), "payload_bytes": len(payload)}
            measure("ensure_cold", lambda: serve._ensure(bar, "next_return", spec), reps=1)
            measure("ensure_memory_hit", lambda: serve._ensure(bar, "next_return", spec))
            measure("ensure_plus_one_feature", lambda: serve._ensure(bar, "next_return", spec + [{"factory": "close", "params": {}}]), reps=1)
            if args.case == "memory":
                import tracemalloc
                tracemalloc.start()
                serve._serialize(joined)
                record.doc["serialize_python_peak_bytes"] = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
                for size in range(1, args.k + 1):
                    serve._ensure(bar, "none", spec[:size])
                record.doc["retained_results"] = {"entries": len(serve._RESULTS),
                    "payload_bytes": sum(len(p) for p, _ in serve._RESULTS.values()),
                    "frame_bytes_sum_not_unique": sum(f.estimated_size() for _, f in serve._RESULTS.values())}
        elif args.case == "factories":
            record.doc["registry_count"] = len(feature_registry.REGISTRY)
            record.doc["skipped"] = []
            for name, entry in feature_registry.REGISTRY.items():
                if any(p["default"] is None for p in entry["params"]):
                    record.doc["skipped"].append({"factory": name, "reason": "requires parameters"})
                    continue
                try:
                    part = measure("factory:" + name, lambda e=entry: e["producer"](raw), reps=1)
                    record.doc["measurements"][-1].update(columns=part.width, frame_bytes=part.estimated_size())
                except Exception as exc:
                    record.doc["skipped"].append({"factory": name, "reason": type(exc).__name__ + ": " + str(exc)})
                record.save()
            pool = measure("catalog_pool", lambda: discover.catalog_pool(raw), reps=1)
            record.doc["pool"] = {"columns": len(pool.names), "bytes": pool.values.nbytes,
                                   "skipped": list(pool.skipped)}
        elif args.case == "views":
            for name in targets.NAMES:
                measure("target:" + name, lambda n=name: targets.apply(raw, n), reps=1)
            for view in serve.VIEWS:
                frame = measure("view:" + view.NAME, lambda v=view: build_from_spec(raw, v.SPEC))
                payload = measure("view_json:" + view.NAME, lambda: serve._serialize(frame))
                record.doc["measurements"][-1].update(columns=frame.width, payload_bytes=len(payload))
        elif args.case == "qc":
            from unittest.mock import patch
            frame = build_from_spec(raw, spec).join(targets.apply(raw, "next_return").select("datetime", "next_return"), on="datetime", how="left")
            sections = {}
            def timed(name, fn):
                def run(*a, **kw):
                    began = time.perf_counter_ns()
                    try:
                        return fn(*a, **kw)
                    finally:
                        sections[name] = sections.get(name, 0) + (time.perf_counter_ns() - began) / 1e6
                return run
            from contextlib import ExitStack
            if args.warm_kernels:
                measure("compiled_kernel_setup", qc._warmup_bootstrap, reps=1)
            qc.clear_kernel_cache()
            record.doc["bootstrap_signatures_before"] = [str(s) for s in qc._bootstrap_spearman_estimates.signatures]
            with ExitStack() as stack:
                for name in ("_health_map", "_pair_map", "_edge_map", "_regime_map", "_block_size"):
                    stack.enter_context(patch.object(qc, name, timed(name, getattr(qc, name))))
                prim = measure("qc_precompute_cold", lambda: qc.precompute(frame, "next_return"), reps=1)
            record.doc["qc_cold_sections_ms"] = sections
            record.doc["bootstrap_signatures_after"] = [str(s) for s in qc._bootstrap_spearman_estimates.signatures]
            if args.k <= 8:
                qc.clear_kernel_cache()
                measure("qc_precompute_second_cold", lambda: qc.precompute(frame, "next_return"), reps=1)
            measure("qc_report_warm_kernel_cache", lambda: qc.report(frame, "next_return"))
            measure("qc_report_for_precomputed", lambda: qc.report_for(prim))
            measure("qc_sweep_sizes_1_2_3", lambda: sum(1 for _ in qc.sweep(prim, (1, 2, 3))))
            report = qc.report_for(prim)
            record.doc["parity_report"] = content_digest(report)
            measure("grade_map", lambda: serve.grading.grade_map(report, n_rows=frame.height, target="next_return"))
            measure("report_json", lambda: json.dumps(report, allow_nan=False).encode())
            record.doc["kernel_cache_entries"] = len(qc._KERNEL_CACHE)
            record.doc["pair_count"] = len(prim.pairs)
            record.doc["feature_columns"] = len(prim.feature_cols)
        elif args.case == "pool":
            from mill import pool_store
            digest = serve._frame_digest(raw)
            partial = []
            def prepare():
                while True:
                    began = time.perf_counter_ns()
                    try:
                        return pool_store.pool_for(raw, bar, None, digest)[0]
                    except pool_store.PoolPending:
                        partial.append((time.perf_counter_ns() - began) / 1e6)
            pool = measure("service_catalog_prepare", prepare, reps=1)
            measure("service_catalog_mmap_hit", lambda: pool_store.pool_for(raw, bar, None, digest))
            record.doc["partial_preparation_steps_ms"] = partial
            record.doc["parity_pool"] = content_digest({"names": pool.names, "values": matrix_digest(pool.values)})
            record.doc["parity_pool_metadata"] = content_digest({name: getattr(pool, name) for name in
                ("names", "natives", "factories", "sources", "params", "skipped")})
            record.doc["mapped_values"] = isinstance(pool.values, np.memmap)
        elif args.case == "discover":
            identity = {"source": record.doc["source"], "input": record.doc["input"],
                        "tree": record.doc["environment"]["source_tree_sha256"], "catalog": args.catalog}
            if args.discover_phase == "search":
                metadata = json.loads((args.pool_artifact / "metadata.json").read_text())
                built = metadata["identity"]
                match = (built == identity or (args.portable_pool and
                    {k: v for k, v in built.items() if k != "tree"} ==
                    {k: v for k, v in identity.items() if k != "tree"}))
                if not match:
                    raise ValueError("pool artifact does not match this source, parameters and engine")
                record.doc["pool_builder_source_tree_sha256"] = built["tree"]
                fields = metadata["fields"]
                if hasattr(discover, "mapped_pool"):
                    pool = discover.mapped_pool(args.pool_artifact, {k: tuple(v) for k, v in fields.items()})
                else:
                    pool = discover.Pool(**{k: tuple(v) for k, v in fields.items()},
                        ms=np.load(args.pool_artifact / "ms.npy", allow_pickle=False, mmap_mode="r"),
                        values=np.load(args.pool_artifact / "values.npy", allow_pickle=False, mmap_mode="r"))
            else:
                pool = measure("pool_build", lambda: discover.catalog_pool(raw) if args.catalog else discover.spec_pool(raw, spec), reps=1)
            record.doc["parity_pool"] = content_digest({"names": pool.names, "values": matrix_digest(pool.values)})
            record.doc["parity_pool_metadata"] = content_digest({name: getattr(pool, name) for name in
                ("names", "natives", "factories", "sources", "params", "skipped")})
            if args.discover_phase == "pool":
                args.pool_artifact.mkdir(parents=True, exist_ok=True)
                np.save(args.pool_artifact / "ms.npy", pool.ms, allow_pickle=False)
                np.save(args.pool_artifact / "values.npy", pool.values, allow_pickle=False)
                fields = {name: getattr(pool, name) for name in
                          ("names", "natives", "factories", "sources", "params", "skipped")}
                write_result(args.pool_artifact / "metadata.json", {"identity": identity, "fields": fields})
                record.doc["status"] = "complete"
                record.save()
                return
            ms = pool.ms
            step = time_bar_seconds(bar) * 1000
            if len(ms) < 4:
                raise ValueError("discovery needs at least four input bars")
            first = min(100, max(1, len(ms) // 10))
            windows = [(int(ms[i]), int(ms[min(i + 2, len(ms)-1)]))
                       for i in range(first, len(ms) - 2, max(3, len(ms) // 40))]
            def search(**overrides):
                if args.ablate == "reuse" and hasattr(discover, "discovery_reuse"):
                    discover.discovery_reuse.clear()
                options = dict(step_ms=step, windows=windows, mode="inside", lead_bars=1,
                    context=(int(ms[0]), int(ms[-1]) + step), display=(int(ms[0]), int(ms[-1]) + step),
                    top=3, max_points=256, bar_source={"method": "limen.data.HistoricalData.get_arrow_file",
                        "params": {"file_path": f"/opt/arrow/{series}/latest.arrow"}}, label="profile")
                return discover.discover(pool, **(options | overrides))
            def cold_search():
                if hasattr(discover, "discovery_reuse"):
                    discover.discovery_reuse.clear()
                return search()
            answer = measure("discover_full", cold_search)
            if args.transition_searches:
                began = time.perf_counter_ns()
                advanced = Path(directory) / "advanced-pool"
                advanced.mkdir()
                n, width = pool.values.shape
                extended = np.lib.format.open_memmap(advanced / "values.npy", mode="w+", dtype=pool.values.dtype,
                                                    shape=(n+1, width), fortran_order=True)
                for j in range(width):
                    extended[:n, j], extended[n, j] = pool.values[:, j], pool.values[-1, j]
                extended.flush()
                del extended
                np.save(advanced / "ms.npy", np.r_[pool.ms, pool.ms[-1]+step], allow_pickle=False)
                fields = {name: getattr(pool, name) for name in
                          ("names", "natives", "factories", "sources", "params", "skipped")}
                if hasattr(discover, "mapped_pool"):
                    pool = discover.mapped_pool(advanced, fields)
                else:
                    pool = discover.Pool(**fields, values=np.load(advanced / "values.npy", mmap_mode="r", allow_pickle=False),
                                         ms=np.load(advanced / "ms.npy", mmap_mode="r", allow_pickle=False))
                record.doc["transition_fixture"] = {"synthetic": True, "appended_rows": 1,
                    "appended_values": "repeat final pool row; outside the original statistical context",
                    "setup_ms": (time.perf_counter_ns()-began)/1e6}
                transitioned = measure("discover_source_advance", lambda: search(
                    display=(int(pool.ms[0]), int(pool.ms[-1])+step)))
                record.doc["parity_transition"] = content_digest({k: v for k, v in transitioned.items() if k != "timing_ms"})
            if args.related_searches:
                related = {"repeat": measure("discover_repeat", search),
                    "display": measure("discover_display_change", lambda: search(top=1, max_points=128,
                        display=(int(ms[len(ms)//2]), int(ms[-1])+step)))}
                sequence = [0]
                def new_windows():
                    sequence[0] += 1
                    return search(windows=[(a+step*sequence[0], b+step*sequence[0]) for a, b in windows])
                related["windows"] = measure("discover_changed_windows", new_windows)
                record.doc["parity_related"] = {name: content_digest({k: v for k, v in result.items()
                    if k != "timing_ms"}) for name, result in related.items()}
            record.doc["discovery"] = {"columns": len(pool.names), "pool_bytes": pool.values.nbytes,
                "timing_last_ms": answer["timing_ms"], "events": answer["target"]["events"],
                "intervals": answer["intervals"], "candidates": len(answer["candidates"])}
            record.doc["parity_discovery"] = content_digest({k: v for k, v in answer.items() if k != "timing_ms"})
        elif args.case == "research-discover":
            # The same process/IPC/JSON boundary used by /discover. No HTTP
            # socket/authentication time is included in these measurements.
            from mill.research import ResearchBusy
            serve._VIEW_SPECS["profile"] = spec
            ms, step = raw["datetime"].dt.epoch("ms").to_numpy(), time_bar_seconds(bar)*1000
            windows = [{"start": int(ms[i]), "end": int(ms[min(i+2, len(ms)-1)])}
                       for i in range(min(100, len(ms)//10), len(ms)-2, max(3, len(ms)//40))]
            body = {"bar": bar, "pool": {"kind": "view", "name": "profile"}, "windows": windows,
                    "context": {"start": int(ms[0]), "end": int(ms[-1])+step}, "top": 3, "max_points": 256}
            retries = [0]
            def request(overrides=None):
                deadline = time.monotonic()+45
                while True:
                    try:
                        return serve._research_discover(body | (overrides or {}))
                    except ResearchBusy:
                        retries[0] += 1
                        if time.monotonic() >= deadline:
                            raise
            try:
                results = {"cold": measure("research_discover_cold", request, reps=1),
                    "repeat": measure("research_discover_repeat", request),
                    "display": measure("research_discover_display", lambda: request({"top": 1, "max_points": 128,
                        "display": {"start": int(ms[len(ms)//2]), "end": int(ms[-1])+step}})),
                    "windows": measure("research_discover_windows", lambda: request({"windows": [
                        {"start": w["start"]+step, "end": w["end"]+step} for w in windows]}))}
                def semantic(result):
                    return {k: (dict(v, built_ms=0.) if k == "pool" else v)
                            for k, v in result.items() if k != "timing_ms"}
                record.doc["parity_research"] = {name: content_digest(semantic(result)) for name, result in results.items()}
                record.doc["preparation_retries"] = retries[0]
            finally:
                serve._RESEARCH.close()
                peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                record.doc["research_child_peak_rss_bytes"] = int(peak if sys.platform == "darwin" else peak*1024)
        elif args.case == "http":
            import base64
            from concurrent.futures import ThreadPoolExecutor
            from http.server import ThreadingHTTPServer
            import threading
            from urllib.parse import urlencode
            from urllib.request import Request, urlopen
            serve._APP_USERNAME, serve._APP_PASSWORD = "profile", "local"
            server_class = getattr(serve, "BoundedHTTPServer", ThreadingHTTPServer)
            server = server_class(("127.0.0.1", 0), serve._Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            prefix = f"http://127.0.0.1:{server.server_port}"
            route = "/data?" + urlencode({"bars": bar, "target": "next_return", "features": json.dumps(spec)})
            auth = {"Authorization": "Basic " + base64.b64encode(b"profile:local").decode()}
            def request(route, validate=False):
                began = time.perf_counter_ns()
                with urlopen(Request(prefix + route, headers=auth), timeout=15) as response:
                    payload = response.read()
                out = {"ms": (time.perf_counter_ns() - began) / 1e6, "bytes": len(payload)}
                if route.startswith("/qc?"):
                    out["report_digest"] = content_digest(json.loads(payload))
                if validate:
                    out["data_digest"] = content_digest(json.loads(payload))
                return out
            try:
                measure("http_data_first", lambda: request(route), reps=1)
                measure("http_data_warm", lambda: request(route))
                serve._RESULTS.clear()
                from unittest.mock import patch
                with patch.object(serve, "build_from_spec", wraps=serve.build_from_spec) as build:
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        measure("http_identical_cold_batch_c4", lambda: list(executor.map(request, [route] * 4)), reps=1)
                    record.doc["identical_cold_build_calls"] = build.call_count
                for concurrency in (1, 4, 8):
                    with ThreadPoolExecutor(max_workers=concurrency) as executor:
                        results = measure(f"http_warm_batch_c{concurrency}",
                            lambda: list(executor.map(request, [route] * 16)), reps=1)
                    record.doc[f"http_c{concurrency}"] = results
                if hasattr(serve, "_research_report"):
                    serve._research_report(raw.select("datetime").head(4), None)
                    owner, method = serve, "_research_report"
                else:
                    owner, method = qc, "report"
                entered = threading.Event()
                real = getattr(owner, method)
                def signaled(*args, **kwargs):
                    entered.set()
                    return real(*args, **kwargs)
                qc_route = route.replace("/data?", "/qc?")
                samples = []
                with patch.object(owner, method, signaled), ThreadPoolExecutor(1) as executor:
                    future = executor.submit(request, qc_route)
                    if not entered.wait(10):
                        raise RuntimeError("QC never entered its report stage")
                    until = time.monotonic() + 25
                    while (not future.done() or len(samples) < 16) and time.monotonic() < until:
                        samples.append(request(route))
                        time.sleep(0.02)
                    qc_result = future.result(timeout=15)
                record.doc["http_during_qc"] = {"data": samples, "qc": qc_result,
                    "p50_ms": float(np.percentile([s["ms"] for s in samples], 50)),
                    "p95_ms": float(np.percentile([s["ms"] for s in samples], 95)),
                    "max_ms": max(s["ms"] for s in samples)}
                # Output validation is outside the timed request batches.
                record.doc["parity_http_data"] = request(route, validate=True)["data_digest"]
                record.doc["parity_http_qc"] = qc_result["report_digest"]
            finally:
                if hasattr(serve, "_RESEARCH"):
                    serve._RESEARCH.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
        record.doc["status"] = "complete"
        record.save()


def main():
    args = arguments()
    if args.worker:
        # Default SIGALRM terminates even stuck native code if the supervisor disappears.
        signal.alarm(49)
        try:
            worker(args)
        except Exception as exc:
            if args.output.exists():
                doc = json.loads(args.output.read_text())
                doc.update(status="error", error=type(exc).__name__ + ": " + str(exc))
                write_result(args.output, doc)
            raise
        return 0
    def finish(status):
        if args.output.exists():
            doc = json.loads(args.output.read_text())
            doc["status"] = status
            write_result(args.output, doc)

    def interrupted(signum, frame):
        raise InterruptedError(f"supervisor received signal {signum}")

    handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        # Parent ownership makes fixture cleanup survive a forcibly terminated worker.
        with tempfile.TemporaryDirectory(prefix="mill_engine_sweep_") as scratch:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                                        "--worker", "--scratch", scratch], start_new_session=True)
            try:
                code = process.wait(timeout=50)
                if code == -signal.SIGALRM:
                    finish("timeout_49s")
                    return 124
                if code != 0:
                    finish("error")
                return code
            except BaseException as exc:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                finish("timeout_50s" if isinstance(exc, subprocess.TimeoutExpired) else "interrupted")
                if isinstance(exc, subprocess.TimeoutExpired):
                    return 124
                raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
