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


CASES = ("startup", "data", "features", "factories", "views", "qc", "discover", "http", "memory")
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
    return args


def peak_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


class Recorder:
    def __init__(self, args):
        self.args = args
        self.doc = {"schema": 1, "case": args.case, "status": "running", "worker_pid": os.getpid(),
                    "peak_rss_scope": "whole worker, including imports, complete source decoding and fixture setup", "measurements": []}
        self.save()

    def save(self):
        self.doc["peak_rss_bytes"] = peak_bytes()
        self.args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.doc, indent=2, allow_nan=False) + "\n")
        tmp.replace(self.args.output)

    def measure(self, name, fn, reps=None, **metadata):
        self.doc["active_measurement"] = name
        self.save()
        samples, result = [], None
        for _ in range(self.args.reps if reps is None else reps):
            began = time.perf_counter_ns()
            result = fn()
            samples.append((time.perf_counter_ns() - began) / 1e6)
        self.doc["measurements"].append({"name": name, "samples_ms": samples,
                                         "median_ms": statistics.median(samples), **metadata})
        self.doc.pop("active_measurement", None)
        self.save()
        return result


def environment():
    from importlib.metadata import distribution, version
    import numba
    import polars as pl
    try:
        checkout = Path(__file__).resolve().parents[1]
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        sha = None
    return {"python": sys.version, "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "polars_threads": pl.thread_pool_size(),
            "numba_threads": numba.get_num_threads(), "numba_layer": numba.threading_layer(),
            "versions": {name: version(name) for name in
                         ("polars", "numpy", "numba", "scipy", "scikit-learn", "statsmodels", "vaquum_limen")},
            "limen_origin": json.loads(distribution("vaquum_limen").read_text("direct_url.json") or "null"),
            "thread_environment": {k: os.environ.get(k) for k in
                                   ("POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
            "engine_sha": sha}


def source_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def worker(args):
    record = Recorder(args)
    # Set before Mill imports; all cache/store/config mutations stay in this directory.
    scratch = nullcontext(str(args.scratch)) if args.scratch else tempfile.TemporaryDirectory(prefix="mill_engine_sweep_")
    with scratch as directory:
        for key, name in (("MILL_ARROW_DIR", "arrow"), ("MILL_CACHE_DIR", "cache"), ("MILL_CONFIG_DIR", "config")):
            os.environ[key] = str(Path(directory) / name)
        began = time.perf_counter_ns()
        from mill import serve, qc, discover, targets, feature_registry
        import numpy as np
        import polars as pl
        from mill.get_data import GetData
        from mill.table import build_from_spec
        from mill.settings import arrow_latest_path
        from mill.bars import arrow_series, time_bar_seconds
        record.doc["import_ms"] = (time.perf_counter_ns() - began) / 1e6
        record.doc["environment"] = environment()
        if args.case == "startup":
            record.doc["peak_rss_scope"] = "imports only; source parquet is not opened"
            record.doc["status"] = "complete"
            record.save()
            return
        record.doc["source"] = {"file": args.parquet.name, "sha256": source_digest(args.parquet)}
        raw = GetData().from_parquet(args.parquet)
        source_rows = raw.height
        if args.rows:
            raw = raw.tail(args.rows)
        record.doc["input"] = {"source_rows": source_rows, "rows": raw.height,
                               "columns": raw.width, "frame_bytes": raw.estimated_size(),
                               "start": str(raw["datetime"][0]), "end": str(raw["datetime"][-1]),
                               "bar": args.bar, "k": args.k, "reps": args.reps}
        bar = args.bar
        series = arrow_series(bar)
        path = arrow_latest_path(series)
        path.parent.mkdir(parents=True)
        raw.with_columns(pl.col("datetime").dt.epoch("ns").alias("ts")).drop("datetime").write_ipc(
            path, compression="uncompressed", record_batch_size=max(1, raw.height))
        # Make every fixture bar closed, including sources with future timestamps.
        stamp = max(time.time_ns(), int(raw["datetime"].dt.epoch("ns")[-1]) + time_bar_seconds(bar) * 1_000_000_000)
        os.utime(path, ns=(stamp, stamp))
        spec = [{"factory": name, "params": {}} for name in FACTORIES[:args.k]]
        measure = record.measure
        if args.case == "data":
            loader = GetData()
            measure("arrow_read_normalize_select", lambda: loader._load_arrow(series, None, None))
            measure("get_data_cold_cache", lambda: loader.get_time_bars(bar), reps=1)
            measure("get_data_disk_cache_hit", lambda: loader.get_time_bars(bar))
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
            measure("grade_map", lambda: serve.grading.grade_map(report, n_rows=frame.height, target="next_return"))
            measure("report_json", lambda: json.dumps(report, allow_nan=False).encode())
            record.doc["kernel_cache_entries"] = len(qc._KERNEL_CACHE)
            record.doc["pair_count"] = len(prim.pairs)
        elif args.case == "discover":
            pool = measure("pool_build", lambda: discover.catalog_pool(raw) if args.catalog else discover.spec_pool(raw, spec), reps=1)
            ms = pool.ms
            step = time_bar_seconds(bar) * 1000
            if len(ms) < 4:
                raise ValueError("discovery needs at least four input bars")
            first = min(100, max(1, len(ms) // 10))
            windows = [(int(ms[i]), int(ms[min(i + 2, len(ms)-1)]))
                       for i in range(first, len(ms) - 2, max(3, len(ms) // 40))]
            answer = measure("discover_full", lambda: discover.discover(pool, step_ms=step,
                windows=windows, mode="inside", lead_bars=1,
                context=(int(ms[0]), int(ms[-1]) + step), display=(int(ms[0]), int(ms[-1]) + step),
                top=3, max_points=256, bar_source=serve._BAR_DATA_SOURCES[bar], label="profile"))
            record.doc["discovery"] = {"columns": len(pool.names), "pool_bytes": pool.values.nbytes,
                "timing_last_ms": answer["timing_ms"], "events": answer["target"]["events"],
                "intervals": answer["intervals"], "candidates": len(answer["candidates"])}
        elif args.case == "http":
            import base64
            from concurrent.futures import ThreadPoolExecutor
            from http.server import ThreadingHTTPServer
            import threading
            from urllib.parse import urlencode
            from urllib.request import Request, urlopen
            serve._APP_USERNAME, serve._APP_PASSWORD = "profile", "local"
            server = ThreadingHTTPServer(("127.0.0.1", 0), serve._Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            prefix = f"http://127.0.0.1:{server.server_port}"
            route = "/data?" + urlencode({"bars": bar, "target": "next_return", "features": json.dumps(spec)})
            auth = {"Authorization": "Basic " + base64.b64encode(b"profile:local").decode()}
            def request(route):
                began = time.perf_counter_ns()
                with urlopen(Request(prefix + route, headers=auth), timeout=15) as response:
                    payload = response.read()
                return {"ms": (time.perf_counter_ns() - began) / 1e6, "bytes": len(payload)}
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
            finally:
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
                args.output.write_text(json.dumps(doc, indent=2) + "\n")
            raise
        return 0
    def finish(status):
        if args.output.exists():
            doc = json.loads(args.output.read_text())
            doc["status"] = status
            args.output.write_text(json.dumps(doc, indent=2) + "\n")

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
