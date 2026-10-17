"""Cold full-catalog /discover process, IPC and JSON boundary, with a 50s limit."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

from engine_sweep import Recorder, content_digest, environment, source_digest, write_result


def worker(source, output, bar, scratch):
    for key, name in (("MILL_ARROW_DIR", "arrow"), ("MILL_CACHE_DIR", "cache"),
                      ("MILL_CONFIG_DIR", "config")):
        os.environ[key] = str(scratch/name)
    os.environ["MILL_RESEARCH_TMPDIR"] = str(scratch)
    record = Recorder(SimpleNamespace(output=output, reps=1))
    record.doc.update(profiler_path="profiling/catalog_service.py",
        profiler_sha256=source_digest(Path(__file__)),
        common_profiler_sha256=source_digest(Path(__file__).with_name("engine_sweep.py")),
        ablation="none", service_pool_kind="catalog",
        boundary="same research process, IPC and JSON as /discover; no HTTP socket/auth")
    began = time.perf_counter_ns()
    from mill import serve
    from mill.bars import arrow_latest_path, arrow_series, time_bar_seconds
    from mill.get_data import GetData
    from mill.research import ResearchBusy
    import polars as pl
    record.doc["import_ms"] = (time.perf_counter_ns()-began)/1e6
    record.doc["environment"] = environment()
    record.doc["source"] = {"file": source.name, "sha256": source_digest(source)}
    raw = record.measure("source_read", lambda: GetData().from_parquet(source))
    record.doc["input"] = {"source_rows": raw.height, "rows": raw.height,
        "columns": raw.width, "frame_bytes": raw.estimated_size(), "bar": bar,
        "start": str(raw["datetime"][0]), "end": str(raw["datetime"][-1]),
        "k": "catalog", "reps": 1}
    began = time.perf_counter_ns()
    path = arrow_latest_path(arrow_series(bar))
    path.parent.mkdir(parents=True)
    raw.with_columns(pl.col("datetime").dt.epoch("ns").alias("ts")).drop("datetime").write_ipc(
        path, compression="uncompressed", record_batch_size=max(1, raw.height))
    step = time_bar_seconds(bar)*1000
    ms = raw["datetime"].dt.epoch("ms").to_numpy()
    stamp = max(time.time_ns(), int(ms[-1]+step)*1_000_000)
    os.utime(path, ns=(stamp, stamp))
    record.doc["fixture_prepare_ms"] = (time.perf_counter_ns()-began)/1e6
    body = {"bar": bar, "pool": {"kind": "catalog"}, "top": 3, "max_points": 256,
        "context": {"start": int(ms[0]), "end": int(ms[-1])+step},
        "windows": [{"start": int(ms[i]), "end": int(ms[min(i+2,len(ms)-1)])}
                    for i in range(min(100,len(ms)//10),len(ms)-2,max(3,len(ms)//40))]}
    retries = [0]
    def request():
        deadline = time.monotonic()+44
        while True:
            try:
                return serve._research_discover(body)
            except ResearchBusy:
                retries[0] += 1
                if time.monotonic() >= deadline:
                    raise
    try:
        result = record.measure("catalog_service_cold", request)
        semantic = {k: (dict(v, built_ms=0.) if k == "pool" else v)
                    for k,v in result.items() if k != "timing_ms"}
        record.doc["parity_research"] = content_digest(semantic)
        record.doc["pool"] = result["pool"]
        record.doc["preparation_retries"] = retries[0]
        record.doc["status"] = "complete"
    finally:
        serve._RESEARCH.close()
        import resource
        peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        record.doc["research_child_peak_rss_bytes"] = int(peak if sys.platform == "darwin" else peak*1024)
        record.save()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bar", choices=("1h_time_bars", "15m_time_bars"), required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--scratch", type=Path)
    args = parser.parse_args()
    if args.output.resolve() == args.parquet.resolve():
        parser.error("output cannot overwrite the source")
    if args.worker:
        signal.alarm(49)
        try:
            worker(args.parquet, args.output, args.bar, args.scratch)
        except Exception as exc:
            if args.output.exists():
                doc = json.loads(args.output.read_text())
                doc.update(status="error", error=type(exc).__name__+": "+str(exc))
                write_result(args.output, doc)
            raise
        return 0
    with tempfile.TemporaryDirectory(prefix="mill-catalog-service-") as directory:
        child = subprocess.Popen([sys.executable, __file__, *sys.argv[1:], "--worker", "--scratch", directory],
                                 start_new_session=True)
        code, status = 124, "interrupted"
        try:
            code = child.wait(timeout=50)
            status = "timeout_49s" if code == -signal.SIGALRM else "error"
        except subprocess.TimeoutExpired:
            status = "timeout_50s"
        finally:
            if code != 0:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            child.wait(timeout=1)
        if code != 0 and args.output.exists():
            doc = json.loads(args.output.read_text())
            doc["status"] = status
            write_result(args.output, doc)
        return 124 if code == -signal.SIGALRM else code


if __name__ == "__main__":
    raise SystemExit(main())
