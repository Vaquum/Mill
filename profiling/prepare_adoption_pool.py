"""Advance a resumable, input-identified pool; this is setup, not a timing run."""
import argparse
import json
import os
from pathlib import Path
import signal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    signal.alarm(49)
    os.environ["MILL_CACHE_DIR"] = str(args.cache.resolve())
    from mill import pool_store, serve
    from mill.get_data import GetData
    raw = GetData().from_parquet(args.parquet)
    try:
        pool, _, _ = pool_store.pool_for(raw, "1h_time_bars", None, serve._frame_digest(raw), quantum=2)
    except pool_store.PoolPending:
        print(json.dumps({"ready": False, "completed_parts": len(list(args.cache.rglob("*.arrow")))
                         + len(list(args.cache.rglob("*.skip")))}))
    else:
        print(json.dumps({"ready": True, "rows": raw.height, "columns": len(pool.names)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
