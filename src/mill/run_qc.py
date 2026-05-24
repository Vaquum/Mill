from __future__ import annotations

import argparse
import importlib
from types import ModuleType

from . import qc
from .bars import DOLLAR_BAR_USD, TIME_BAR_SECONDS
from .get_data import GetData
from .settings import DEFAULT_BAR, PROJECT_END_STR, PROJECT_START_STR
from .targets import apply as apply_target


def _view_module(name: str) -> ModuleType:
    clean = name.removeprefix("mill.views.")
    return importlib.import_module(f"{__package__}.views.{clean}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mill-qc")
    parser.add_argument("--bar", default=DEFAULT_BAR)
    parser.add_argument("--view", default="gen4")
    parser.add_argument("--target")
    parser.add_argument("--start", default=PROJECT_START_STR)
    parser.add_argument("--end", default=PROJECT_END_STR)
    return parser


def _load_raw(gd: GetData, bar: str, start: str, end: str):
    if bar in TIME_BAR_SECONDS:
        return gd.get_time_bars(bar, start_date_limit=start, end_date_limit=end)
    if bar in DOLLAR_BAR_USD:
        return gd.get_dollar_bars(bar, start_date_limit=start, end_date_limit=end)
    try:
        return gd.get_time_bars(bar, start_date_limit=start, end_date_limit=end)
    except ValueError:
        return gd.get_dollar_bars(bar, start_date_limit=start, end_date_limit=end)


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    view = _view_module(args.view)
    target = args.target if args.target is not None else getattr(view, "TARGET", "none")
    print("Loading data...")
    gd = GetData()
    raw = _load_raw(gd, args.bar, args.start, args.end)

    print("Building features...")
    df = view.build(raw)

    if target != "none":
        print("Building target...")
        raw_with_target = apply_target(raw, target)
        print("Joining target...")
        df = df.join(raw_with_target.select(["datetime", target]), on="datetime", how="left")

    print("Running QC...")
    report = qc.report(df, target if target != "none" else None)

    columns = report["columns"]
    rows = report["rows"]

    print(f"{'Metric':<30} | " + " | ".join(f"{c:<25}" for c in columns[1:]))
    print("-" * 120)
    for row in rows:
        metric = row[0]
        vals = row[1:]
        val_strs = []
        for v in vals:
            if v is None:
                val_strs.append(f"{'None':<25}")
            elif isinstance(v, float):
                val_strs.append(f"{v:<25.4f}")
            else:
                val_strs.append(f"{str(v):<25}")
        print(f"{metric:<30} | " + " | ".join(val_strs))

if __name__ == "__main__":
    main()
