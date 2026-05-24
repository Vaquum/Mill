import polars as pl
from . import qc
from .get_data import GetData
from .targets import apply as apply_target
from .views import gen4 as view

def main():
    print("Loading data...")
    gd = GetData()
    raw = gd.get_time_bars("1h", start_date_limit="2024-01-01", end_date_limit="2026-05-15")
    
    print("Building target...")
    raw_with_target = apply_target(raw, view.TARGET)
    
    print("Building features...")
    df = view.build(raw)
    
    print("Joining target...")
    df = df.join(raw_with_target.select(["datetime", view.TARGET]), on="datetime", how="left")
    
    print("Running QC...")
    report = qc.report(df, view.TARGET)
    
    # Print the report
    columns = report["columns"]
    rows = report["rows"]
    
    # Simple formatting
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
