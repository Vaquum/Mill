"""Deterministic cross-version factory cases; no market data or timing claims."""
import hashlib
import json
import random
import warnings
from datetime import datetime, timedelta, timezone


CASES = ("empty", "one", "short", "ties", "missing", "infinite", "nondefault", "float32", "integer")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def frame_digest(frame):
    import numpy as np
    import polars as pl
    result = hashlib.sha256(digest(dict(frame.schema)).encode())
    for column in frame:
        result.update(column.is_null().to_numpy().tobytes())
        if column.dtype == pl.String:
            result.update(json.dumps(column.to_list(), ensure_ascii=False, separators=(",", ":")).encode())
            continue
        values = column.to_physical()
        if values.null_count():
            values = values.fill_null(False if values.dtype == pl.Boolean else 0)
        array = values.to_numpy()
        if np.issubdtype(array.dtype, np.floating):
            array = array.copy()
            array[np.isnan(array)] = np.nan  # payload/sign is not a NaN value
        if array.dtype.hasobject:
            result.update(digest(column.to_list()).encode())
        else:
            result.update(array.tobytes())
    return result.hexdigest()


def bars():
    import polars as pl
    rows = []
    for i in range(73):
        close = 100.+i*.05+((i*17)%23-11)*.02
        opening = close+(i%3-1)*.03
        high, low = max(opening, close)+.07, min(opening, close)-.05
        volume, ratio = 1000.+17*i, .5+.02*((i*3)%9-4)
        liquidity = 100.+2*i
        rows.append({"datetime": datetime(2024, 1, 1, tzinfo=timezone.utc)+timedelta(hours=i),
            "open": opening, "high": high, "low": low, "close": close, "volume": volume,
            "mean": (opening+high+low+close)/4, "std": .04+(i%5)*.002,
            "maker_ratio": ratio, "no_of_trades": 100+i%51,
            "open_liquidity": liquidity, "high_liquidity": liquidity+5,
            "low_liquidity": liquidity-2, "close_liquidity": liquidity+1,
            "liquidity_sum": liquidity*6, "maker_volume": volume*ratio,
            "maker_liquidity": liquidity*6*ratio})
    return pl.DataFrame(rows).with_columns(pl.col("datetime").dt.cast_time_unit("ms"))


def inputs(case):
    import polars as pl
    raw = bars().head({"empty": 0, "one": 1, "short": 13}.get(case, 73))
    floating = [name for name, dtype in raw.schema.items() if dtype.is_float()]
    if case == "ties":
        raw = raw.with_columns([pl.lit(0. if name == "std" else 100.).cast(dtype).alias(name)
            for name, dtype in raw.schema.items() if name != "datetime"])
    elif case in ("missing", "infinite"):
        columns = []
        for name in floating:
            values = raw[name].to_list()
            for i in range(5, raw.height, 17):
                values[i] = None if case == "missing" else float("inf")
            if case == "missing":
                for i in range(9, raw.height, 19):
                    values[i] = float("nan")
            columns.append(pl.Series(name, values, dtype=raw.schema[name]))
        raw = raw.with_columns(columns)
    elif case == "float32":
        raw = raw.with_columns([pl.col(name).cast(pl.Float32) for name in floating])
    elif case == "integer":
        raw = raw.with_columns([pl.col(name).round(0).cast(pl.Int64) for name in floating])
    return raw


def parameters(spec):
    result = {}
    for item in spec:
        value = item["default"]
        if value is None:
            value = {"int": 7, "float": .01, "str": "close", "bool": False}[item["type"]]
        elif item["type"] == "int" and value >= 4:
            value = max(2, value//2)
        elif item["type"] == "float" and value > 0:
            value *= .8
        elif item["type"] == "bool":
            value = not value
        result[item["name"]] = value
    return result


def snapshot():
    import numpy as np
    from importlib.metadata import distribution, version
    from mill import feature_registry
    catalog = feature_registry.catalog()
    output = {"catalog": catalog, "versions": {name: version(name) for name in
        ("polars", "numpy", "numba", "scipy", "scikit-learn", "statsmodels", "vaquum_limen")},
        "origin": json.loads(distribution("vaquum_limen").read_text("direct_url.json")), "cases": {}}
    warnings.filterwarnings("ignore")
    for case in CASES:
        np.random.seed(0)
        random.seed(0)
        raw = inputs(case)
        results = {}
        for item in catalog:
            kwargs = parameters(item["params"]) if case == "nondefault" else {}
            try:
                frame = feature_registry.REGISTRY[item["name"]]["producer"](raw, **kwargs)
                results[item["name"]] = {"schema": str(frame.schema), "digest": frame_digest(frame)}
            except Exception as exc:
                results[item["name"]] = {"error": type(exc).__name__, "detail": str(exc)}
        output["cases"][case] = {"rows": raw.height, "input_digest": frame_digest(raw), "results": results}
    return output
