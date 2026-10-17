"""Encode normalized table rows without constructing Python objects per cell."""
import orjson
import polars as pl

_SCALARS = {pl.Boolean, pl.String, pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32,
            pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}


def encode_rows(frame):
    if frame.height < 256 or not frame.width or any(dtype not in _SCALARS for dtype in frame.dtypes):
        return orjson.dumps({"columns": frame.columns, "rows": frame.rows()})
    tokens = []
    for name, dtype in frame.schema.items():
        value = pl.col(name)
        if dtype.is_float():
            # rows() promotes Float32 to Python Float64. Preserve the promoted
            # value, including its extra digits, and orjson's nonfinite nulls.
            value = pl.when(value.is_finite()).then(value.cast(pl.Float64).cast(pl.String)).otherwise(pl.lit("null"))
        elif dtype == pl.String:
            # Delegate escaping to Polars' JSON encoder, then remove only the
            # fixed outer object. Quoting by hand would mishandle user strings.
            value = pl.struct(value.alias("v")).struct.json_encode().str.slice(5).str.strip_suffix("}")
        else:
            value = value.cast(pl.String)
        tokens.append(value.fill_null("null"))
    rows = frame.select(pl.concat_str(pl.lit("["), pl.concat_str(tokens, separator=","),
                                     pl.lit("]")).str.join(",")).item() or ""
    return b'{"columns":' + orjson.dumps(frame.columns) + b',"rows":[' + rows.encode() + b']}'
