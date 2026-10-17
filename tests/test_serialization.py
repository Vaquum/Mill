"""Native encoding must preserve parsed JSON values exactly, including Float32."""
import orjson
import numpy as np
import polars as pl
import pytest

from mill.serialization import encode_rows


@pytest.mark.parametrize("n", [0, 3, 256, 5001])
def test_native_rows_match_orjson_for_scalar_types_and_escaping(n):
    rng = np.random.default_rng(771)
    frame = pl.DataFrame({"float": rng.normal(size=n),
        "float32": pl.Series(rng.normal(size=n), dtype=pl.Float32),
        "int": pl.Series(rng.integers(-10000, 10000, n), dtype=pl.Int64),
        "uint": pl.Series([2**64-1]*n, dtype=pl.UInt64),
        "bool": pl.Series(rng.integers(0, 2, n).astype(bool)),
        "text": pl.Series(([None, 'a"b\\c\n\t}\x00', 'ä🙂', 'null']*((n+3)//4))[:n], dtype=pl.String)})
    expected = orjson.loads(orjson.dumps({"columns": frame.columns, "rows": frame.rows()}))
    assert orjson.loads(encode_rows(frame)) == expected


def test_float_extremes_and_signed_zero_round_trip_exactly():
    rng = np.random.default_rng(819)
    floats = np.ldexp(rng.uniform(-1, 1, 5000), rng.integers(-1074, 1024, 5000))
    floats[:8] = [-0., 0., np.inf, -np.inf, np.nan, np.finfo(float).max,
                  np.nextafter(0., 1.), np.nextafter(1., 2.)]
    frame = pl.DataFrame({"value": floats})
    expected = orjson.loads(orjson.dumps({"columns": frame.columns, "rows": frame.rows()}))
    actual = orjson.loads(encode_rows(frame))
    assert actual == expected
    assert np.signbit(actual["rows"][0][0]) and not np.signbit(actual["rows"][1][0])


def test_unsupported_scalar_schemas_keep_the_original_encoder():
    frame = pl.DataFrame({"nested": [[1, 2], None]*150})
    assert encode_rows(frame) == orjson.dumps({"columns": frame.columns, "rows": frame.rows()})
