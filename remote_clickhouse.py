from __future__ import annotations

import base64
import json
import shlex
import subprocess

import polars as pl
import pyarrow as pa

from frames import normalize_datetime


_REMOTE_SCRIPT = """
import base64
import json
import sys
import pyarrow as pa
from tdw_control_plane.utils.get_clickhouse_client import get_clickhouse_client

payload = json.loads(sys.stdin.read())
table = get_clickhouse_client().query_arrow(
    payload["query"],
    parameters=payload["parameters"],
)

sink = pa.BufferOutputStream()
with pa.ipc.new_stream(sink, table.schema) as writer:
    writer.write_table(table)

sys.stdout.write(base64.b64encode(sink.getvalue().to_pybytes()).decode())
"""


def query_frame(
    ssh_target: str,
    query: str,
    parameters: dict[str, str | int | float],
) -> pl.DataFrame:
    payload = json.dumps({"query": query, "parameters": parameters})
    command = (
        "c=$(docker ps -q --filter label=com.docker.compose.service=dagster | head -n1); "
        f"docker exec -i \"$c\" python -c {shlex.quote(_REMOTE_SCRIPT)}"
    )
    result = subprocess.run(
        ["ssh", ssh_target, command],
        input=payload,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())

    reader = pa.ipc.open_stream(pa.BufferReader(base64.b64decode(result.stdout)))
    return normalize_datetime(pl.from_arrow(reader.read_all()))
