# profiling

Developer tooling for profiling and validating the QC pipeline. Not part of the
package and not run by the test suite — these are standalone scripts run by hand
in venv `mill`.

They read real BTCUSDT bars from the local GetData cache (`~/.cache/mill/*.parquet`,
populated by normal Mill use), so they need that cache present on the machine.

- **`qc_profile.py`** — profiles every step of the `/qc` pipeline (load → build →
  serialize → group → `qc.report` → grade → json) and breaks `qc.report` into its
  sections, sweeping the feature count. Optional `--cprofile` attributes time to
  the underlying library calls.
- **`qc_validate.py`** — captures (`--dump`) and compares (`--check`) exact QC
  outputs across a grid of (feature count, target, row_mode). A byte-for-byte
  JSON match proves an optimization is lossless. Use it to gate any change to
  `qc.py`: dump on the old code, check on the new.
- **`qc_memo_demo.py`** — shows the kernel-memo win on the real flows: cold vs
  identical re-run vs incremental (+1 feature).

Example:

    python profiling/qc_validate.py --dump  /tmp/qc_baseline.json   # before change
    python profiling/qc_validate.py --check /tmp/qc_baseline.json   # after change
    python profiling/qc_profile.py  --ks 1,2,4,8,16 --reps 3
    python profiling/qc_memo_demo.py
