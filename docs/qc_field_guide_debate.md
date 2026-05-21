# QC field guide audit log

Historical record of the audit that produced the current `docs/QC.md` and the
follow-up changes to `qc.py` and `grading.py`. Kept for traceability; the
operative documents are `docs/QC.md`, `qc.py`, `grading.py`, and
`grading_rules.py`.

## Scope of the original audit

Target: evaluate an earlier draft of `docs/QC.md` against the `qc.py` revision
that existed at the time.

Primary code read:
- `qc.py`
- `index.html`
- `docs/QC.md`

No data was pulled.

## Debaters

| Debater | Invocation | Note |
|---|---|---|
| Opus | Claude Code, `--model opus --effort max` | Literal `claude-opus-4-7-max` was not accepted by Claude Code. |
| Codex | Codex CLI, `gpt-5.5`, `model_reasoning_effort=xhigh` | Read-only execution. |
| Gemini | Cursor CLI, `gemini-3.1-pro` | Provider failed on some retries; successful rounds recorded below. |
| Grok | Cursor CLI, `grok-4.3` | Cursor CLI only. |
| Moderator | Codex session at the time | Direct code witness and final alignment. |

## Round record

| Round | Target | Alignment |
|---|---|---|
| 1 | Fact audit | All active debaters found the prose broadly faithful but not exact enough on `ref`, regime chunks, missingness, and `block_size`. |
| 2 | Text fix vs system critique | Consensus: fix prose only. Do not propose QC implementation changes. |
| 3 | Voice and hierarchy | Majority: compress to a compact field guide. Grok initially argued for table-first manual form. |
| 4 | Style conflict | Opus, Codex, Gemini, and moderator chose compact prose-led guide. Grok dissented for table-only. |
| 5 | Final alignment | Opus, Codex, Gemini, Grok, and moderator accepted compact field guide plus surgical factual fixes. |

## What was identified vs what is true today

The audit's "required fixes" list was written against the older `qc.py` and
older `docs/QC.md`. Several items have since been addressed in code, not just
in prose. Annotated for current accuracy:

1. `ref` is frontend presentation, inserted by `index.html` from `QC_META`. It is not emitted by `qc.report`.
   *Today:* still true. The UI no longer renders a REF column; row-level guidance lives in tooltips (`QC_META.tip`) and the grading thresholds.

2. `missing_rate` is `df[c].null_count() / n`. It counts Polars nulls, not floating `NaN`.
   *Today:* **outdated.** `qc._health_rows` now computes `missing_rate` from the Float64-cast numpy arrays via `np.isnan(...).sum() / n`, so it counts both Polars nulls and NaN/cast failures.

3. Regime rows split the input row order into four contiguous chunks.
   *Today:* `qc.report()` now sorts by `datetime` at entry when the column is present, so regime chunks are chronological by construction. Caller order no longer matters when `datetime` is a column.

4. `block_size=24` can mean a real first crossing at lag 24 or fallback.
   *Today:* `_block_size()` returns `int | None`. None when `n <= 3` or the ACF computation fails; an integer 1..200 when a lag clears the threshold; `_BLOCK_MAX_LAG` (200) when no lag clears within the search window. The fallback `_BLOCK_FALLBACK = 24` is applied at use sites via `_resolve_block(...)`. Block size is no longer surfaced as a row in the report.

5. Mutual information must stay target-scoped. Pairwise MI always uses `mutual_info_regression`; edge MI switches to classification only when target unique count is `<= 10`.
   *Today:* still accurate.

6. Thresholds are operator guidance from the UI layer, not Python source constants.
   *Today:* the interpretive thresholds now live in `grading_rules.py` as Python data; the UI per-row tooltips describe them. The threshold ladders are explicit and editable in one place.

7. The draft said "six layers" while naming seven. *Today:* the new draft is organised into the five groups the UI actually renders (HEALTH / STATIONARITY / REDUNDANCY / EDGE / REGIME). Persistence/decorrelation rows were removed from `qc.report()` output.

8. Subsampling specifics. *Today:* still accurate — ADF/KPSS via `_stride`; pairwise and edge caps via `np.linspace` grid indices.

## Items that are still keep

- `None` means undefined, not zero.
- Target pairwise and edge cells are blank by construction.
- ADF and KPSS must be read jointly.
- Pairwise metrics use the strict all-feature non-NaN row filter.
- The QC report creates candidates only; OOS validation, leakage review, costs, and live monitoring remain outside the panel.

## Outdated statements that have since been replaced

- The audit's keep list said: *"Bootstrap RNG is module-scoped and reproducible only under identical call sequence after import."* This is **outdated.** `_bootstrap_spearman_ci` now creates a fresh `np.random.default_rng(seed)` per call where `seed` is derived from the column name via `zlib.crc32`. CIs are now per-column deterministic regardless of call order or imports.

## Style decision (still operative)

Final form: compact field guide.

Structure:
- short opening contract
- read-order funnel
- one compact decision table
- metric sections only where interpretation changes action
- known limitations kept short and code-faithful

Rejected:
- table-only manual
- long atmospheric prose
- code improvement proposals
- claims not witnessed in `qc.py` or `index.html`
