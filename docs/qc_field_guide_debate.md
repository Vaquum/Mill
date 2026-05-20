# QC field guide debate

## Scope

Target: evaluate the proposed "QC report - field guide for the operator" against the current implementation.

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
| Moderator | Current Codex session | Direct code witness and final alignment. |

## Round record

| Round | Target | Alignment |
|---|---|---|
| 1 | Fact audit | All active debaters found the prose broadly faithful but not exact enough on `ref`, regime chunks, missingness, and `block_size`. |
| 2 | Text fix vs system critique | Consensus: fix prose only. Do not propose QC implementation changes. |
| 3 | Voice and hierarchy | Majority: compress to a compact field guide. Grok initially argued for table-first manual form. |
| 4 | Style conflict | Opus, Codex, Gemini, and moderator chose compact prose-led guide. Grok dissented for table-only. |
| 5 | Final alignment | Opus, Codex, Gemini, Grok, and moderator accepted compact field guide plus surgical factual fixes. |

## Final verdict

The current draft is close in substance but not publish-ready as-is.

Use it as a compact field guide, not as a table-only manual and not as long literary prose. Keep prose only where it changes operator judgment. Cut ornament, transitions, and repeated caveats.

## Required fixes

1. `ref` is frontend presentation, inserted by `index.html` from `QC_META`. It is not emitted by `qc.report`.
2. `missing_rate` is `df[c].null_count() / n`. It counts Polars nulls, not floating `NaN`.
3. Regime rows split the input row order into four contiguous chunks. Say row-order chunks, not statistical quartiles. Ordering is the caller's responsibility.
4. `block_size=24` can mean a real first crossing at lag 24 or fallback. Fallback covers `n<=3`, ACF failure, and no lag clearing `abs(ACF) < 0.05`.
5. Mutual information must stay target-scoped. Pairwise MI always uses `mutual_info_regression`; edge MI switches to classification only when target unique count is `<=10`.
6. Thresholds are operator guidance from the UI layer, not Python source constants.
7. The draft says "six layers" while naming seven if decorrelation is separate. Either merge persistence/decorrelation or say seven.
8. If describing subsampling exactly: ADF/KPSS use `_stride`; pairwise and edge caps use `np.linspace` grid indices.

## Keep

- `None` means undefined, not zero.
- Target pairwise and edge cells are blank by construction.
- ADF and KPSS must be read jointly.
- Pairwise metrics use the strict all-feature non-NaN row filter.
- Bootstrap RNG is module-scoped and reproducible only under identical call sequence after import.
- The QC report creates candidates only; OOS validation, leakage review, costs, and live monitoring remain outside the panel.

## Style decision

Final form: compact field guide.

Structure:
- short opening contract
- read-order funnel
- one compact decision table
- metric sections only where interpretation changes action
- known limitations kept short and code-faithful

Reject:
- table-only manual
- long atmospheric prose
- code improvement proposals
- claims not witnessed in `qc.py` or `index.html`
