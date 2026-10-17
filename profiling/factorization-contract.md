# Factorization contract

Reference: merged Mill `847cd481e82b1000496db082e52acb4d9794c207`, with the
same installed Limen, Polars, NumPy, SciPy, sklearn and Numba versions, raw source
bytes, complete closed history, thread settings and workload parameters.

Supported inputs remain the registry's existing factories and parameters, raw
bar schema, saved views, targets, QC selection/grouping and discovery context,
event windows and options. Empty/short series, ties, constants, nulls/NaNs,
infinities, distinct factory warmups, source replacement and concurrent callers
retain their existing results or errors. No new feature, date gate or sampling
rule is introduced.

Feature schemas, names, ordering, values and missingness must match exactly.
Parsed API outputs, QC reports and discovery decisions must match exactly;
elapsed-time fields are excluded. Bootstrap samples, seeds, pair direction and
downsampling positions are unchanged. No numerical tolerance is granted to
published outputs. Differential tests use exact equality, including the FFT
null and the marginal neighbour counts. Any future tolerance or approximation
requires a separate, explicit utility contract.

Dependency graph: source snapshot -> producers -> feature/target frame ->
presentation or QC; source snapshot -> catalog/view pool -> context ranks ->
target-specific screen/null -> leader bootstrap/rules/presentation. Native
producer expressions can share a node only after all input column dependencies
and literal parameters match. Opaque kernels keep their original eager path.
QC prerequisites can share only within an exact pairwise retained-row cohort,
sampling cap, column and MI left/right RNG role. Its workspace is request-local
and bounded; whole-kernel reuse retains existing content keys. Discovery null
counts depend on target labels, shifts and the complete validity pattern, while
rank sums remain column-specific. Only one pattern's counts and a bounded
column tile are live at once.

Native plans apply to view frames and bounded catalog/checkpoint batches. One
factory failure retains the original per-factory skip behavior. Bootstrap rank
means are exactly `(n+1)/2`; at the existing 20,000-row cap all rank products and
sums are exactly representable half/quarter integers. Reducing those directly
removes sampled/centered arrays without changing floating-point results.
Scratch has at most eight lanes; no allocation occurs inside the draw loop.

Measurement gates: repeated matched full-history runs against the reference,
including cold setup, feature building, target joins, output serialization,
research execution and steady state; separate ablations for native expression
sharing, QC preparation sharing and discovery validity factoring. Record source
and result digests, versions, threads, host load, wall/CPU time and peak RSS.
Re-profile after each accepted change; report regressions and irreducible costs.
Measurements are local observations, not projected production guarantees.
