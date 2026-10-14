# Mill as the Cube's feature factory

Proposal, 2026-10-14. It brings Mill into the Market State Cube Explorer as a built-in feature factory, reached from a **Custom** entry in **Lines** (one or more lines) and **Columns** (one column at a time), with Limen's rule-based SFD as the representation underneath, and without moving any of the Cube's performance budgets.

It has five parts: what the three systems are today, with measurements; the foundation every approach needs; three experiences built on it; the performance contract; and the decisions and order I recommend.

## 1. What exists today

### Mill

- A Polars and Limen workbench over `GetData`. The registry exposes **214 producers**: 16 raw bar columns and 198 Limen factories (81 from `limen.features`, 117 from `limen.indicators`), registered by signature introspection. Parameters are typed from their defaults (bool, int, float, str). Nothing records a parameter's domain, a unit, a warmup, or whether a factory looks ahead.
- A spec is a flat list of `{factory, params}`, and every producer receives only the raw bars (`table.build_from_spec`). Features cannot compose: there is no z-score of an EMA, and no rule over feature outputs.
- QC is the strongest part: the content-addressed kernel memo, the subset-independent primitives behind `qc.sweep`, the grading ladders and the health, edge and selection verdicts.
- `/data` returns JSON rows with string datetimes, memoised per whole spec (`(bar, target, spec_hash)`). The result, QC and raw caches are unbounded.
- `GetData` fences every read to 2024-01-01 through 2026-05-15.
- Its interface has its own design system (dark only, IBM Plex, 13 px, yellow accent), unrelated to the Cube's.

Measured on an M1 Max (Python 3.13, Polars 1.41.2, Limen 3.15.2), real cached bars read through `GetData.from_parquet`, best of 5 to 7 runs:

| Work | 1h, 20,784 bars | 15m, 83,136 bars |
|---|---|---|
| load bars | 5.5 ms | 20.5 ms |
| view `gen4` (8 factories, 17 columns) | 15.5 ms | 41.4 ms |
| view `super` (9 factories, 18 columns) | 26.1 ms | 62.8 ms |
| rule layer: 3 Limen predicates and 1 compound | 0.34 ms | 0.65 ms |
| one feature as `/data` JSON | 0.62 MB | 2.49 MB |
| the same as f32 and gzip | 0.14 MB | 0.58 MB |
| `gen4` as `/data` JSON | 2.33 MB | 9.38 MB |
| the same as f32 and gzip | 0.60 MB | 2.38 MB |

Decoding those payloads in V8 (Node 23.10, same machine): `JSON.parse` of one 15m feature takes 8.3 ms; gunzip plus a typed-array view takes 2.7 ms. For `gen4` at 15m, parsing takes 39.9 ms and converting to typed arrays another 14.5 ms, against 17.5 ms of gunzip.

The conclusion is that compute is not the bottleneck. Transport and main-thread parsing are: today's JSON alone would put 8 to 55 ms on the Cube's main thread for one feature set, past its 20 ms frame budget.

### Limen's rule-based SFD

- `RuleBasedManifest.with_strategy(conditions, entry)`. Leaves are `threshold`, `relative`, `crossover`, `slope` and `sql_expr`, each optionally wrapped by `persistence_n` or `recency_n`. Compounds are `and`, `or` and `not` over leaf ids. `RuleBasedConfig` rejects duplicate ids, unknown operands and cycles.
- Leaves compile to Polars boolean expressions and materialise as columns named by their ids. `RuleBasedStrategy` folds the tree into per-bar positions and evaluates each split with backtests.
- Values can be templates (`'{rsi_threshold}'`) filled from `params()`, whose lists are the search space Furnace runs as rounds. The YAML form carries `indicators`, `features`, `strategy`, `reference_architecture`, walk-forward splits and acceptance gates (deflated Sharpe probability, PBO).

A rule-based SFD is therefore already a declarative, validated, parameterised graph of Polars expressions, with a text form and an evaluation path. That is what a feature factory needs as its internal representation.

### The Cube Explorer

- **Lines (P)** holds families, one hue each, every stroke 1.5 px. The pattern says how a line is known: solid observed, dashed held, dotted lead-in. Any number can be on; a 20% occlusion budget decides what is drawn; every row says why it is not drawn; Focus is temporary. Families already take user-added entries ("Add day"), and there is a "Your drawings" family.
- **Columns (B)** shows one measure per column, chosen in a stepped flow menu (Measure, then Timeframe for RSI). Oscillators are drawn on their own bars along the time axis; lengths use a registered Auto axis or a fixed domain.
- Indicators are computed in the page from `/cube/bars` (binary MSCB blocks in 4,096-bar chunks, read in the bridge's second slot and never ahead of the view's own reads), memoised per frame on typed arrays. Curves are drawn at one point per pixel column.
- Budgets (`tools/benchmark/navigation.v4.json`): p95 draw at most 10 ms, p95 frame at most 20 ms, p95 input-to-paint at most 50 ms, at most 1% of gesture intervals over 33.3 ms. The per-cell path is allocation-free (U49). The page has no workers.
- Honesty is structural: 14 typed results, so a missing value is never a zero; a known-at for every annotation; replay that never uses data after its edge; separate observation and model provenance.
- The address holds the view (`vis=2`, at most 8,192 characters, one codec table), alongside portable codes, saved Views and an immutable SHA-256-addressed investigation store. DR-05 adds no global shortcuts; strings live in `E.text`; System reference topics carry Purpose, Read, Use and Show; reduced motion stops all motion; touch targets are 44 px.
- `docs/visual-contract.md` section 11 is the checklist every new visual feature states before it merges: measurement, support, scale, channel and policy, role and states, generated legend, persistence and tests.

### Seven tensions the design resolves

1. **Two sources for one market.** Mill reads Limen's tdw Arrow bar store. The Cube reads the Origo cube and says every number it shows is the cube's answer.
2. **Two windows.** Mill stops at 2026-05-15. The Cube runs from 2021-01-01 to now.
3. **Two compute worlds.** 198 Limen factories run in Python. The page runs five indicator families in JavaScript.
4. **Two transports.** Mill sends JSON rows. The Cube reads chunked binary blocks.
5. **Missing semantics.** Mill has no known-at, warmup, causality, unit or typed non-value. The Cube needs all five for anything it draws.
6. **No composition.** Mill's features cannot consume each other, and the useful custom lines are compositions.
7. **Two design systems.**

Sections 2.1 to 2.4 resolve 3 to 6, and the decisions in section 5 resolve 1, 2 and 7.

## 2. The foundation every approach needs

### 2.1 The feature descriptor

A custom feature is stored as a **feature descriptor**: a Limen rule-based SFD fragment plus everything the Cube's section 11 checklist asks a contributor to state. A compiler fills those fields in, never a person.

```yaml
schema: mill-feature@1
source: {binding: cube-bars, timeframe: 1h}        # or limen-bars
graph:
  - {id: z,   op: limen.features.rolling_zscore, args: {column: close}, params: {window: 96}}
  - {id: hot, op: threshold, column: z, operator: ">", value: 1.5, persistence_n: 2}
output: hot
params: {}                 # SFD params(): a list becomes steps on the row and rounds in Furnace
# Written by the compiler:
unit: boolean              # so the channel is events, or a column
axis: none                 # fixed{lo, hi} | auto | auto-signed
warmup_bars: 96
memory: finite(97)         # finite(k) | decaying(K) | path
causal: {method: prefix-invariance, checkpoints: 8}
provenance: {mill: 0.3.0, limen: 3.15.2, factory_sources: {rolling_zscore: sha256}}
```

Its id is the SHA-256 of its canonical JSON, shown as 96 bits in base64url (the Cube's mapping-id convention, DR-03). Descriptors are immutable: editing one makes a new one, and saved views keep the old.

### 2.2 Certification

Nothing reaches the canvas uncertified.

- **Causality**, by prefix invariance on real bars: computing on the first k bars must give the same first k values as computing on all of them, at 8 checkpoints. A centred window, a full-sample normalisation or a negative shift fails, and the factory is refused with that reason. Each factory is certified once per Limen version, and each descriptor once more as a whole graph at compile time (about 8 recomputations, roughly 0.2 s for a three-factory graph on 1h by the measurements above), cached by id.
- **Warmup** is measured as the first finite value, and checked against the formula where one is known.
- **Memory class** decides how live tails update. A finite window recomputes the last (warmup + new) bars exactly. A decaying recursion such as an EMA or Wilder smoothing recomputes from a 4,096-bar seed, the bound the Cube already uses for its own EMAs. A path-dependent node recomputes from a stored checkpoint.
- **Units** come from a signature table: close is USDT; a ratio of two values in one unit is unitless; RSI is bounded 0 to 100; a z-score is signed and unitless; a predicate is boolean. The unit decides the channel. Only a USDT value can be a line, a boolean can be events, and any numeric value can be a column. A factory with no recorded unit is offered only as a column on an Auto axis.
- **Satisfiability** checks the columns and the timeframe a descriptor needs against its source. Cube bars carry open, high, low, close, USDT volume, taker-buy USDT volume, BTC volume and trades, at every grid timeframe (3.75m, 15m, 30m, 1h, 2h, 4h, 8h, and the days and weeks built from 8-hour bars). They do not carry Limen's `mean`, `std`, maker trade ratio or liquidity OHLC. Limen's bar store carries all of those, but only as 15m, 30m, 1h, 2h and 4h time bars (`bars.TIME_BAR_ARROW_SERIES`; `GetData.get_time_bars` raises on any other size). An unmet column or timeframe makes the feature a typed `unsupported` with that reason, never an approximation: a server feature at 3.75m, 1D or 1W is refused at compile time, and the flow never offers it.

### 2.3 Where each value is computed

- **On the page, for portable features.** A small operator set runs on typed arrays in the page, on the Cube's own bars: what `E.indicators` already has (SMA, EMA, Bollinger, RSI, MACD) plus about fifteen more (shift, difference, returns, ratio, absolute value, log, clip, rolling sum, mean, standard deviation, minimum and maximum, z-score, true range and Wilder ATR, range-based volatility, rolling VWAP), with the four predicate leaves other than `sql_expr`, persistence, recency and the compounds. A descriptor made only of these never leaves the page. It needs no new read, because the bars are the chunks the Cube already holds for its averages; it is exact in replay; and it can be scrubbed within a frame. Golden vectors from Mill's Polars and Limen implementation are the oracle, in the Cube's existing golden discipline. `sql_expr` is not offered, because the Cube never executes text from a payload.
- **On the server, for everything else.** Mill computes on the host, on Limen's bar store, through a Polars plan with a per-node content-addressed memo. An edit recomputes the changed node and the nodes that depend on it. Predicates cost under a millisecond, so threshold scrubs recompute freely.

Measured on typed arrays in V8: EMA, rolling z-score, Wilder RSI and a three-condition rule each take at most 0.63 ms over 83,136 bars, and 2.9 to 9.8 ms over 1,048,576 bars.

### 2.4 Transport

- One new read, `GET /cube/feature?id&n&b0&b1&pack`, served through the bridge (same origin, same portal login) and answered by Mill. It returns an **MSCF** block shaped like MSCB: a 32-byte header; values in f64, or f32 where the unit allows; a u8 state per bar (value, warmup, outside support, missing input); and u32 columns. It is gzipped, cut into 4,096-bar chunks aligned to the history start, read latest first, and extended by tails as the pack moves on.
- It is the page's lowest-priority read, never ahead of the view's own reads, single-flight per (id, n, chunk, pack), and it never takes one of the cube service's two query slots, because Mill reads its own bar store.
- Against today's `/data`, one 15m feature drops from 2.49 MB to 0.58 MB, and main-thread decoding from 8.3 ms to 2.7 ms, or none with `DecompressionStream`.

### 2.5 Drawing, with no new primitives

- A custom line goes through `curveVertices` (one point per pixel column), the 1.5 px reference stroke and its casing, the occlusion planner and the tags. It costs what a moving average costs.
- A custom column goes through the pane's painter and the axis registry: a fixed domain when the unit is bounded, a symmetric Auto axis for signed values, an Auto axis otherwise, with the existing non-value glyphs.
- A boolean feature becomes events: glyphs at bar ends, as the crosses are drawn, or intervals in the event strip, as the squeezes are.
- No new hue. Screening the six existing reference and signed hues with an OKLab colour-vision simulation already gives a worst pair of ΔE 4.5 in the light theme (negative clay against the POC gold, deutan), which the Cube separates by label, glyph and place. Every candidate seventh hue I screened (two teals, a magenta and a green, per theme) fell below the target of 8 against at least one existing hue, the best at 7.2 in light and 5.9 in dark. So the **Your features** family draws in a neutral carrier, ink or muted as the co-occurrence matrix decides, the way Historical comparison is neutral; each line is told apart by its tag, an `ƒ` mark in the tag and the row, and its name. An authored RGB per feature stays possible on the drawings' terms: object identity only, with its name and exact value beside every swatch.

### 2.6 Semantics the factory inherits

- A value is known at its bar's end; on a forming bar it is a candidate, so far. One row joins `E.readout.events.TABLE`.
- Warmup uses the existing "warming up" status and names the first full window.
- In replay, page kernels run on replay-clipped bars and server series are cut at the edge.
- Pending, failed, unsupported (with its reason), outside support and hidden keep their keyed marks.
- Inspect's Measurement details shows the descriptor as Limen YAML, its source, the Limen version, the certificate and the warmup.

### 2.7 Interaction patterns the factory keeps

- No global shortcut is added (DR-05). The factory is reached from Lines (P) and Columns (B), and its flow uses the keys every flow menu already uses.
- Every string goes into `E.text`. A System reference topic, **Your features**, carries Purpose, Read and Use, with Show targets for the family, the pane and the band.
- Controls are 44 px targets on touch, and on a phone the flow opens in the controls sheet.
- The factory adds no motion of its own. A preview appears and disappears without a transition, and any feature drawn on the chart follows the existing 170 ms level morph, which reduced motion already stops.

### 2.8 Identity and persistence

- Descriptors live in an immutable content-addressed store, the investigation store's pattern: SHA-256 bytes in SQLite on a volume.
- Addresses carry 16-character ids (`lines=…,f.<id>`, `pane=f.<id>`), well inside 8,192 characters. Portable codes carry descriptors inline. Saved Views carry ids.
- An unknown id is a typed `unsupported`, with a route to import the code that defines it.

### 2.9 Agents

`cubeAgent.factory.{catalog, compile, preview, place}` and the matching MCP tools use the same compiler and certificates as the human steps, so an agent's feature is admitted exactly as a person's is.

### 2.10 What changes in Mill

1. A descriptor compiler (graph, units, channel, warmup, memory class) and a certifier (causality, satisfiability).
2. Composition: any node can consume any other node.
3. Rule nodes from `limen.sfd.rule_based`.
4. A per-node memo, and caches bounded by bytes (the follow-up the concurrency audit parked).
5. MSCF chunks and live tails in place of JSON rows.
6. Catalog v2: parameter domains, units, warmup formulas, a portable form where one exists, and tags.
7. QC per descriptor, a ranked pool per timeframe for discovery, and YAML hand-off to Furnace.
8. Its interface moves onto the Cube's design system, or Mill becomes API-only with the Cube as its interface.

## 3. Three approaches

### A. In-menu factory: Custom in Lines and Columns (your approach)

**The experience.**

- **Lines (P)** gains a family, **Your features**, after Your drawings. Each row is a custom line with the standard anatomy: toggle, name and tag, value at the right edge, Focus, and a status that says why it is not drawn. The family ends with **New line…**, which turns the popover into the factory's flow, or the sheet on a phone. Any number can be on; the 20% budget decides what draws.
- **Columns (B)**: the Measure step lists the built-in measures, then **Your columns**, then **New custom column…**. One custom column shows at a time, because the pane shows one measure.
- **The flow** reuses `buildFlowMenu`: numbered steps, Back, Continue, Done, right-click to apply and finish, and steps omitted where they do not apply.
  1. **Start**: a recipe (Average, Band, Oscillator, Z-score, Ratio, Rule), any of the 198 Limen factories, searchable and grouped by your Mill tags, or a member of a Mill view such as `gen4`.
  2. **Build**: the graph as a sentence of chips, for example "z-score of close over [96] bars, above [1.5] for [2] bars". A chip opens its parameter; "+ condition" adds a Limen leaf (above, below, crosses, rising) with persistence or recency; "and", "or" and "not" join conditions. A unit chip updates as you build: "USDT, can be a line", "signed, a column on its own axis", "true or false, events".
  3. **Timeframe**, as RSI's Timeframe step works, offering only what the descriptor's source has: any grid timeframe from 3.75m to 1W for a portable feature computed on the Cube's bars; 15m, 30m, 1h, 2h or 4h for a feature that needs Mill. Any other choice is shown disabled with its reason ("Limen's bar store has no 1D series; build it from portable operators to use 1D").
  4. **Check**: Mill's QC in the Cube's language. Health, Stationarity, Redundancy against what is already on, and Edge once a target is chosen, as filled and hollow dots in the state ink with words beside them; the warmup, the known-at, the source, the window and the causal certificate.
  5. **Place**: a generated name and tag, and the channel preselected from the unit. A channel the unit cannot take is disabled with its reason: "A z-score has no USDT value to draw on the price axis."
- While the flow is open the candidate draws as a temporary mark, as Focus is temporary, with a "Preview, not saved" chip. It never enters the address. Numbers step with the arrow keys in a focused field and scrub with a drag. A portable descriptor redraws within the frame; a server one shows "Updating" until its read settles.
- A parameter can keep a list (SFD `params()`). The row then steps through its values in place: "EMA ‹34› · 4h".
- Edit reopens Build. Saving makes a new descriptor, and saved views keep the one they had.

**Why it fits.** It adds no layout and no shortcut, and reuses the flow menu, the families, the pane, Inspect, the address and the reference.

**Where it stops.** A popover suits graphs of up to about six nodes. It only helps when you already know what to build. Its QC is a summary, not a workbench.

**Performance.** One custom column costs what the RSI pane costs, and N custom lines cost what N averages cost. Both are existing heavy cases in the benchmark.

### B. Find features: discovery from what you see

**The experience.** It reverses the direction. You mark market states with tools the Cube already has (a Select rectangle, a set of columns, a Rallies result, Cases, Compare captures), and the factory finds which features describe them and tests whether that is more than chance. The verb **Find features…** appears in the selection card, the Rallies results and the Query tab.

1. **Target**: inside the selection against the rest of the view; the four hours before each rally starts; the bar after each case's anchor. Label counts are shown, and below 30 events intervals are withheld, as Continuations already does.
2. **Pool**: your Mill views, the catalog at default parameters, or parameter grids, with the size stated ("233 candidates on 1h").
3. **Search**, in two stages, with results streaming in.
4. **Results**: a ranked shortlist. Each card gives the separation (AUC with a block-bootstrap interval, ranked by its lower bound), regime agreement over four quarters, redundancy with what is already on, and where it stands against chance. The selected card draws as a ghost column, and the arrow keys step through the cards.
5. **Rule**: "Make it a rule" fits a threshold or a crossing with persistence as a Limen leaf, draws its hits as events, and places it as A does, or emits a RuleBasedManifest for a walk-forward Furnace run with acceptance gates.

**The method.**

- The pool is computed once per timeframe and data revision, and ranked once. Measured on 1h: 233 columns from every default-parameter factory in 4.4 s, ranked in 65 ms.
- Stage one is the AUC of every column from its ranks, a single matrix-vector product. Measured: **0.37 ms** for all 233 columns over 20,784 bars.
- Chance is measured. Circularly shifting the labels keeps their autocorrelation and breaks their alignment with the features; 200 shifts cost about 75 ms at the measured per-pass cost and give the null distribution of the best AUC in the whole pool. A card reads "beyond chance" only above that null's 95th percentile, which controls the family-wise error across every candidate tested.
- Stage two runs Mill's full QC kernels (block bootstrap, mutual information, regime) on the top dozen, one per correlation cluster, memoised and streamed.
- Guards: only certified-causal features; lead windows end strictly before the event; Mill's research window bounds every label; in replay, discovery uses only data before the edge.

**Why it is the leap.** The Cube is where states are seen and Mill is where features are tested; neither can tell you what describes the state in front of you. B uses Mill's strongest machinery, makes Limen's rule language the explanation, and ends in something that can be falsified, a Furnace manifest, rather than a line that looks right.

**Risk.** Data snooping is its central risk. The permutation null, the window, the counts and the Furnace hand-off contain it. B also spends the shared host's compute, so it runs on one queued worker.

### C. Feature tracks: Mill's table on the Cube's time axis

**The experience.** Mill's grid has bars as rows and features as columns. Turned 90 degrees, time becomes the Cube's x-axis and features become thin tracks under the price pane: many features at once, each read in market context.

- A **Features** band sits between the price pane and the Columns pane. Each track is a 16 px strip with one value per bar, painted by the Cube's scale machinery: signed values on the positive and negative arms about the midpoint, unsigned values on the slate ramp, bounded values on a fixed domain, booleans as occupancy. Each track has an Explore scale per context, held by Comparison lock.
- A plate at the left names the track, its timeframe and its QC dots. A tag at the right gives the latest value, as lines' tags do.
- The crosshair reads every track at once, so the hover card is Mill's row for that bar. Inspect gains a Features surface: up and down move between tracks, left and right between bars.
- A track's menu promotes it: **Show in Columns** (full height, with its axis), **Show as line** (USDT only), **Show as events** (booleans).
- **QC** swaps the strips for Mill's QC matrix over the visible range, or over the selection, which takes the role of Mill's date filter.
- Mill's views load as sets: `gen4` adds eight tracks. A pinned Limen target track enables the Edge rows.
- **+ Track** opens A's flow in the band, where the chip editor has room.

**Why it is the eventual home.** It moves all of Mill into the Cube while keeping one canvas, one time axis, one scale system and one Inspect. It shows how your features behaved around the move you are looking at, which neither tool can show today.

**Risk.** It is the largest change: a new region, new contract rows, colour-matrix contexts, benchmark cases and a phone layout, where the tracks fold to one line that opens a sheet.

**Performance.** Each track renders to an offscreen bitmap per (track, level, span) on the settled schedule. Frames blit it, translated during gestures, so a gesture never touches values. Budgets: at most 0.1 ms per track per frame, 0.5 ms per settled re-render, and 16 tracks visible.

### Side by side

| | A. In-menu factory | B. Find features | C. Feature tracks |
|---|---|---|---|
| Starts from | a feature you have in mind | a state you can see | a set you want to study |
| Adds to the Cube | flow steps in two menus | one verb and a results flow | one band under the chart |
| At once | lines: many; column: one | one candidate previewed | many tracks; one promoted |
| Mill's role | compiler, certifier, QC summary | adds the ranked pool, permutation null, rule fit and Furnace hand-off | adds QC over a range, sets and targets |
| Limen SFD's role | the recipe | the explanation, then the experiment | the sets and targets |
| Main risk | none beyond the foundation | data snooping | layout and per-frame cost |
| Cost on the page | same as averages and RSI | negligible | one blit per track per frame |

## 4. The performance contract

**Rules on the page.**

1. Off costs nothing: no reads, no compute and no draw hook while no custom feature is on.
2. Nothing is computed in a paint. Kernels run on the settled schedule, memoised per descriptor and frame, and time-sliced in 4 ms slices beyond 100,000 bars.
3. Feature reads have the lowest priority, never go ahead of the view's, are single-flight, and fill byte-bounded caches.
4. Drawing reuses the existing primitives, so its cost scales with pixels, not with history.
5. Decoding happens off the main thread, into typed-array views without copies.

**Gates.** The Cube's own method: paired A/B runs against the preceding main with A/A noise floors, and new heavy cases (one custom line computed in the page, one from the server, five custom lines, one custom column, eight and sixteen tracks) that must hold p95 draw at 10 ms, frame at 20 ms and input-to-paint at 50 ms. The fake cube's request log must show the view's first reads unchanged with custom features on.

**Mill's targets.** Compile a descriptor of certified nodes in under 5 ms; serve a cached chunk in under 20 ms; recompute only the node that changed; keep every cache inside a byte budget.

## 5. Decisions I need from you

1. **The source split.** Portable features compute on the Cube's bars in the page; everything else comes from Mill on Limen's bar store, labelled as such. The alternative, cube bars for everything, needs a new `GetData` path and server reads that compete with the page. I recommend the split, with a one-time parity report of Limen's 1h bars against the cube's n = 6 bars over the window.
2. **The 2026-05-15 window.** If it protects an out-of-sample period, I propose that QC, discovery and Furnace never cross it; that drawing crosses it only for features computed in the page, which the Cube already shows over all history; and that the Check step records when a feature was tuned while later data was on screen.
3. **Mill's own interface.** Retire it into the Cube, or rebuild it on the Cube's design system.
4. **A new read kind.** `/cube/feature` is deliberate. The Cube's DR-17 forbade new read kinds for its calibration work.
5. **The custom family's carrier.** I recommend no new hue (section 2.5), with the co-occurrence matrix choosing between ink and muted.

## 6. Recommendation

Build the foundation once, then ship in this order:

1. **A**, because it carries the descriptor, certification, page kernels, MSCF, persistence and the contract rows into the Cube, and it is the experience you described.
2. **B**, because it is the step neither system can take alone, it is mostly server work, and its results land in A's surfaces.
3. **C**, as the long-run home for systematic research once A and B have proved the descriptor and the performance contract.

## Appendix: method of the measurements

- Mill: `GetData().from_parquet` on the cached `1h_time_bars` (20,784 rows) and `15m_time_bars` (83,136 rows) for 2024-01-01 to 2026-05-15; `table.build_from_spec` with the built-in views; Limen `build_predicate` for the rule layer; `/data` JSON reproduced with `serve._serialize`'s expressions; binary as int64 milliseconds plus one array per column, gzip level 6.
- V8: Node 23.10 on the same payloads; `JSON.parse` with conversion to `Float64Array`, against `zlib.gunzipSync` with typed-array views; kernels on a seeded random walk (timing only).
- Pool: every registry factory whose parameters all have defaults, applied to the 1h bars (233 numeric columns), ranked with Polars, AUC as a rank-sum matrix-vector product against one binary label (timing only).
- The QC values quoted in the companion page come from `qc.report` and `grading.grade_report` on the 1h bars with target `next_bar_up`.
- Hue screen: OKLab ΔE ×100 under the Machado (2009) severity-1.0 protan and deutan simulations and unsimulated vision, pair by pair, against the Cube's light and dark values of `--ol-line-poc`, `--ol-line-level`, `--ol-line-average`, `--ol-line-vwap`, `--ol-positive` and `--ol-negative`. It is an engineering screen in a different metric from the Cube's CIEDE2000 matrix, which stays the authority.
