# Design guidelines

Minimal spec for lab-style UIs. Copy the full `:root` token block from the reference app and the role classes below. Three layers — **color**, **typography**, **rhythm** — one system.

**Global rules**

1. Literals (`#…`, `px`, `ms`) live **only in `:root` atoms**.
2. Rules use **semantic tokens** only — never `--c-*` or bare palette hex.
3. One hue, one meaning. One density. No magic numbers in component rules.

---

## Color

### Palette (literals — `:root` only)

```css
--c-near-black: #121212;
--c-warm-black: #231F20;
--c-dark-gray:  #444444;
--c-mid-gray:   #808080;
--c-light-gray: #D3D3D3;
--c-pale-gray:  #F3F3F3;
--c-off-white:  #F8F8F8;
--c-pale-pink:  #FCE2EB;
--c-pink:       #EAA3C8;
--c-magenta:    #DC65A6;
--c-coral:      #F16068;
--c-lavender:   #BCABD3;
--c-yellow:     #DDD941;
```

Neutrals + six hues. **Never reference `--c-*` in component rules.**

### Surfaces

```css
--bg:        var(--c-near-black);
--highlight: var(--c-warm-black);  /* subtle lift: row/col hover, in-cell bars */
```

- **`--bg`** — page, modals, sticky headers, inputs
- **`--highlight`** — the only lift surface; warm black, not grey. Row/column hover and in-cell bar fills — **not** borders or selection chrome

### Borders (neutral only)

```css
--border:        var(--c-dark-gray);
--border-subtle: var(--c-warm-black);
--border-strong: var(--c-mid-gray);
```

No hue on borders.

### Text (neutral ramp)

```css
--fg:   var(--c-off-white);   /* primary body, values, active labels */
--fg-1: var(--c-pale-gray);  /* de-emphasis within prose */
--fg-2: var(--c-light-gray); /* secondary chrome: headers, icons, placeholders */
--fg-3: var(--c-mid-gray);   /* disabled, nulls, chevrons, separators */
```

Default: `color: var(--fg)` on `html, body`. Icons/controls idle at `--fg-2`, hover to `--fg`.

### Semantics — one hue, one meaning

```css
--accent: var(--c-yellow);   /* focus, active toggles, copy confirm, selected presets */
--pos:    var(--c-lavender); /* positive numeric values only */
--neg:    var(--c-coral);    /* negative numeric values only */
--danger: var(--c-magenta);  /* destructive actions */
```

| Token | Use | Do not use for |
|---|---|---|
| `--accent` | Active icon, radio checked, copy success, date preset active | Arbitrary emphasis, links, headers |
| `--pos` / `--neg` | Signed numbers in grids and QC | Labels, buttons, grades |
| `--danger` | Destructive controls | General warnings, negative numbers |

### Syntax highlighting (neutral only)

```css
--syntax-fg:    var(--fg);
--syntax-base:  var(--fg-2);
--syntax-dim:   var(--fg-3);
--syntax-muted: var(--fg-1);
```

YAML/JSON/config blocks use **grey ramp only** — no lavender/coral/yellow/pink in code.

### QC grades (dot system)

```css
--grade-f: var(--c-pink);
```

Grades are **white dot counts**, not colored letters:

| Grade | Dots |
|---|---|
| A | 4 × `--fg` |
| B | 3 × `--fg` |
| D | 2 × `--fg` |
| F | 1 × `--grade-f` |

Use `●` at `--t-icon`, same character as binary columns in the main grid.

### Color rules

1. **Palette literals live only in `:root`.** No `#…` or `rgb()`/`rgba()` anywhere else.
2. **Rules use semantic tokens only.**
3. **One hue, one job.** Yellow = interaction; lavender/coral = numeric sign; magenta = danger; pink = QC fail only.
4. **Surfaces stay dark.** `--highlight` is the only elevation — no lighter modal fills, no coloured shadows.
5. **Syntax stays neutral.**
6. **Sign color is for numbers.** Pos/neg on data cells only.
7. **Active state = accent.** Toggle `.active`, copy `.copied`, focused presets — all `--accent`.
8. **Null/missing = `--fg-3`.**

---

## Typography

### Atoms

```css
--font-sans: "IBM Plex Sans", sans-serif;
--font-mono: "IBM Plex Mono", monospace;
--t: 13px;
--t-icon: 11px;
--leading-tight: 1;
--leading-body: 1.4;
--track-label: 0.08em;
```

Load **400 and 500 only**:

```
IBM+Plex+Mono:wght@400;500
IBM+Plex+Sans:wght@400;500
```

### Composites (only these in `font:` declarations)

```css
--text-body:    400 var(--t)/var(--leading-body) var(--font-sans);
--text-ui:      400 var(--t)/var(--leading-tight) var(--font-sans);
--text-label:   500 var(--t)/var(--leading-tight) var(--font-sans);
--text-data:    400 var(--t)/var(--leading-tight) var(--font-mono);
--text-data-em: 500 var(--t)/var(--leading-tight) var(--font-mono);
--text-code:    400 var(--t)/var(--leading-body) var(--font-mono);
--text-icon:    500 var(--t-icon)/var(--leading-tight) var(--font-sans);
```

No bare `font-weight`, `font-size`, or `line-height` in rules.

### Roles

| Class / role | Token | Family rule |
|---|---|---|
| `.t-body` | `--text-body` | sans — prose, status, tooltips |
| `.t-ui` | `--text-ui` | sans — controls, selects, chrome |
| `.t-label` | `--text-label` + uppercase + `--track-label` | sans — section headers, column headers, actions |
| `.t-data` | `--text-data` + `tabular-nums` | mono — grid values, dates, numeric inputs, stats |
| `.t-code` | `--text-code` + `tabular-nums slashed-zero` | mono — YAML/JSON/config blocks |
| `.t-icon` | `--text-icon` | sans — ×, ▲, ▼, ▶ glyphs |

Emphasis inside data (e.g. bold counts): `--text-data-em`, still mono.

### Typography rules

1. **One size for UI/data (13px).** Icons/glyphs only at 11px (`--t-icon`).
2. **Only data is mono.** Everything else is sans — including selects, palette items, metric names, placeholders.
3. **Two line heights:** `1` (controls, data cells, icons) and `1.4` (body, prose, code blocks).
4. **Labels are one treatment:** 500 sans, 13px, uppercase, `0.08em` tracking — always via `.t-label` or the shared label selector block.
5. **`font-variant-numeric`:** `tabular-nums` on all mono data/code; `slashed-zero` on code blocks only.
6. **No literal typography** in rules — only composite tokens and role classes.

---

## Rhythm & spacing

### Atoms (literals — `:root` only)

```css
--s-1: 4px;
--s-2: 8px;
--s-3: 12px;
--s-4: 16px;
--s-5: 24px;
--s-6: 32px;

--row-h:             36px;
--col-dt:            184px;
--col-metric:        240px;
--col-feature:       132px;
--col-row-group-min: 220px;
--field-w-sm:        56px;
--field-w-md:        64px;
--tip-max-w:         360px;
--radio-size:        14px;
--radio-dot:         6px;
--scrollbar-w:       10px;
--radius:            2px;
--border-w:          1px;
--border-w-accent:   2px;
--dur:               80ms;
--ease:              ease;
--opacity-drag:      0.4;
--opacity-muted:     0.5;
--tab-size:          2;
```

4px base scale. All other rhythm values are derived or named recipes — no bare `px` in rules.

### Derived

```css
--transition:      var(--dur) var(--ease);
--control-h:       var(--row-h);
--icon-btn-w:      var(--control-h);
--caret-w:         var(--s-2);
--chevron-w:       var(--s-2);
--radio-inset:     calc((var(--radio-size) - var(--radio-dot)) / 2);

--cell-pad-x:      var(--s-4);
--cell-pad-label:  var(--s-5);
--chrome-pad-y:    var(--s-4);
--chrome-pad-x:    var(--s-5);
--pane-pad:        var(--s-5);

--qc-grade-dots-max: 4;
--qc-grade-dot-size: var(--t-icon);
--qc-grade-dot-gap:  var(--s-2);
--qc-grade-track: calc(
  var(--qc-grade-dots-max) * var(--qc-grade-dot-size)
  + (var(--qc-grade-dots-max) - 1) * var(--qc-grade-dot-gap)
);
--qc-grade-pad: calc(var(--cell-pad-x) + var(--qc-grade-track));

--z-modal:            10;
--z-popover:          50;
--z-qc-sticky-col:    1;
--z-qc-sticky-head:   2;
--z-qc-sticky-corner: 3;
--z-tip:              100;
```

### Recipes

| Recipe | Token(s) | Use |
|---|---|---|
| **Chrome bar** | `--chrome-pad-y` × `--chrome-pad-x` | Header, modal head |
| **Pane body** | `--pane-pad` | Features builder, YML, row modal |
| **Data row** | `--row-h`, `0` × `--cell-pad-x` | Main grid cells, QC cells |
| **Label gutter** | `--cell-pad-label` left | First/sticky columns (datetime, metric names) |
| **Compact control** | `--s-1` × `--s-2` padding | Modal icon buttons, small inputs |
| **Status / empty** | `--s-6` vertical, `--chrome-pad-x` horizontal | QC loading, empty states |

### Gap ladder

| Token | Use |
|---|---|
| `--s-6` | Major clusters (header groups, row-group grid) |
| `--pane-pad` / `--s-5` | Modal interior layout |
| `--s-4` | Control groups |
| `--s-3` | Related items (icon toolbar, chip head) |
| `--s-2` | Tight groups (date filter, palette wrap, params) |
| `--s-1` | Minimal (modal actions, vertical option list, active feature stack) |
| `0` | Flush pairs (label + select) |

### Layout primitives

- **App shell** — column flex; header fixed; content `flex: 1; min-height: 0`.
- **Main table** — grid: `var(--col-dt)` + flexible columns; rows at `--row-h`.
- **QC table** — same row height and cell padding as main; sticky metric column `--col-metric`; feature columns `--col-feature`.
- **Modals** — full overlay at `--z-modal`; panel inset `--s-6`.
- **Split panes** — 50/50 (features) or `1fr 2fr 1fr` (YML).

### Borders & motion

- Standard border: `var(--border-w) solid var(--border)`.
- Accent emphasis: `var(--border-w-accent)` (tooltip rail, drag outline).
- Corner radius: `var(--radius)` everywhere except full circles (radios).
- Transitions: `var(--transition)` on color, border, background, opacity.

### Z-index stack

Low → high: QC sticky col → QC sticky head → QC sticky corner → modal → popover → tooltip.

### Rhythm rules

1. **One density.** All data surfaces use `--row-h` and `--cell-pad-x` / `--cell-pad-label`. QC follows main.
2. **Controls match rows.** `--control-h` equals `--row-h`; icon buttons are square at `--icon-btn-w`.
3. **No magic numbers in rules.** Literals live only in `:root` atoms; derived values use `calc()`.
4. **Spacing uses `--s-*` or named recipes** — not ad hoc `px`.
5. **QC grade gutter is derived.** `--qc-grade-pad` from dot count × size × gap.
6. **JS reads CSS.** Virtualized row height uses `getComputedStyle` on `--row-h`.

---

## Defaults

```css
html, body {
  margin: 0;
  padding: 0;
  background: var(--bg);
  color: var(--fg);
  font: var(--text-body);
}

#app {
  display: flex;
  flex-direction: column;
  height: 100vh;
}
```

## Sister-app checklist

- [ ] Full `:root` block copied (palette → semantics → syntax → type → rhythm)
- [ ] Six typography role classes (`.t-body` … `.t-icon`)
- [ ] Zero hex/rgba outside `:root`
- [ ] No direct `--c-*` in rules
- [ ] Hover/selection uses `--highlight`, not `--border`
- [ ] Only data cells get `--pos` / `--neg`
- [ ] Code/config uses `--syntax-*` only
- [ ] QC and main grid share `--row-h` and cell padding
- [ ] No bare `px` / `font-size` / `line-height` in component rules
