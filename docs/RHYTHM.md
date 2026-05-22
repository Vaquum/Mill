# Rhythm & spacing

Minimal spec for lab-style UIs. Copy `:root` rhythm atoms/derived tokens and the layout recipes below.

## Atoms (literals — `:root` only)

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

## Derived

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

## Recipes

| Recipe | Token(s) | Use |
|---|---|---|
| **Chrome bar** | `--chrome-pad-y` × `--chrome-pad-x` | Header, modal head |
| **Pane body** | `--pane-pad` | Features builder, YML, row modal |
| **Data row** | `--row-h`, `0` × `--cell-pad-x` | Main grid cells, QC cells |
| **Label gutter** | `--cell-pad-label` left | First/sticky columns (datetime, metric names) |
| **Compact control** | `--s-1` × `--s-2` padding | Modal icon buttons, small inputs |
| **Status / empty** | `--s-6` vertical, `--chrome-pad-x` horizontal | QC loading, empty states |

## Gap ladder

| Token | Use |
|---|---|
| `--s-6` | Major clusters (header groups, row-group grid) |
| `--pane-pad` / `--s-5` | Modal interior layout |
| `--s-4` | Control groups |
| `--s-3` | Related items (icon toolbar, chip head) |
| `--s-2` | Tight groups (date filter, palette wrap, params) |
| `--s-1` | Minimal (modal actions, vertical option list, active feature stack) |
| `0` | Flush pairs (label + select) |

## Layout primitives

- **App shell** — column flex; header fixed; content `flex: 1; min-height: 0`.
- **Main table** — grid: `var(--col-dt)` + flexible columns; rows at `--row-h`.
- **QC table** — same row height and cell padding as main; sticky metric column `--col-metric`; feature columns `--col-feature`.
- **Modals** — full overlay at `--z-modal`; panel inset `--s-6`.
- **Split panes** — 50/50 (features) or `1fr 2fr 1fr` (YML).

## Borders & motion

- Standard border: `var(--border-w) solid var(--border)`.
- Accent emphasis: `var(--border-w-accent)` (tooltip rail, drag outline).
- Corner radius: `var(--radius)` everywhere except full circles (radios).
- Transitions: `var(--transition)` on color, border, background, opacity.

## Z-index stack

Low → high: QC sticky col → QC sticky head → QC sticky corner → modal → popover → tooltip.

## Rules

1. **One density.** All data surfaces use `--row-h` and `--cell-pad-x` / `--cell-pad-label`. QC follows main — no tighter table padding.
2. **Controls match rows.** `--control-h` equals `--row-h`; icon buttons are square at `--icon-btn-w`.
3. **No magic numbers in rules.** Literals live only in `:root` atoms; derived values use `calc()` from tokens.
4. **Spacing uses `--s-*` or named recipes** — not ad hoc `px` in padding/gap/margin.
5. **QC grade gutter is derived.** `--qc-grade-pad` from dot count × size × gap — never a hardcoded pixel width.
6. **JS reads CSS.** Virtualized row height and layout math use `getComputedStyle` on `--row-h`, not a duplicate constant.

## Body default

```css
#app {
  display: flex;
  flex-direction: column;
  height: 100vh;
}
```
