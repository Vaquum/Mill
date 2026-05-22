# Typography

Minimal spec for lab-style UIs. Copy `:root` typography atoms/composites and the six role classes.

## Atoms

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

## Composites (only these in `font:` declarations)

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

## Roles

| Class / role | Token | Family rule |
|---|---|---|
| `.t-body` | `--text-body` | sans — prose, status, tooltips |
| `.t-ui` | `--text-ui` | sans — controls, selects, chrome |
| `.t-label` | `--text-label` + uppercase + `--track-label` | sans — section headers, column headers, actions |
| `.t-data` | `--text-data` + `tabular-nums` | mono — grid values, dates, numeric inputs, stats |
| `.t-code` | `--text-code` + `tabular-nums slashed-zero` | mono — YAML/JSON/manifest blocks |
| `.t-icon` | `--text-icon` | sans — ×, ▲, ▼, ▶ glyphs |

Emphasis inside data (e.g. bold counts): `--text-data-em`, still mono.

## Rules

1. **One size for UI/data (13px).** Icons/glyphs only at 11px (`--t-icon`).
2. **Only data is mono.** Everything else is sans — including selects, palette items, metric names, placeholders.
3. **Two line heights:** `1` (controls, data cells, icons) and `1.4` (body, prose, code blocks).
4. **Labels are one treatment:** 500 sans, 13px, uppercase, `0.08em` tracking — always via `.t-label` or the shared label selector block.
5. **`font-variant-numeric`:** `tabular-nums` on all mono data/code; `slashed-zero` on code blocks only.
6. **No literal typography** in rules — only composite tokens and role classes.

## Body default

```css
html, body { font: var(--text-body); }
```
