# Design System Manifest Guide

Every project has ONE design system, recorded in a single manifest file:

    mission-control-notes/design_system.md

This manifest is the source of truth for how the UI looks and is built. It is
**project-owned**: Mission Control never overwrites it. Read it before any
frontend work; keep it accurate when you change the system.

The mandated stack for all Mission Control frontend work is **Radix UI
primitives + Tailwind CSS**. The manifest records how THIS project realizes that
stack — its tokens, its primitive kit, its theme identity — because the palette,
type scale, and component conventions differ per project. There is no global
theme; each project defines its own.

---

## Step 0 for every frontend task: read the manifest

Before writing or changing any UI, read `mission-control-notes/design_system.md`.

- **It exists and `Status: established`** → this is law. Use its tokens and its
  primitive kit. Do not introduce a second styling vocabulary, a parallel button,
  or a color outside the token set. Extend the system in place; never fork it.
- **It is missing, or `Status: not-established`** → the design system has not been
  designed yet. Do NOT invent an ad-hoc style for this one task — that is exactly
  how projects drift into inconsistency. Stop and run the bootstrap flow first:
  see `design_system_bootstrap_guide.md`. An execution agent that hits this state
  must surface it (`NEED_INPUT:`) rather than guess a theme.

---

## Manifest format

The manifest is markdown with this structure. Fill every field; write `none` /
`n/a` explicitly rather than leaving a field blank.

```markdown
# Design System

- **Status:** established | not-established
- **Stack:** Radix UI <version> + Tailwind CSS <version>
- **Migration posture:** full-migration | greenfield-only    (see below)

## Token source of truth
- File: <path to the token/theme file, e.g. src/styles/theme.css>
- How tokens are exposed to Tailwind: <e.g. @theme inline block / tailwind config>

## Primitive kit
- Directory: <path to shared UI primitives, e.g. src/components/ui/>
- Primitives available: Button, Input, Select, Dialog, DropdownMenu, Tooltip, ...
- The `cn()` helper: <path, e.g. src/lib/utils.ts>

## Theme identity
- Color: <named semantic tokens — bg/surface, foreground, border, primary,
  muted, destructive/warning/success, ...>. Semantic names only; no raw hex.
- Radius: <token, e.g. --radius: 0.5rem>
- Type scale: <the ladder, e.g. text-sm body / text-xs dense / text-2xs micro>
- Spacing rhythm: <base unit, e.g. 4px / 8pt grid>
- Motion: <default durations/easing, e.g. 150ms micro, 200ms transitions, ease-out>

## Conventions (non-negotiable)
- Radix primitives for every interactive element (dialog, menu, select, tooltip,
  popover, tabs, checkbox, radio, switch, slider) — no hand-rolled equivalents.
- Tailwind utilities for styling; semantic tokens only.
- Compose variants with CVA + `cn()`; no inline style objects except for values
  that are genuinely dynamic (measured sizes, drag offsets).
- Banned: shadcn component dumps, CSS modules, styled-components, standalone
  `.css` files beyond the token file, raw hex/palette classes (`bg-blue-500`),
  manual `dark:` overrides on token-driven colors.

## Dark mode
- Mechanism: <e.g. `.dark` class on <html>, tokens overridden under `.dark`>
```

---

## Keeping the manifest true

The manifest describes reality, not aspiration. If a task adds a new shared
primitive, a new semantic token, or changes the theme, update the manifest in the
same change (additive edit, commit with the work). A manifest that lies is worse
than none — agents will trust it. Cross-check: `design_system_bootstrap_guide.md`
(how the system was designed), and the `frontend-design` skill (the enforcement
rules that read this manifest).
