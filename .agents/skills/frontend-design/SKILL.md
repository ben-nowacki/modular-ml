---
name: frontend-design
description: Use when creating or modifying any UI — React components, pages, styling, layout, dialogs/menus/forms, or visual design. Enforces this project's Radix UI + Tailwind design system: one theme, shared primitives, full feature coverage, non-generic design.
---

# Frontend design

All frontend work in this project is built on **Radix UI primitives + Tailwind
CSS**, against the project's own established theme. UI you add must look and behave
like the same hand built the rest of the app, and must expose every user-facing
backend capability in its scope — not just the happy path the prompt named.

## Step 0 — read the design system manifest (always)

Before writing any UI, read `mission-control-notes/design_system.md`.

- **`Status: established`** → it is law. Use its tokens and its primitive kit;
  extend the system in place, never fork a parallel style. Details:
  `mission-control-notes/guides/design_system_manifest_guide.md`.
- **Missing / `not-established`** → do NOT invent a one-off style. The design
  system must be bootstrapped first (`design_system_bootstrap_guide.md`). If you
  are an execution agent, stop and surface this with `NEED_INPUT:` rather than
  guessing a theme — a guessed theme is how projects drift into inconsistency.

## Read next

Under `mission-control-notes/guides/` (read the ones that apply):
`frontend_design_guide.md` (judgment: hierarchy, density, states, anti-generic),
`tailwind_ui_patterns.md` (token mechanics), `react_component_patterns.md`
(component/hook structure), `frontend_feature_coverage_guide.md` (exposing all
backend features). This skill's own `references/`, `examples/`, and `templates/`
hold the Radix + Tailwind patterns to copy.

## Non-negotiables

- **Radix for every interactive primitive.** Dialog, dropdown/context menu,
  select, combobox, tooltip, popover, tabs, checkbox, radio, switch, slider,
  accordion → Radix. Never hand-roll these (focus traps, keyboard nav, ARIA,
  portals must be correct). Style them with Tailwind + the project tokens.
- **Reuse the project's primitive kit.** Import `Button`, `Input`, `Select`,
  `Dialog`, etc. from the kit named in the manifest. Do not re-declare a button's
  classes inline in a feature component — compose the shared primitive.
- **Semantic tokens only.** `bg-surface`, `text-muted-foreground`,
  `border-border`, `text-destructive`, and the project's named tokens — never raw
  hex (`bg-[#…]`), never a raw palette shade (`bg-blue-500`, `text-zinc-700`),
  never a color outside the token file. Compose variants with CVA + `cn()`.
- **All states.** Every interactive element: hover, focus-visible, disabled,
  pending. Every data view: loading, error, and empty handled explicitly.
- **Accessibility.** Semantic elements, labeled inputs, `aria-label` on icon-only
  buttons, meaning never carried by color alone. Radix gives you focus/keyboard
  for free — use it, don't defeat it.
- **Full feature coverage.** Follow `frontend_feature_coverage_guide.md`: expose
  every user-facing backend capability in scope; no silent gaps.
- **No AI attribution** in any commit, PR, comment, or file.

## Banned (do not introduce)

shadcn component dumps (author against the project tokens instead); CSS modules;
styled-components; standalone `.css` files beyond the token file; raw hex/palette
utility classes; manual `dark:` color overrides on token-driven colors; emoji as
icons (use the project icon set); inline `style={{…}}` except for genuinely
dynamic values (measured sizes, drag offsets).

## Anti-generic checklist

Do not ship: uniform rounded-shadow card grids where a table or form fits;
gradients/glassmorphism; hero-plus-three-feature-cards layouts; everything
emphasized at once; placeholder marketing copy. One signature element per screen,
restraint everywhere else; one spacing rhythm per structural level; hierarchy from
weight/size/color. See `references/anti-generic-checklist.md`.

## Before finishing

1. Re-read your diff against the manifest, the applicable guides, and the feature
   exposure matrix — every user-facing capability in scope is reachable in the UI.
2. Run the repo's frontend gates for the files you changed (typecheck, lint,
   tests, build) and fix everything they report.
3. If you added a shared primitive, a token, or changed the theme, update
   `mission-control-notes/design_system.md` in the same change.
