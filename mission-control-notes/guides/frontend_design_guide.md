# Frontend Design Quality Guide

The judgment layer for frontend work: what makes a screen look deliberate,
consistent, and professionally finished rather than generated. `tailwind_ui_patterns.md`
documents the styling mechanics (tokens, CVA, Radix primitives); this guide governs
the choices those mechanics leave open — hierarchy, density, interaction states,
and not looking generic.

**Baseline (non-negotiable):** the project's design system is mandatory. Read
`mission-control-notes/design_system.md` first — it names this project's tokens,
its Radix + Tailwind primitive kit, its type/size scale, its motion. Use those;
never invent a parallel style. The stack is always **Radix UI + Tailwind**. If
the manifest is missing or `not-established`, stop and bootstrap it
(`design_system_bootstrap_guide.md`) — an execution agent surfaces this with
`NEED_INPUT:` rather than guessing a theme. Nothing below overrides the manifest;
this guide governs taste on top of it.

---

## 1. Avoiding generic "AI-looking" UI

Certain defaults instantly read as template output. The full enumeration lives in
the `frontend-design` skill's `references/anti-generic-checklist.md` — read it. The
essentials that matter most here:

- **No card-grid-for-everything.** Not every list is a grid of rounded-shadow
  cards. Dense tabular data belongs in a real table; a short list of settings
  belongs in a plain stacked form. Reach for a card gallery only when items are
  genuinely peer objects the user scans visually.
- **No gratuitous gradients, glows, or glassmorphism.** Surfaces are flat, on the
  manifest's `surface`/`card` tokens with hairline `border` tokens. No gradient
  headers, no drop-shadow halos.
- **No emoji as icons.** Use the project's icon set, sized to the control. Never
  emoji in labels, buttons, headings, or empty states.
- **No hero-with-three-feature-cards / marketing whitespace** on an app screen.
  Open with a real header row and get straight to the content.
- **No placeholder-sounding copy.** "Manage your equipment with ease" is filler.
  Labels state what a thing is; empty states say what's missing and the next
  action ("No protocols yet. Create one to start a run."). Terse, concrete,
  sentence case.
- **Decoration is not differentiation.** If two elements differ in importance, show
  it with hierarchy (size/weight/color — §3), not borders and backgrounds piled on
  both.

One screen earns **at most one signature moment** — a stat row, a highlighted
current-state panel, a color-coded timeline. Everything else stays quiet so that
one element carries the screen. When everything is emphasized, nothing is.

## 2. Spacing & layout discipline

- **One rhythm per screen.** Pick the gap for a given grouping level (tight gaps
  inside clusters, wider gaps between blocks — use the spacing rhythm the manifest
  records) and hold it. Mixed gaps at the same structural level is the single most
  common tell of unconsidered UI.
- **Align to an implicit grid.** Edges of cards, table margins, and header rows
  share left/right boundaries down the page. If an element is indented differently
  from its siblings, that indentation must mean something.
- **Density matches the content.** Data-heavy surfaces stay compact; don't air them
  out to look "modern" or cram a settings form to look "dense". Match the
  neighboring screens and the manifest's scale.
- **Group by proximity first.** Related controls sit closer to each other than to
  unrelated ones; a divider or heading is the fallback, not the default.

## 3. Typography

- **Stay inside the manifest's type scale.** Use the ladder it defines (a small,
  deliberate set — body / dense-UI / micro / headings). Do not introduce new sizes,
  new fonts, or oversized display type the scale doesn't include.
- **Hierarchy comes from weight, size, and text color** — a heavier weight and the
  `foreground` token for the heading, `muted-foreground` for supporting text and
  annotations. Not from underlines, italics, or ALL-CAPS (uppercase is reserved for
  an established section-header idiom if the project has one).
- **Two levels of emphasis per block, maximum.** A row that is bold, colored,
  larger, and uppercase all at once has no hierarchy at all.

## 4. Color discipline

- **Semantic tokens only — and semantically.** The `destructive` token means an
  error state, not "I wanted red here". The `primary` accent marks interactive
  elements, not decoration. Never raw hex or a raw palette shade; see
  `tailwind_ui_patterns.md`.
- **Neutral by default.** Screens are surface/neutral-toned; color is information.
  A view with five accent panels reads as noise — most screens carry one accent
  (the primary action) plus status colors where state genuinely differs.
- **Statuses go through the project's status primitive** (the badge/chip named in
  the manifest), so the same state is the same color everywhere. Never a second
  color vocabulary, never a hand-picked status color at the call site.

## 5. Interaction states

Every interactive element ships with all of its states — a control with only a
resting style is half-built. Radix primitives give you the behavior; you supply the
visible states via tokens:

- **Hover**: visible but subtle (a token hover variant, e.g. `hover:bg-muted` or a
  `-hover` token).
- **Focus**: keyboard focus must be visible — `focus-visible:` ring on the
  manifest's ring/focus token. Never `outline-none` without a replacement. Don't
  defeat the focus ring Radix provides.
- **Disabled**: `disabled:opacity-50 disabled:pointer-events-none` (or the kit's
  equivalent) — and actually set `disabled`, not just style it.
- **Pending/loading**: an async action disables its trigger and shows progress
  (spinner or label swap) while in flight; double-submits must be impossible. On
  completion the UI reflects the result — success invalidates the affected query;
  failure shows an inline error near the action (`react_component_patterns.md` §4).
  Prefer showing the pending state over optimistic updates unless the interaction
  demands instant feedback.
- **Empty/error surfaces** are designed states with a useful description and next
  action, not afterthoughts.

## 6. Accessibility basics

Accessibility is part of design quality, not a separate pass. Radix handles most of
the interactive mechanics — use it, don't break it:

- **Semantic elements first**: `<button>` for actions, `<a>` for navigation, real
  `<table>` for tables, headings in order. No clickable `<div>`s.
- **Every input has a label** (`<label htmlFor>` or `aria-label`); every icon-only
  button has an `aria-label`/`title`.
- **Overlays manage focus**: Radix Dialog/DropdownMenu/Popover move focus in on
  open, trap it, return it to the trigger on close, and close on Escape — get these
  from Radix rather than hand-rolling. Always give a Dialog a Title.
- **Contrast**: body text on surface tokens meets AA already; don't put
  `muted-foreground` on tinted backgrounds, and never encode meaning in color alone
  — pair color with a label, icon, or shape.
- **Keyboard reachability**: everything clickable is tabbable and activatable with
  Enter/Space, in a sensible order.

## 7. Self-check before the PR

- [ ] The screen reads as one deliberate design: one signature element, quiet
      everything else; no §1 anti-patterns (card-grid default, gradients, emoji
      icons, filler copy). Cross-checked against the anti-generic checklist.
- [ ] Spacing follows one rhythm; edges align; density matches neighboring screens.
- [ ] Type stays in the manifest's scale; hierarchy from weight/size/color; ≤2
      emphasis levels per block.
- [ ] Color is semantic and sparse; statuses via the project's status primitive.
- [ ] Every interactive element has hover, focus-visible, disabled, and pending
      states; async actions can't double-fire and surface their errors inline.
- [ ] Interactive primitives come from Radix via the kit; inputs labeled, icon
      buttons `aria-label`ed, overlays trap/restore focus, meaning never carried by
      color alone.
- [ ] No AI attribution in any commit, PR, comment, or file.
- [ ] Put side by side with an existing screen in this app, it looks like the same
      hand built both.
