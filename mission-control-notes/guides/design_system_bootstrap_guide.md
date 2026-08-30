# Design System Bootstrap Guide

Follow this guide when a project has no established design system — the manifest
`mission-control-notes/design_system.md` is missing or says `Status:
not-established` — and frontend work needs one. The output is a real, working
theme + Radix primitive kit plus a completed manifest, not a document about one.

The mandated stack is **Radix UI primitives + Tailwind CSS**. Bootstrapping picks
THIS project's theme (color, type, spacing, motion) and scaffolds the primitive
kit that every future screen will reuse.

This is normally done in a spec/design-system session with the user, because
theme choices are taste choices. An execution agent that discovers a missing
design system must not silently invent one — surface it and let a design-system
session run first.

---

## 1. Decide the posture (ask the user)

Detect what already exists in the frontend:

- **Greenfield** (no frontend, or no styling system yet) → design a fresh theme
  and kit. Posture is `greenfield-only` by definition.
- **An existing non-Radix design system** (hand-rolled components, a different UI
  library, ad-hoc CSS) → the DEFAULT is a full migration to Radix + Tailwind, but
  this is expensive and you must **ask the user to confirm** before committing:
  - `full-migration` — rebuild existing interactive components on Radix, move
    styling onto the token system, converge everything on the new kit over time.
  - `greenfield-only` — build all NEW surfaces on Radix + the new theme; leave
    existing screens as they are until separately migrated.

  Record the chosen posture in the manifest so it is never re-litigated per task.

---

## 2. Design the tokens (the theme identity)

Every color, size, and motion value traces back to a named token. No raw hex in
components, ever. Define, in the project's token file (Tailwind v4: a `@theme` /
`@theme inline` block in a CSS file; Tailwind v3: the `tailwind.config` theme):

- **Color — semantic, not literal.** Name by role: `background`, `foreground`,
  `surface`/`card`, `border`, `muted` + `muted-foreground`, `primary` +
  `primary-foreground`, and the status trio `destructive` / `warning` / `success`
  (each with a foreground). Pick a restrained palette; one accent, not five.
  Every interactive state (hover/active/focus) must gain, not lose, contrast.
  Gate all text/background pairs at WCAG AA (4.5:1 body, 3:1 large/UI).
- **Radius** — one base `--radius` token; derive nested radii concentrically.
- **Type scale** — a small, deliberate ladder (e.g. body / dense-UI / micro /
  headings). Self-host fonts (WOFF2, preload above-the-fold) to avoid FOUT.
  Avoid the default "AI look": don't reach for Inter-for-everything by reflex.
- **Spacing** — one rhythm (4px or 8pt grid). Consistent gaps per structural level.
- **Motion** — defaults of ~150ms for micro-interactions, ~200–250ms for
  transitions, ease-out; animate only `transform`/`opacity`.
- **Dark mode** — define the mechanism now (token overrides under a `.dark` class
  on `<html>`), so no component ever needs manual `dark:` color overrides.

Map tokens to Tailwind utilities so components write `bg-surface`,
`text-muted-foreground`, `border-border` — never `bg-[#...]` or `bg-zinc-800`.

## 3. Scaffold the primitive kit (Radix + Tailwind)

Create a single shared UI directory (e.g. `src/components/ui/`) holding the
primitives every screen composes from. Build interactive ones on Radix so
accessibility (focus management, keyboard nav, ARIA, portals) is correct by
construction. Minimum kit:

- `Button` (variants via CVA: primary / outline / ghost / destructive; sizes)
- `Input`, `Textarea`, `Select` (Radix Select), `Label`, field/error helpers
- `Dialog` (Radix Dialog — requires a Title), `DropdownMenu` (Radix), `Tooltip`
  (Radix), and the status indicator this project uses (chip/badge)

Theme them with the `cn()` helper (`clsx` + `tailwind-merge`) and CVA variants —
the pattern documented in the `frontend-design` skill's `references/` and
`examples/`. Do not paste a shadcn dump; author the kit against the project's
tokens. Every primitive supports hover / focus-visible / disabled / pending.

## 4. Write the manifest and prove it renders

Fill in `mission-control-notes/design_system.md` per
`design_system_manifest_guide.md`: `Status: established`, the token file path, the
kit directory, the theme identity, the recorded posture, the conventions.

Then verify it actually works, not just compiles: build the project, typecheck and
lint clean, and render at least one real screen (or a primitives showcase) so the
theme, dark mode, and interaction states are visibly correct. A design system that
only exists on paper is not established.

## Anti-generic bar

The point of a design system is consistency AND intentional design — not bland
uniformity. Avoid the AI tells: uniform rounded-shadow card grids where a table or
form fits, gradients/glassmorphism, emoji-as-icons (use an icon set), everything
emphasized at once, centered hero + three feature cards. One signature element per
screen, restraint everywhere else, hierarchy from weight/size/color.
