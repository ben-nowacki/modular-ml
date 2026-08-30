# Anti-generic checklist

Consistency is not blandness. A design system exists so screens share a vocabulary
AND look intentionally designed. Adapted from the public `design-taste-frontend`
and `frontend-design` skills.

## The "AI look" — forbidden defaults

- Uniform rounded-shadow **card grids** where the data is really a table, list, or
  form. Pick the form that fits the data; cards are not the default container.
- **Gradients and glassmorphism** as decoration; blur/opacity used to look "modern."
- **Centered hero + three feature cards** marketing layout inside an app.
- **Everything emphasized** — multiple competing accent colors, many bold weights,
  full-saturation fills everywhere. Emphasis only works when most things are quiet.
- **Emoji as icons.** Use the project's icon set at a consistent size.
- **Placeholder marketing copy** ("Lorem", "Supercharge your workflow").
- Reflexive **Inter-for-everything** and reflexive AI-purple accent. Choose type
  and color for this product, from the manifest tokens.

## What intentional design looks like

- **One signature element per screen** — a single focal moment (a well-designed
  table, a strong header, one chart) — and restraint everywhere else.
- **Hierarchy from weight, size, and color** — not from boxes and borders around
  everything. Whitespace does most of the grouping.
- **One spacing rhythm per structural level.** Consistent gaps; align to the grid.
- **Real states designed, not afterthoughts** — loading (skeleton or spinner that
  matches layout), empty (helpful, with the primary action), error (recoverable).
- **Motion with purpose** — subtle `transform`/`opacity` transitions at the
  manifest's durations; nothing bounces or slides without reason.
- **Density that matches the task** — dense for data-heavy tools, roomy for focused
  flows. Match the density the rest of the app already uses.

## Quick self-check before shipping a screen

1. Does it use the same primitives, tokens, and spacing rhythm as existing screens?
2. Is there exactly one focal point, with everything else supporting it?
3. Are loading / empty / error actually designed?
4. Would this look like the same team built it as the screen next to it?
5. Any card/gradient/emoji/hero tell to remove?
