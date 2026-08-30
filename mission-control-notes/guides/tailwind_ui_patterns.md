# Tailwind UI patterns

The styling-mechanics layer for frontend work: how classes resolve to the
project's theme, how variants are composed, and why interactive primitives must be
Radix. The goal is **consistency** — new UI should look like the same hand built
the existing screens. `frontend_design_guide.md` is the judgment layer (hierarchy,
density, states); read both for any visual work. This guide goes deeper than the
`frontend-design` skill's `references/tailwind-tokens.md` and `radix-primitives.md`
but stays consistent with them — those are the quick reference, this is the fuller
treatment.

> **Step 0: resolve everything to the manifest.** Read
> `mission-control-notes/design_system.md`. It records this project's token file,
> the Tailwind version and how tokens are exposed, the shared primitive kit
> directory, the `cn()` helper path, and the type/size/motion scale. Every token
> name and file path below is an **example** — use the project's actual names from
> the manifest. If the manifest is missing or `not-established`, stop and bootstrap
> it (`design_system_bootstrap_guide.md`); do not invent ad-hoc styling.

The mandated stack is **Radix UI + Tailwind CSS**, always.

---

## 1. Semantic tokens, never raw palette

Components use **semantic utility classes** that resolve to CSS variables named by
role. They never write raw hex or a raw palette shade — that is how a theme change
turns into a thousand-file diff, and how two screens drift apart.

- Write `bg-surface`, `text-muted-foreground`, `border-border`, `text-destructive`,
  `bg-primary text-primary-foreground`.
- Never `bg-[#1e293b]`, `bg-zinc-800`, `text-blue-600`, `bg-white`.

> Example token names (`surface`, `muted-foreground`, `border`, `primary`,
> `destructive`, `warning`, `success`). **Use the project's actual tokens from the
> manifest** — the roles are universal, the names are per project.

### Where tokens live (Tailwind v4, CSS-first)

Tokens are CSS variables in the project's token file, mapped to Tailwind utilities
via an `@theme inline` block. Dark mode overrides the variables under a `.dark`
class on `<html>`, so components never need `dark:` color variants.

```css
:root {
  --surface: 0 0% 98%;
  --foreground: 222 47% 11%;
  --border: 220 13% 91%;
  --muted: 220 14% 96%;
  --muted-foreground: 220 9% 46%;
  --primary: 222 47% 11%;
  --primary-foreground: 0 0% 98%;
  --destructive: 0 72% 51%;
  --radius: 0.5rem;
}
.dark { --surface: 222 47% 11%; --foreground: 0 0% 98%; /* …overrides… */ }
@theme inline {
  --color-surface: hsl(var(--surface));
  --color-foreground: hsl(var(--foreground));
  --color-border: hsl(var(--border));
  --color-muted: hsl(var(--muted));
  --color-muted-foreground: hsl(var(--muted-foreground));
  --color-primary: hsl(var(--primary));
  --color-primary-foreground: hsl(var(--primary-foreground));
  --color-destructive: hsl(var(--destructive));
  --radius: var(--radius);
}
```

On **Tailwind v3** the same semantic names live in `tailwind.config.{js,ts}` under
`theme.extend.colors` pointing at those CSS variables, with `darkMode: "class"`.
Either way: **add or change a token in the token file, never inline in a
component.** The manifest names the exact file and mechanism.

### Contrast

Gate every text/background pair at WCAG AA (4.5:1 body, 3:1 large text and UI
boundaries). Interactive states (hover/active/focus) should *gain* contrast, not
lose it. Never encode meaning in color alone — pair with text, icon, or shape.

---

## 2. `cn()` and CVA variants

**`cn()`** merges conditional classes and resolves Tailwind conflicts (`clsx` +
`tailwind-merge`). Use the project's existing helper — the manifest gives its path
(commonly `@/lib/utils`). Compose every className list with it; don't leave
duplicated/conflicting utilities for tailwind-merge to silently resolve.

```ts
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
```

**CVA** defines a primitive's look once, as data — variants and sizes in one place.
Never re-type these class strings in a feature component; import the primitive.

```tsx
import { cva, type VariantProps } from "class-variance-authority";

const button = cva(
  "inline-flex items-center justify-center gap-2 rounded-md text-sm font-medium " +
  "transition-colors focus-visible:outline-none focus-visible:ring-2 " +
  "focus-visible:ring-ring disabled:opacity-50 disabled:pointer-events-none",
  {
    variants: {
      variant: {
        primary: "bg-primary text-primary-foreground hover:bg-primary/90",
        outline: "border border-border bg-transparent hover:bg-muted",
        ghost: "hover:bg-muted",
        destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90",
      },
      size: { sm: "h-8 px-3", md: "h-9 px-4", icon: "size-9" },
    },
    defaultVariants: { variant: "primary", size: "md" },
  }
);
```

> The token names and the size rungs above are **illustrative** — use the
> project's actual tokens and the size ladder recorded in the manifest so rows of
> mixed controls line up.

### The interpolation gotcha

Tailwind's JIT only generates classes it finds as **complete literal strings** in
the source scan. A computed class like `` `bg-${variant}` `` is purged and renders
unstyled. So:

> **Colors/classes chosen at runtime from data must be applied via inline `style`
> with `var(--color-…)`, or selected from a lookup whose full class strings appear
> literally in source.** Never build a Tailwind class by string interpolation.

Statically-known classes (a CVA variant map, `bg-primary`) are fine — they are
literal in the source. Inline `style` is otherwise reserved for genuinely dynamic
values (measured sizes, drag offsets).

---

## 3. Radix for every interactive primitive (required)

Hand-rolled dialogs, menus, and selects get focus management, keyboard nav, ARIA,
and portals wrong. **Every interactive primitive comes from Radix** — Dialog,
DropdownMenu/ContextMenu, Select, Combobox, Tooltip, Popover, Tabs, Checkbox,
Radio, Switch, Slider, Accordion — styled with Tailwind + the project tokens and
wrapped once in the shared kit. Import from the kit named in the manifest; do not
re-declare a primitive's classes inline in a feature component.

Core patterns:

- **Compound parts + context.** Radix parts (`Root`, `Trigger`, `Content`, `Item`)
  share state through context. Compose them; don't reach around them.
- **`asChild`** merges a primitive's behavior onto your own element — e.g. a styled
  `Button` as a Radix trigger — instead of rendering an extra wrapper:
  ```tsx
  <Dialog.Trigger asChild><Button variant="outline">Open</Button></Dialog.Trigger>
  ```
- **Controlled vs uncontrolled.** Uncontrolled by default; lift to controlled
  (`open` / `onOpenChange`) only when a parent must drive it.
- **Portals + overlays.** Use the primitive's `Portal` so overlays escape
  overflow/stacking contexts; don't hand-manage z-index or focus traps. Always give
  a Dialog a `Title` (visually-hidden if needed).
- **Animation.** Animate `transform`/`opacity` only, on Radix `data-[state=…]`
  attributes (`data-[state=open]`, `data-[state=closed]`), at the manifest's motion
  durations. Keep it subtle.

Keyboard/ARIA come from Radix for free — Tab/Escape/arrow keys, focus return to the
trigger, roles. Don't remove them; add `aria-label` on icon-only triggers and keep
a visible `focus-visible` ring. The `frontend-design` skill's `references/` and
`examples/` hold full runnable Dialog / DropdownMenu / Select components to copy.

---

## 4. Reuse the primitive kit; don't reinvent

Prefer the shared primitives in the kit directory (the manifest's path, e.g.
`src/components/ui/`) over raw markup — they encode the conventions above. Reach for
raw Tailwind only for one-off layout, not for a button/badge/table/select/dialog a
primitive already covers. A page composes primitives; it does not restyle them.

- Need a variant that doesn't exist (a destructive button, a new badge state)? Add
  it to the primitive's CVA map, don't hand-style at the call site.
- Added a new shared primitive or a new token, or changed the theme? Update
  `mission-control-notes/design_system.md` in the same change — the manifest must
  stay true.

---

## 5. Layout patterns

- **Header/toolbar row:** `flex items-center justify-between` (title left, actions
  right) — the most common page-header pattern.
- **Inline clusters:** `flex items-center gap-2` for icon+label groups, button
  groups, badge rows.
- **Vertical stacks:** `flex flex-col gap-*` or `space-y-*`.
- **Card galleries / stat rows:** grid with mobile-first progressive columns —
  `grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4`. Tables stay full-width and
  scroll (`min-w-full`, `overflow-x-auto` wrapper) rather than reflowing per column.

Default to flex for one-dimensional layout and grid for card/stat galleries; add a
breakpoint prefix only where the layout genuinely needs to reflow. Don't scatter
`sm:`/`md:` across every element. Follow the manifest's spacing rhythm for the gap
values.

---

## Banned

shadcn component dumps (author the kit against the project's tokens instead); CSS
modules; styled-components; standalone `.css` files beyond the token file; raw
hex/palette utility classes (`bg-[#…]`, `bg-blue-500`); manual `dark:` color
overrides on token-driven colors; interpolated Tailwind class names; hand-rolled
interactive primitives where Radix has one; inline `style={{…}}` except for
genuinely dynamic values.

## Pre-PR checklist

- [ ] **Tokens, not raw shades.** Colors use the manifest's semantic tokens, never
      `bg-white`/`text-gray-*`/`bg-[#…]`. New tokens go in the token file.
- [ ] **Radix for every interactive primitive**, imported from the shared kit;
      none hand-rolled, none re-declared inline in a feature component.
- [ ] **Variants via CVA + `cn()`**; no re-typed class strings, no conflicting
      utilities left for tailwind-merge to resolve.
- [ ] **No interpolated Tailwind class names.** Runtime/data-driven colors use
      inline `style` with `var(--color-…)` or a literal class lookup.
- [ ] **On the manifest's scale.** Sizes, type, and spacing follow the recorded
      ladder and rhythm; statuses go through the project's status primitive.
- [ ] Manifest updated if a primitive, token, or the theme changed.
- [ ] No AI attribution anywhere.
