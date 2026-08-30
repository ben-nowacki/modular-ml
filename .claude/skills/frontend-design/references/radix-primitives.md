# Radix primitives + Tailwind — patterns

How to build accessible, themeable components on Radix UI, styled with Tailwind
and the project's tokens. Adapted from the public `radix-ui-design-system` and
`tailwind-design-system` skills; conform all of it to the project manifest.

## The `cn()` helper

Merge conditional classes and resolve Tailwind conflicts (`clsx` +
`tailwind-merge`). Use the project's existing helper (path in the manifest).

```ts
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
```

## Variants with CVA

Define a primitive's look once, as data. Never re-type these classes in feature
components — import the primitive.

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

Colors above are illustrative token names — use the project's actual tokens.

## Core Radix patterns

**Compound components + context.** Radix parts (`Root`, `Trigger`, `Content`,
`Item`, …) share state through context. Compose them; don't reach around them.

**`asChild` (polymorphism).** Merge a primitive's behavior onto your own element
instead of rendering an extra wrapper — e.g. a styled `Button` as a Radix trigger:

```tsx
<Dialog.Trigger asChild>
  <Button variant="outline">Open</Button>
</Dialog.Trigger>
```

**Controlled vs uncontrolled.** Uncontrolled (internal state) by default; lift to
controlled (`open` / `onOpenChange`) only when a parent must drive it.

**Portals + overlays.** Use the primitive's `Portal` for dialogs/menus/tooltips so
they escape overflow/stacking contexts. Don't hand-manage z-index or focus traps —
Radix does it. Always give a `Dialog` a `Title` (visually-hidden if needed) for
screen readers.

**Animation.** Animate `transform`/`opacity` only, on Radix `data-[state=…]`
attributes (`data-[state=open]`, `data-[state=closed]`), at the manifest's motion
durations. Keep it subtle.

## Accessibility checklist (Radix gives most of this — don't break it)

Keyboard: Tab/Shift-Tab, Escape to close, arrow keys in menus/selects. Focus
returns to the trigger on close. `aria-label` on icon-only triggers. Visible
`focus-visible` ring on every interactive element. Don't remove Radix's roles/ARIA.

See `examples/` for full, runnable Dialog / DropdownMenu / Select components.
