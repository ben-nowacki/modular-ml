# Tailwind token architecture

How styling resolves to the project's theme so components never hardcode color or
size. Adapted from the public `tailwind-design-system` and `design-system` skills;
the concrete token names and file live in the project manifest.

## The rule

Components use **semantic utility classes** that resolve to CSS variables. They
never write raw hex or raw palette shades.

- Write `bg-surface`, `text-muted-foreground`, `border-border`, `text-destructive`.
- Never `bg-[#1e293b]`, `bg-zinc-800`, `text-blue-600`.

## Tailwind v4 (CSS-first)

Tokens live in a CSS file as CSS variables, mapped to Tailwind utilities via an
`@theme inline` block. Dark mode overrides the variables under a `.dark` class on
`<html>`; components never need `dark:` color variants.

```css
:root {
  --background: 0 0% 100%;
  --foreground: 222 47% 11%;
  --surface: 0 0% 98%;
  --border: 220 13% 91%;
  --muted: 220 14% 96%;
  --muted-foreground: 220 9% 46%;
  --primary: 222 47% 11%;
  --primary-foreground: 0 0% 98%;
  --destructive: 0 72% 51%;
  --radius: 0.5rem;
}
.dark {
  --background: 222 47% 11%;
  --foreground: 0 0% 98%;
  /* …overrides… */
}
@theme inline {
  --color-background: hsl(var(--background));
  --color-foreground: hsl(var(--foreground));
  --color-surface: hsl(var(--surface));
  --color-border: hsl(var(--border));
  --color-muted: hsl(var(--muted));
  --color-muted-foreground: hsl(var(--muted-foreground));
  --color-primary: hsl(var(--primary));
  --color-primary-foreground: hsl(var(--primary-foreground));
  --color-destructive: hsl(var(--destructive));
  --radius: var(--radius);
}
```

## Tailwind v3

The same semantic names, defined in `tailwind.config.{js,ts}` under
`theme.extend.colors` pointing at CSS variables, with `darkMode: "class"`.

## Token categories to define

Color (background/foreground, surface/card, border, muted + muted-foreground,
primary + foreground, destructive/warning/success + foregrounds), radius (one base,
nested radii concentric), type scale (a small ladder), spacing (one grid), motion
(durations + easing). Keep the set small and semantic — one accent, not five.

## Contrast

Gate every text/background pair at WCAG AA (4.5:1 body text, 3:1 large text and UI
boundaries). Interactive states (hover/active/focus) should GAIN contrast, not lose
it. Never encode meaning in color alone — pair with text, icon, or shape.
