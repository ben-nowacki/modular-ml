# JavaScript / TypeScript Docstring & Comment Style Guide

Reference: https://jsdoc.app/

This guide applies to all `.js`, `.ts`, `.jsx`, and `.tsx` files in the AmpWell codebase.
Read it before writing or editing any docstring or inline comment. Rules below are
non-negotiable — do not deviate even when existing code does otherwise.

---

## 1. Docstring Style

Use **JSDoc** for all documentation comments. No plain block comments (`/* */`) for
documentation purposes — those are for inline suppression only (e.g. eslint-disable).

JSDoc comments use `/** ... */` syntax. Regular block comments use `/* ... */`.
Do not use `/** */` for non-documentation purposes.

---

## 2. File-Level Docstrings

Every `.ts` and `.tsx` file must have a one-line `/** */` docstring at the very top,
before any imports.

```typescript
/** API client for all AmpWell backend endpoints. */

import { Equipment } from '../types/equipment';
```

One line only. No multi-line file-level docstrings.

---

## 3. Interface and Type Docstrings

All exported interfaces and type aliases must have a docstring. For interfaces, document
each property inline using `/** */` on the line immediately above it.

Multi-line rule: the first line of text starts on a **new line after the opening `/**`**.
The closing `*/` sits on its own line, preceded by a blank line if the docstring is
multi-line.

```typescript
/**
 * Equipment record as returned by the AmpWell registry API.
 *
 * Represents a physical cycler or auxiliary device registered in the system.
 * Use {@link EquipmentDriver} to communicate with the device at runtime.
 */
export interface Equipment {
  /** UUID of the equipment record. */
  id: string;

  /** Human-readable display name assigned during registration. */
  name: string;

  /** Current connection status reported by the Bridge Agent heartbeat. */
  status: 'online' | 'offline' | 'error';

  /** Timestamp of the last successful Bridge Agent poll, in ISO 8601 format. */
  lastSeenAt: string | null;
}
```

Single-property docstrings sit inline on one line: `/** Description. */`
Multi-sentence property descriptions use the multi-line format:

```typescript
/**
 * Raw capability flags from the equipment registry.
 *
 * Keyed by capability name (e.g. `max_voltage_v`, `eis_capable`).
 * Validated against protocol requirements before a run starts.
 */
capabilities: Record<string, unknown>;
```

---

## 4. Class Docstrings

All exported classes must have a multi-line JSDoc docstring. Required tags:

- **Description** — first block, no tag, explains what the class is and does
- `@param` on the constructor — document constructor parameters here, not on
  the constructor method itself
- `@example` — include when instantiation is non-obvious

```typescript
/**
 * WebSocket client for live channel monitoring.
 *
 * Manages a persistent connection to the AmpWell WebSocket endpoint,
 * reconnecting automatically on dropout. Emits typed events for each
 * incoming data frame.
 *
 * @param channelId - UUID of the channel to subscribe to.
 * @param onData - Callback invoked with each normalised data frame.
 * @param options - Optional configuration overrides.
 *
 * @example
 * const client = new ChannelWebSocket(channelId, (frame) => {
 *   console.log(frame.voltageV);
 * });
 * client.connect();
 */
export class ChannelWebSocket {
```

---

## 5. Function and Method Docstrings

All exported functions and all public class methods must have JSDoc docstrings.

Required tags:

- **Description block** — first, no tag, one sentence minimum
- `@param name - description` — one entry per parameter (no type annotation in the tag;
  TypeScript types are in the signature)
- `@returns description` — omit entirely if the function returns `void` or `Promise<void>`
- `@throws {ErrorType} description` — include when the function explicitly throws

Multi-line rule: same as classes and interfaces — first line on a new line after `/**`.

```typescript
/**
 * Add a command to the pending queue for a connected device.
 *
 * Signs the request with HMAC headers and posts to the long-poll endpoint.
 * Resolves with the newly created command UUID on success.
 *
 * @param equipmentId - UUID of the target equipment record.
 * @param command - Command name as defined in the Bridge Agent protocol.
 * @param params - Command-specific parameters passed verbatim to the agent.
 *
 * @returns UUID of the newly created pending command record.
 *
 * @throws {EquipmentOfflineError} If the equipment's last heartbeat exceeds
 *   the offline threshold.
 */
async function enqueueCommand(
  equipmentId: string,
  command: string,
  params: Record<string, unknown>,
): Promise<string> {
```

**Single-line functions** that are self-explanatory may use a one-liner:

```typescript
/** Convert a zero-based channel index to its human-readable display label. */
function channelLabel(index: number): string {
```

But if the function has parameters that need explanation, always use the multi-line form.

---

## 6. React Component Docstrings

Document all exported React components with a JSDoc comment above the function or
`const` declaration. Document props via the interface (see §3), not on the component
itself. The component docstring describes behaviour and usage, not the props.

```typescript
/**
 * Live monitoring card for a single equipment channel.
 *
 * Subscribes to the WebSocket feed for `channelId` and renders a real-time
 * voltage/current chart. Reconnects automatically on dropout. Displays a
 * disconnected state badge when the Bridge Agent goes offline.
 */
export function ChannelMonitorCard({ channelId, onError }: ChannelMonitorCardProps) {
```

Do not put `@param` tags on React component docstrings — props are documented on the
`Props` interface.

---

## 7. Cross-References

Use JSDoc `{@link}` inline tags to reference other symbols in prose descriptions.

```typescript
/**
 * Return the {@link EquipmentDriver} for this channel's equipment.
 *
 * Calls {@link EquipmentDriver.connect} internally if the driver is not yet
 * initialised. See {@link ChannelCapabilities} for the returned capability flags.
 */
```

Use `{@link}` in prose blocks only — not inside `@param` or `@returns` descriptions
unless the reference genuinely aids understanding.

---

## 8. Backtick Usage in Prose

Use **single backticks** in JSDoc prose to mark variable names, argument names,
property names, and string literals. Do not use double backticks.

```typescript
/**
 * Parse the raw poll response and extract any pending commands.
 *
 * Returns an empty array if `commands` is absent or null in the response.
 * The `command_id` field on each entry must be echoed back in the result
 * payload after the command executes.
 */
```

---

## 9. Line Length

No line in a JSDoc comment may exceed **88 characters**, including the ` * ` prefix
and indentation.

For a top-level function (no indentation), the usable content width per line is
85 characters (`/** ` = 4 chars, ` */` = 3 chars; prose starts after ` * ` = 3 chars,
leaving 85 chars of content).

For a method inside a class (4-space indent), usable width is 81 characters.

Wrap long sentences at word boundaries, aligning continuation lines to the same
` * ` column:

```typescript
/**
 * Normalise a raw Arbin timeseries file and write the result to the Iceberg
 * timeseries table, triggering metric extraction on completion.
 *
 * @param artifactPath - Absolute path to the raw `.res` file on the local
 *   filesystem. Must be accessible to the ingestion service process.
 */
```

---

## 10. ASCII-Only Docstrings

Never use non-standard ASCII characters in docstrings or comments. Specifically
forbidden:

- Greek letters (`alpha`, `beta`, `mu`, `sigma`) — spell them out
- Math symbols (`×`, `≤`, `≥`, `±`, `∑`) — use `x`, `<=`, `>=`, `+/-`, `sum`
- Em dash (`—`) and en dash (`–`) — use a plain hyphen `-`
- Arrow `→` — use `-->` if an arrow is genuinely needed
- Any Unicode symbol not in the 7-bit ASCII range (codepoint > 127)

```typescript
// CORRECT
/** Compute the mean +/- standard deviation of discharge capacity values. */

// WRONG
/** Compute the mean ± standard deviation of discharge capacity values. */
```

---

## 11. Typo and Description Accuracy

Fix any typos encountered while editing a docstring. Common ones:

| Wrong | Correct |
|---|---|
| `isntance` | `instance` |
| `specfiied` | `specified` |
| `Wether` | `Whether` |
| `indicies` | `indices` |
| `recieve` | `receive` |

Fix incorrect descriptions too — e.g. if a function named `getTargets` has a `@returns`
that says "Feature data", change it to "Target data". Do not leave wrong descriptions
in place.

---

## 12. Minimal Rewriting

Do not rewrite existing correct descriptions. Add what is missing and fix what is
wrong — do not rephrase for style.

---

## 13. Type Annotation Fixes

TypeScript types live in the **function signature**, not in JSDoc tags. Do not write
`@param {string} name` — write `@param name` and let the signature carry the type.

If you notice an incorrect type in a signature while updating a docstring, fix the
signature and keep the docstring consistent with it.

```typescript
// WRONG — type annotation in JSDoc tag (redundant with TypeScript)
/**
 * @param {string} equipmentId - UUID of the equipment record.
 * @returns {Promise<string>} The command UUID.
 */
async function enqueueCommand(equipmentId: string): Promise<string>

// CORRECT — type only in signature
/**
 * @param equipmentId - UUID of the equipment record.
 * @returns UUID of the newly created pending command record.
 */
async function enqueueCommand(equipmentId: string): Promise<string>
```

---

## 14. Comment Style

### General rules

- Comments are never long sentences — no periods or end-of-sentence punctuation
- Concise but descriptive — explain the *why*, not the *what*
- No Unicode characters (same rule as docstrings)
- Use `//` for all inline and single-line block comments, never `/* */`

### Inline comments

Single-thought inline comments sit on the same line, separated by two spaces:

```typescript
const timeout = 25;  // long-poll window in seconds
let retries = 0;     // reset on successful poll
```

### Short block comments

For a single thought that needs its own line:

```typescript
// sign and attach HMAC headers before sending
const headers = signRequest(equipmentId);
const response = await fetch(url, { method: 'POST', headers });
```

### Multi-line block comments

When a comment requires multiple lines or explains a non-obvious design decision,
use the banner block format. Structure:

- Opening and closing lines: `// ===...===` (match length to content width)
- First content line: main descriptive phrase (no period)
- Subsequent lines: bullet list with `-` for each sub-point

```typescript
// ==========================================================================
// Exit rule conditions
// - Outer list = OR, inner list = AND (mirrors schema: list[list[ValueExpr]])
// - Each UICondition serialises to a ValueExpr {expr: "metric comparator rhs"}
// ==========================================================================
const exitRules: UICondition[][] = [];
```

```typescript
// ==========================================================================
// WebSocket reconnect strategy
// - Initial backoff: 1s, doubles on each failure, capped at 30s
// - Resets to 1s after any successful message received
// - Does not reconnect if the component has unmounted (check mountedRef)
// ==========================================================================
const reconnect = useCallback(() => {
```

Keep banner widths consistent within a file. 74 characters total (`// ` + 71 `=`) is
the default. Adjust shorter for deeply indented code.

### What not to do

```typescript
// WRONG - end-of-sentence punctuation
// Sign and attach HMAC headers before sending.

// WRONG - restates the code without adding meaning
const equipment = await getEquipment(id);  // get equipment by id

// WRONG - Unicode in comment
// Voltage range: 2.5 --> 4.2 V  (arrow is fine, but no → symbol)

// WRONG - JSDoc syntax used for a non-documentation comment
/** This is not a docstring, just a note */
const POLL_TIMEOUT = 25;

// CORRECT - explains the why, no punctuation
// skip if equipment went offline between poll and command dispatch
if (!equipment.isConnected) {
  continue;
}
```

---

## Quick Reference Checklist

Use this when reviewing a file before committing:

- [ ] File-level docstring present at top (one line, before imports)
- [ ] All exported interfaces and type aliases have `/** */` docstrings
- [ ] All interface properties have inline `/** */` docstrings
- [ ] All exported classes have multi-line JSDoc with constructor `@param` tags
- [ ] All exported functions and public methods have JSDoc
- [ ] Multi-line docstrings: first line on new line after `/**`, closing `*/` on its own line
- [ ] No `@param {type}` annotations — types live in the TypeScript signature
- [ ] `@returns` omitted for `void` and `Promise<void>` return types
- [ ] No lines exceed 88 characters
- [ ] Single backticks for variable/prop names in prose, not double
- [ ] `{@link}` used for cross-references in prose
- [ ] No Unicode characters anywhere
- [ ] No end-of-sentence punctuation in `//` comments
- [ ] Multi-line `//` comments use banner block format
- [ ] No `/** */` used for non-documentation purposes
- [ ] Typos fixed, descriptions accurate, return types match signatures
