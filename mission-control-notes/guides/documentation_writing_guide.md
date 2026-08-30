# Documentation Writing Guide

This guide applies to all `.md` and `.html` documentation files in the AmpWell codebase.
Read it before creating or significantly editing any documentation file.

---

## 1. Know Your Audience

The single most important decision before writing any documentation is who will read it.
Write for that reader, not for yourself. Every rule in this guide is secondary to this one.

AmpWell has three documentation audiences. Identify which one (or which combination)
applies to your document before writing a single word.

---

### 1.1 User-Facing Documentation

**Who reads it:** End users of the AmpWell platform - lab technicians, researchers, and
engineers who log in and use the software to run battery tests. They understand their
domain (battery testing, electrochemistry) but are not software developers.

**What they need:** Practical steps that get them to a result. Clear explanations of what
the interface shows them and what they can do with it. Honest descriptions of what happens
when things go wrong and how to recover.

**What they do not need:** Database schemas, API routes, authentication mechanisms, code
snippets, internal architecture decisions, or any information about how the software works
underneath. These details create noise and erode trust.

**Writing rules for user docs:**
- Lead with the outcome, not the mechanism ("To start a test run..." not "The protocol
  execution system allows users to...")
- Use the exact label names shown in the UI, formatted in bold (click **Start Run**, not
  "click the button that initiates execution")
- Numbered steps for any sequence of actions - never prose-only instructions
- Screenshots or diagrams wherever a UI element is described
- One concept per section - do not combine "how to register equipment" and "how to edit
  equipment" in the same section
- If something can go wrong, say so - include a brief "If you see X, do Y" note
- Write at a reading level that assumes no programming knowledge
- Never use jargon without explaining it the first time it appears

**Example - good:**

> **To register a new cycler:**
>
> 1. Navigate to **Equipment** in the left sidebar.
> 2. Click **Add Equipment** in the top-right corner.
> 3. Enter the cycler's serial number - you can find this on the label on the back of
>    the unit.
> 4. Click **Save**. The cycler will appear in your equipment list within a few seconds.
>
> If the cycler does not appear after 30 seconds, check that the Bridge Agent is running
> on the connected PC (the tray icon should be green).

**Example - bad (do not write like this for users):**

> Equipment registration is handled via a POST request to `/api/equipment/register`,
> which validates the HMAC signature and stores the device record in PostgreSQL.
> The Bridge Agent polls the long-poll endpoint to receive the registration acknowledgment.

---

### 1.2 Developer Documentation

**Who reads it:** Software engineers working on the AmpWell codebase - including Claude
Code agents, contractors, and future team members. They understand code, architecture,
and technical tradeoffs.

**What they need:** Complete technical detail. Every relevant parameter, edge case, failure
mode, data shape, and design decision. References to the actual code. Explanations of
*why* a decision was made, not just what was decided.

**What they do not need:** Simplified analogies, motivation for why the software exists,
or business context (unless it explains a technical constraint).

**Writing rules for developer docs:**
- Include actual code examples for any non-obvious pattern
- Specify types, not just names ("returns a `dict[str, str]`", not "returns a dictionary")
- Document failure modes explicitly - what exceptions are raised, what HTTP status codes
  are returned, what happens on timeout
- Link to the relevant source file or function when referencing implementation details
- Explain the *why* behind non-obvious decisions ("we use soft deletes here because...")
- Use precise technical vocabulary without defining common terms (you can say "idempotent"
  without explaining what it means)
- Keep examples runnable - code snippets should work if copy-pasted into the right context
- Version or date-stamp any information that is likely to change

---

### 1.3 Management and IT Documentation

**Who reads it:** Non-technical stakeholders, IT administrators, compliance officers,
and security reviewers. They understand organizational risk, infrastructure decisions,
and regulatory requirements, but do not read code.

**What they need:** Clear statements of what the system does, what data it handles, what
access controls are in place, what the risks are, and what procedures exist for auditing
and incident response. Outcome-level descriptions, not implementation details.

**What they do not need:** Code snippets, algorithm descriptions, database schemas, or
internal architecture diagrams. These create confusion and imply a level of review the
reader is not equipped to perform.

**Writing rules for management/IT docs:**
- State facts plainly - "All data is encrypted in transit using TLS 1.2 or higher",
  not "We use SSL/TLS with a modern cipher suite configured via Cloudflare"
- Define any technical term the first time it appears ("Bridge Agent - the software
  installed on each lab PC that connects cycler hardware to the AmpWell server")
- Use a summary section at the top of longer documents - one paragraph, plain language,
  no jargon
- For security and compliance docs, organize by concern (data handling, access control,
  audit logging, incident response) not by technical component
- Quantify where possible ("access tokens expire after 15 minutes", "audit logs are
  retained for 90 days")
- Avoid hedging language - say "passwords are hashed using bcrypt", not "passwords are
  stored using a strong hashing algorithm"

---

## 2. Markdown Formatting Rules

These rules apply to all `.md` files regardless of audience.

### 2.1 Prohibited characters and symbols

Never use the following in any markdown or HTML documentation file:

- Em dash (`-`) - use a plain hyphen (`-`) or rewrite the sentence
- En dash (`-`) - use a plain hyphen
- Section symbol (`§`) - use a hyperlink to the section instead
- Any Unicode symbol not in the 7-bit ASCII range (codepoint above 127)
- Greek letters or math symbols - spell them out or use ASCII equivalents
- The right arrow symbol - use `-->` if an arrow is needed

```markdown
<!-- WRONG -->
See §3 for details.
The timeout — configurable per deployment — defaults to 25 seconds.
Voltage range: 2.5 → 4.2 V

<!-- CORRECT -->
See [Configuration](#configuration) for details.
The timeout (configurable per deployment) defaults to 25 seconds.
Voltage range: 2.5 --> 4.2 V
```

### 2.2 Section references

Never use `§` or bare section numbers to reference another part of a document.
Always use a markdown hyperlink to the heading anchor.

```markdown
<!-- WRONG -->
See §4.2 for migration rules.
See section 4 for details.

<!-- CORRECT -->
See [Migration Rules](#migration-rules) for details.
```

To find the anchor for a heading, lowercase it, replace spaces with hyphens, and remove
punctuation. For example, `## Error Handling Patterns` becomes `#error-handling-patterns`.

### 2.3 Headings

- Use `##` for top-level sections within a document (reserve `#` for the document title only)
- Use `###` for subsections and `####` for sub-subsections
- Do not skip heading levels (no jumping from `##` to `####`)
- Headings are sentence case, not title case ("Error handling patterns", not "Error
  Handling Patterns") - except for proper nouns and product names

### 2.4 Lists

- Use unordered lists (`-`) for items with no meaningful order
- Use ordered lists (`1.`) for sequential steps where order matters
- Do not use lists for fewer than three items - write a sentence instead
- Every list item starts with a capital letter
- List items do not end with a period unless they are complete sentences

### 2.5 Code blocks

Always specify the language on fenced code blocks:

````markdown
```python
async def poll(equipment_id: str) -> PollResponse:
```

```typescript
const response = await apiClient.get<Equipment>('/api/equipment');
```

```bash
pytest backend/tests/ -x -q
```

```json
{
  "equipment_id": "abc123",
  "command": "start_protocol"
}
```
````

For inline code (variable names, file paths, command names), use single backticks:
`equipment_id`, `docs/guides/`, `pytest`.

### 2.6 Links

- Use descriptive link text - never "click here" or "this link"
- Internal links use relative paths: `[Migration Guide](../guides/alembic_migration_guide.md)`
- Section links use heading anchors: `[Error Handling](#error-handling)`
- External links include the full URL

### 2.7 Emphasis

- Use `**bold**` for UI element labels, key terms on first use, and critical warnings
- Use `*italic*` sparingly - for titles of external documents, or light emphasis where
  bold would be too strong
- Do not use bold or italic for decoration or general interest

### 2.8 Tables

Use tables for structured comparisons, reference data with multiple attributes, or
any content where alignment adds meaning. Do not use tables for simple lists.

Always include a header row. Align columns for readability in the raw markdown source:

```markdown
| Property | Type | Description |
|---|---|---|
| equipment_id | string | UUID of the equipment record |
| status | string | Current connection status |
```

### 2.9 Line length

Wrap markdown prose at **88 characters** per line. This makes diffs readable and keeps
the raw source navigable. Code blocks are exempt - do not break code at 88 characters.

---

## 3. HTML Documentation Rules

These rules apply to `.html` documentation files.

- Use semantic HTML elements (`<article>`, `<section>`, `<nav>`, `<header>`, `<main>`)
  rather than generic `<div>` containers for document structure
- Every `<section>` must have an `id` attribute matching its heading text (lowercase,
  hyphenated) so it can be deep-linked
- All cross-references within the document use `<a href="#section-id">` anchor links -
  never bare section numbers or the `§` symbol
- Code examples use `<pre><code class="language-python">` (or the appropriate language)
  so syntax highlighting can be applied
- No inline styles - use a linked stylesheet or utility classes
- All images must have descriptive `alt` text
- The same prohibited characters from [Section 2.1](#prohibited-characters-and-symbols)
  apply to HTML files

---

## 4. Document Structure

Every documentation file should follow this structure:

```
# Document Title

One-paragraph summary of what this document covers and who it is for.
Keep it to three sentences maximum.

---

## [First section]

...

## [Second section]

...
```

Longer documents (more than five sections) should include a table of contents after the
summary paragraph:

```markdown
## Contents

- [Section One](#section-one)
- [Section Two](#section-two)
- [Section Three](#section-three)
```

---

## 5. Tone and Voice

**Be direct.** Say what the system does, not what it "aims to" do or "is designed to" do.

**Use active voice.** "The Bridge Agent sends a heartbeat every 30 seconds" not "A
heartbeat is sent by the Bridge Agent every 30 seconds."

**Be specific.** "Tokens expire after 15 minutes" not "Tokens expire quickly." "The
request times out after 25 seconds" not "The request may time out."

**Do not hedge unless uncertainty is real.** If you do not know a value, say so and
mark it for follow-up rather than writing "approximately" or "usually."

**Do not editorialize.** Documentation states facts. It does not describe the software
as "powerful", "robust", "seamless", or "intuitive."

---

## 6. What to Document

When in doubt about whether something deserves documentation, use these tests:

**For user docs:** Would a new lab technician need to look this up during their first
week? If yes, document it.

**For developer docs:** Would a new engineer make a wrong assumption about this if it
were not written down? If yes, document it. Would a Claude Code agent working on a
related task get this wrong without the context? If yes, document it.

**For management/IT docs:** Would a security auditor or compliance reviewer need to see
this in writing? If yes, document it.

Do not document things that are self-evident from the code, the UI labels, or general
programming knowledge. Documentation that restates the obvious creates noise and makes
the genuinely important parts harder to find.
