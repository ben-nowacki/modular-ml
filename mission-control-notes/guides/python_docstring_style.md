# Python Docstring & Comment Style Guide

Reference: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

This guide applies to all Python files in the AmpWell codebase. Read it before writing
or editing any docstring or inline comment. The rules below are non-negotiable — do not
deviate from them even when existing code does.

---

## 1. Docstring Style

Use **PEP 257 with Google-style docstrings** throughout. No NumPy style, no plain RST.

---

## 2. Module-Level Docstrings

Every `.py` file must have a one-line module docstring at the very top — before any
`from __future__` imports, before any other imports.

```python
"""Async SQLAlchemy engine and session factory for the AmpWell backend."""

from __future__ import annotations

import sqlalchemy as sa
```

One line only. No multi-line module docstrings.

---

## 3. Class Docstrings

All public classes must have a docstring. Required sections:

- **Brief description** — one line, on the line immediately after the opening `"""`.
- **Description:** — optional extended prose block; keep existing ones, add only if needed.
- **Attributes:** — required for dataclasses and any class where instance attributes are
  not obvious from `__init__` signatures. List every public attribute.

```python
class EquipmentDriver:
    """
    Base class for all cycler hardware drivers.

    Description:
        Subclasses implement brand-specific communication protocols while
        exposing a normalised interface to the orchestrator daemon. All
        methods that touch hardware must be async.

    Attributes:
        equipment_id (str): UUID of the registered equipment record.
        channel_count (int): Number of independently controllable channels.
        capabilities (dict): Hardware capability flags from the equipment registry.
    """
```

Multi-line rule: the first line of text starts on a **new line after the opening `"""`**.
The closing `"""` sits on its own line, preceded by a blank line.

```python
# CORRECT
class Foo:
    """
    Brief description.

    Longer paragraph if needed.
    """

# WRONG — first line must not be inline with the opening triple-quote
class Foo:
    """Brief description.

    Longer paragraph if needed.
    """
```

---

## 4. Method and Function Docstrings

All public methods and functions must have docstrings. Required sections:

- **Brief one-line summary** — first line, on the same line as `"""`.
- **Args:** — every parameter, including `self`-excluded ones. Format: `name (type): description`.
- **Returns:** — omit entirely if the function returns `None`. Format: `Type: description`.
- **Raises:** — include only when the function explicitly raises, or when callers must handle
  a specific exception.

```python
def enqueue_command(
    self,
    equipment_id: str,
    command: str,
    params: dict,
) -> str:
    """
    Add a command to the pending queue for a connected device.

    Args:
        equipment_id (str): UUID of the target equipment record.
        command (str): Command name as defined in the Bridge Agent protocol.
        params (dict): Command-specific parameters passed verbatim to the agent.

    Returns:
        str: UUID of the newly created pending command record.

    Raises:
        EquipmentNotFoundError: If no equipment record exists for `equipment_id`.
        EquipmentOfflineError: If the equipment's last heartbeat exceeds the
            offline threshold.
    """
```

Multi-line rule: same as classes — first line of text on a **new line after `"""`**, closing
`"""` on its own line after a blank line.

```python
# CORRECT
def foo(x: int) -> str:
    """
    Convert an integer channel index to its display label.

    Args:
        x (int): Zero-based channel index.

    Returns:
        str: Human-readable label, e.g. "Channel 3".
    """

# WRONG — first line must not be inline with the opening triple-quote
def foo(x: int) -> str:
    """Convert an integer channel index to its display label.

    Args:
        x (int): Zero-based channel index.

    Returns:
        str: Human-readable label, e.g. "Channel 3".
    """

# WRONG — omit Returns when return type is None
def bar() -> None:
    """
    Reset all channel state flags.

    Returns:
        None: This method returns nothing.
    """
```

---

## 5. Sphinx Cross-References

Use Sphinx cross-reference syntax in **docstring prose only** — not inside Args, Returns,
Attributes, or Raises type annotations.

| What you're referencing | Syntax |
|---|---|
| A class | `:class:\`ClassName\`` |
| A method | `:meth:\`method_name\`` |
| An attribute or constant | `:attr:\`attr_name\`` |

```python
def get_driver(self) -> EquipmentDriver:
    """
    Return the :class:`EquipmentDriver` for this channel's equipment.

    Calls :meth:`connect` internally if the driver is not yet initialised.
    The returned driver exposes :attr:`channel_count` for channel validation.
    """
```

Do not use cross-reference syntax inside type annotation strings (e.g. inside Args sections).

---

## 6. Backtick Usage in Prose

Use **single backticks** around variable names, argument names, and constants in prose.
Never double backticks.

```python
# CORRECT
"""Return the value of `DOMAIN_SAMPLE_ID` for the given run."""

# WRONG
"""Return the value of ``DOMAIN_SAMPLE_ID`` for the given run."""
```

---

## 7. Line Length

No line in a docstring may exceed **88 characters**, including the indentation prefix.

For a method indented 4 spaces, the usable content width is 84 characters.
For a method indented 8 spaces (inside a class), it is 80 characters.

Wrap long sentences across multiple lines, aligning continuation at the content column:

```python
    """
    Normalise a raw Arbin timeseries file and write the result to the
    Iceberg timeseries table.

    Args:
        artifact_path (str): Absolute path to the raw `.res` file on the
            local filesystem.
    """
```

---

## 8. ASCII-Only Docstrings

Never use non-standard ASCII characters in docstrings or comments. Specifically forbidden:

- Greek letters (`alpha`, `beta`, `mu`, `sigma`, etc.) — spell them out
- Math symbols (`×`, `≤`, `≥`, `±`, `∑`, `∫`) — use `x`, `<=`, `>=`, `+/-`, `sum`, `integral`
- Em dash (`—`) and en dash (`–`) — use a plain hyphen `-`
- Arrow `→` — use `-->` if an arrow is genuinely needed
- Any Unicode symbol not in the 7-bit ASCII range (codepoint > 127)

```python
# CORRECT
"""Compute the mean +/- standard deviation of discharge capacity."""

# WRONG
"""Compute the mean ± standard deviation of discharge capacity."""
```

---

## 9. Typo and Description Accuracy

Fix any typos encountered while editing a docstring. Common ones in this codebase:

| Wrong | Correct |
|---|---|
| `isntance` | `instance` |
| `specfiied` | `specified` |
| `Wether` | `Whether` |
| `indicies` | `indices` |

Also fix incorrect descriptions — e.g. if a function named `get_targets` has a Returns
section that says "Feature data", change it to "Target data". Don't leave wrong
descriptions in place even if the rest of the docstring is fine.

---

## 10. Minimal Rewriting

Do not rewrite existing correct descriptions. The goal is to add what is missing and
fix what is wrong — not to rephrase for style. If an existing Args entry is accurate,
leave its wording alone even if you would have phrased it differently.

---

## 11. Type Annotation Fixes

While updating docstrings, fix any incorrect return type annotations in the function
signature if you notice them. Examples:

```python
# Wrong signature — should return dict[str, str], not dict[str, tuple[int, ...]]
def get_column_mapping(self) -> dict[str, tuple[int, ...]]:

# Corrected
def get_column_mapping(self) -> dict[str, str]:
```

Do not change the docstring's Returns section without also fixing the signature, and
vice versa. Keep them in sync.

---

## 12. Comment Style

### General rules

- Comments are never long sentences — no periods or end-of-sentence punctuation
- Concise but descriptive — a reader should understand the *why*, not just the *what*
- No Unicode characters (same rule as docstrings)

### Inline comments

Single-thought inline comments sit on the same line, separated by two spaces:

```python
timeout = 25  # long-poll window in seconds
retries = 0   # reset on successful poll
```

### Short block comments

For a single thought that needs its own line, use a plain `#` prefix:

```python
# Sign and attach HMAC headers before sending
headers = security.sign_request(self.equipment_id)
response = requests.post(url, json=payload, headers=headers)
```

### Multi-line block comments

When a comment requires multiple lines or explains a non-obvious design decision,
use the banner block format. Structure:

- Opening and closing lines: `# ===...===` (match length to content width)
- First content line: main descriptive phrase (no period)
- Subsequent lines: bullet list with `-` for each sub-point
- Any code following the closing `# ===...===` line should not have a blank line between it

```python
# ==========================================================================
# Exit rule conditions
# - Outer list = OR, inner list = AND (mirrors schema: list[list[ValueExpr]])
# - Each UICondition serialises to a ValueExpr {expr: "metric comparator rhs"}
# ==========================================================================
exit_rules: list[list[UICondition]] = []
```

```python
# ==========================================================================
# Long-poll loop behaviour
# - Blocks for up to LONG_POLL_TIMEOUT_SECONDS waiting for a command
# - Returns immediately when a command is enqueued (asyncio.Event.set)
# - Empty response ({commands: []}) when timeout expires with no commands
# ==========================================================================
async def poll(equipment_id: str, timeout: int = 25) -> PollResponse:
```

Keep banner widths consistent within a file. 74 characters total (`# ` + 72 `=`) is
the default. Adjust shorter for deeply indented code.

### What not to do

```python
# WRONG - end-of-sentence punctuation
# Sign and attach HMAC headers before sending.

# WRONG - restates the code without adding meaning
result = db.query(Equipment)  # query the Equipment table

# WRONG - Unicode in comment
# Voltage range: 2.5 --> 4.2 V (avoid --> is fine, but no → symbol)

# CORRECT - explains the why, no punctuation
# skip if equipment went offline between poll and command dispatch
if not equipment.is_connected:
    continue
    
# WRONG - use the `=` dividers, not these single lines
# ── Claude API call ──────────────────

# CORRECT - uses `=` for start and end line dividers; no blank line between function and '=' line
# ==========================================================================
# Claude API call
# ==========================================================================
def some_fnc():
    pass
```

---

## Examples

### Function with args, return value, and raises

```python
def get_sample_ids(domain: str, limit: int = 100) -> dict[str, str]:
    """
    Retrieve sample IDs for a given domain.

    Args:
        domain (str): The domain name to query. Must match a value in
            :attr:`VALID_DOMAINS`.
        limit (int): Maximum number of results to return. Defaults to 100.

    Returns:
        dict[str, str]: A mapping of sample ID to display label.

    Raises:
        ValueError: If `domain` is not a recognised domain name.
    """
```

### Method that returns None (omit Returns section)

```python
def reset_state(self, clear_cache: bool = False) -> None:
    """
    Reset the internal state of this :class:`Pipeline` instance.

    Args:
        clear_cache (bool): If ``True``, also clears the on-disk cache.
            Defaults to False.
    """
```


# Comment Standard

Blocks of distinct logic (eg, class definitions or long functional blocks) should  have header comments like the following:

```python
# ================================================
# {title of section of code}
# - {optional details of this sections}
# - {more optional details}
# ================================================
```

All other comments should try to be 1 line (no trailing period).
If more details are needed than can fit into one line, I prefer the following format:

```python
# {primary comment}
# - {optional additional details of this comment}
# - {more optional details}
```

---

## Quick Reference Checklist

Use this when reviewing a file before committing:

- [ ] Module docstring present at top of file (one line)
- [ ] All public classes have docstrings with Attributes section where needed
- [ ] All public methods have docstrings with Args and Returns (omit Returns if `-> None`)
- [ ] Multi-line docstrings: first line on same line as `"""`, blank line before closing `"""`
- [ ] No lines exceed 88 characters
- [ ] Single backticks for variables in prose, not double
- [ ] Sphinx cross-refs used in prose (`:class:`, `:meth:`, `:attr:`)
- [ ] No Unicode characters anywhere
- [ ] No end-of-sentence punctuation in comments
- [ ] Multi-line comments use banner block format
- [ ] Typos fixed, descriptions accurate, return types match signatures
