from __future__ import annotations

from dataclasses import dataclass

from modularml.core.io.serialization_policy import SerializationPolicy, normalize_policy


@dataclass(frozen=True)
class SymbolSpec:
    """
    Serializable specification describing how to identify and resolve a Python symbol (class or function).

    It defines how a class/function is identified at save time and how it
    should be resolved at load time, such that:
    - A runtime class/fnc can be converted into a SymbolSpec
    - A SymbolSpec can be resolved back into a runtime class/fnc

    It *does not* describe behaviour, configuration, or state.
    """

    policy: SerializationPolicy

    # Symbolic identity (always present unless STATE_ONLY)
    module: str | None = None
    qualname: str | None = None

    # Optional logical identifier (registry / display / diagnostics)
    key: str | None = None

    # Optional fallback source reference (PACKAGED only)
    # Format: "code/<file>.py:<qualname>"
    source_ref: str | None = None

    def validate(self) -> None:
        object.__setattr__(self, "policy", normalize_policy(self.policy))

        if self.policy is SerializationPolicy.STATE_ONLY:
            if any([self.key, self.module, self.qualname, self.source_ref]):
                raise ValueError("STATE_ONLY SymbolSpec must not define class identity fields.")

        elif self.policy is SerializationPolicy.BUILTIN:
            if not self.key:
                raise ValueError("BUILTIN policy requires a registry key.")

        elif self.policy is SerializationPolicy.REGISTERED:
            if not (self.module and self.qualname):
                raise ValueError("REGISTERED policy requires module and qualname.")

        elif self.policy is SerializationPolicy.PACKAGED:
            if not self.source_ref:
                raise ValueError("PACKAGED policy requires a source_ref.")

        else:
            msg = f"Unsupported policy: {self.policy}"
            raise TypeError(msg)

    # ================================================
    # Convenience methods/properties
    # ================================================
    @property
    def import_path(self) -> str | None:
        """
        Fully qualified import path if available.

        Returns:
            str | None: Dotted import path (module.qualname) or None.

        """
        if self.module and self.qualname:
            return f"{self.module}.{self.qualname}"
        return None
