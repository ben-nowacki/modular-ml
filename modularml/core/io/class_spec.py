from __future__ import annotations

from dataclasses import dataclass

from modularml.core.io.serialization_policy import SerializationPolicy, normalize_policy


@dataclass(frozen=True)
class ClassSpec:
    """
    Serializable specification describing how to identify and resolve a class.

    It defines how a class is identified at save time and how it should be
    resolved at load time, such that:
    - A runtime class can be converted into a ClassSpec
    - A ClassSpec can be resolved back into a runtime class (when possible)

    It *does not* describe behaviour, configuration, or state.
    """

    policy: SerializationPolicy

    # Symbolic identity (always present unless policy == STATE_ONLY)
    module: str | None = None
    qualname: str | None = None

    # Optional logical identifier (registry / display / diagnostics)
    key: str | None = None

    # Optional fallback source reference (if policy == PACKAGED)
    source_ref: str | None = None

    def validate(self) -> None:
        object.__setattr__(self, "policy", normalize_policy(self.policy))

        if self.policy is SerializationPolicy.STATE_ONLY:
            if any([self.key, self.module, self.qualname, self.source_ref]):
                raise ValueError("STATE_ONLY ClassSpec must not define class identity fields.")

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
