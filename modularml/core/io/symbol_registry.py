from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from modularml.core.io.serialization_policy import SerializationPolicy

if TYPE_CHECKING:
    from types import ModuleType

    from modularml.core.io.symbol_spec import SymbolSpec


class SymbolResolutionError(RuntimeError):
    """Raised when a SymbolSpec cannot be resolved into a class."""


class SymbolRegistry:
    """
    Central registry for resolving class identity during (de)serialization.

    Responsibilities:
    - Convert runtime classes to SymbolSpec (identify)
    - Convert SymbolSpec to runtime classes (resolve)
    - Enforce SerializationPolicy semantics
    """

    # ================================================
    # Construction
    # ================================================
    def __init__(self):
        self._builtin_registry: dict[str, type] = {}
        self._registered_registry: dict[str, type] = {}

    # ================================================
    # Registration APIs
    # ================================================
    def register_builtin(self, key: str, cls: type) -> None:
        """
        Register a built-in ModularML class.

        Args:
            key: Stable registry key (e.g. "FeatureSet").
            cls: Class object.

        """
        if key in self._builtin_registry:
            msg = f"Builtin key '{key}' already registered."
            raise ValueError(msg)
        self._builtin_registry[key] = cls

    def register_registered(self, cls: type) -> None:
        """
        Register a user-defined runtime class (REGISTERED policy).

        The class is identified by module + qualname and is only resolvable
        within the same Python runtime.
        """
        key = self._make_runtime_key(cls)
        if key in self._registered_registry:
            msg = f"Runtime key '{key}' already registered."
            raise ValueError(msg)
        self._registered_registry[key] = cls

    def clear_registered(self) -> None:
        """Clear all REGISTERED classes (useful for tests / notebooks)."""
        self._registered_registry.clear()

    # # ================================================
    # # Identify: class to SymbolSpec
    # # ================================================
    # def identify_class(
    #     self,
    #     cls: type,
    #     *,
    #     policy: SerializationPolicy,
    #     source_ref: str | None = None,
    #     key: str | None = None,
    # ) -> ClassSpec:
    #     """
    #     Convert a runtime class into a ClassSpec according to policy.

    #     Args:
    #         cls: Runtime class object.
    #         policy: SerializationPolicy to apply.
    #         source_ref: Optional bundled source reference (PACKAGED).
    #         key: Optional logical identifier (BUILTIN).

    #     Returns:
    #         ClassSpec

    #     """
    #     policy = normalize_policy(policy)

    #     if policy is SerializationPolicy.STATE_ONLY:
    #         return ClassSpec(policy=policy)

    #     if policy is SerializationPolicy.BUILTIN:
    #         if not key:
    #             raise ValueError("BUILTIN policy requires a registry key.")
    #         return ClassSpec(
    #             policy=policy,
    #             key=key,
    #         )

    #     if policy is SerializationPolicy.REGISTERED:
    #         self.register_registered(cls)
    #         return ClassSpec(
    #             policy=policy,
    #             module=cls.__module__,
    #             qualname=cls.__qualname__,
    #         )

    #     if policy is SerializationPolicy.PACKAGED:
    #         if not source_ref:
    #             raise ValueError("PACKAGED policy requires source_ref.")
    #         return ClassSpec(
    #             policy=policy,
    #             module=cls.__module__,
    #             qualname=cls.__qualname__,
    #             source_ref=source_ref,
    #         )

    #     msg = f"Unsupported SerializationPolicy: {policy}"
    #     raise TypeError(msg)

    # ================================================
    # Resolve: SymbolSpec to class or function
    # ================================================
    def resolve_symbol(
        self,
        spec: SymbolSpec,
        *,
        allow_packaged_code: bool = False,
        packaged_code_loader=None,
    ) -> type:
        """
        Resolve a SymbolSpec into a runtime class or function.

        Args:
            spec: SymbolSpec to resolve.
            allow_packaged_code: Whether executing bundled code is allowed.
            packaged_code_loader: Callable to load bundled source if needed.

        Returns:
            Resolved class.

        Raises:
            SymbolResolutionError

        """
        spec.validate()

        policy = spec.policy

        if policy is SerializationPolicy.STATE_ONLY:
            raise SymbolResolutionError("STATE_ONLY artifacts require user-supplied class.")

        if policy is SerializationPolicy.BUILTIN:
            try:
                return self._builtin_registry[spec.key]
            except KeyError as exc:
                msg = f"Builtin class '{spec.key}' not registered."
                raise SymbolResolutionError(msg) from exc

        if policy is SerializationPolicy.REGISTERED:
            key = f"{spec.module}:{spec.qualname}"
            try:
                return self._registered_registry[key]
            except KeyError as exc:
                msg = f"REGISTERED class '{key}' not found in runtime registry."
                raise SymbolResolutionError(msg) from exc

        if policy is SerializationPolicy.PACKAGED:
            # Step 1: Try normal import
            cls_or_fnc = self._try_import(spec)
            if cls_or_fnc is not None:
                return cls_or_fnc

            # Step 2: Fallback to bundled code
            if not allow_packaged_code:
                msg = (
                    "The serialized file contains packaged code. Do not load this until "
                    "all code has been inspected. Then use `allow_packaged_code=True`."
                )
                raise RuntimeError(msg)
            if spec.source_ref:
                if not packaged_code_loader:
                    raise SymbolResolutionError("Bundled code loader not provided.")
                module = packaged_code_loader(spec.source_ref)
                return self._get_attr(module, spec.qualname)

            msg = f"PACKAGED class '{spec.import_path}' could not be resolved."
            raise SymbolResolutionError(msg)

        msg = f"Unsupported SerializationPolicy: {policy}"
        raise TypeError(msg)

    # ================================================
    # Helpers & Convenience
    # ================================================
    def obj_is_a_builtin(self, obj_or_cls: Any) -> bool:
        """Checks whether an instance is a registered built-in."""
        cls = obj_or_cls if isinstance(obj_or_cls, type) else obj_or_cls.__class__
        return cls in set(self._builtin_registry.values())

    def _make_runtime_key(self, cls: type) -> str:
        return f"{cls.__module__}:{cls.__qualname__}"

    def _try_import(self, spec: SymbolSpec) -> type | None:
        if not spec.import_path:
            return None
        try:
            module = importlib.import_module(spec.module)  # type: ignore[arg-type]
            return SymbolRegistry._get_attr(module, spec.qualname)
        except Exception:  # noqa: BLE001
            return None

    def _get_attr(self, module: ModuleType, qualname: str | None) -> type:
        if not qualname:
            raise SymbolResolutionError("Missing qualname for resolution.")

        obj = module
        for attr in qualname.split("."):
            if isinstance(obj, dict):
                try:
                    obj = obj[attr]
                except KeyError as exc:
                    msg = f"Name '{attr}' not found in packaged code namespace."
                    raise SymbolResolutionError(msg) from exc
            else:
                try:
                    obj = getattr(obj, attr)
                except AttributeError as exc:
                    msg = f"Attribute '{attr}' not found while resolving '{qualname}'."
                    raise SymbolResolutionError(msg) from exc
        return obj


symbol_registry = SymbolRegistry()
