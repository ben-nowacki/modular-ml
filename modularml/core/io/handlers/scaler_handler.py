from __future__ import annotations

from dataclasses import asdict
from typing import Any

from modularml.core.io.class_registry import class_registry
from modularml.core.io.class_spec import ClassSpec
from modularml.core.io.handlers.handler import LoadContext, SaveContext, TypeHandler
from modularml.core.io.serialization_policy import SerializationPolicy
from modularml.core.transforms.scaler import Scaler
from modularml.core.transforms.scaler_registry import SCALER_REGISTRY


class ScalerHandler(TypeHandler[Scaler]):
    """
    TypeHandler for Scaler objects.

    Description:
        Encodes Scaler configuration via Configurable and stores learned
        parameters in JSON form.
    """

    def encode_config(self, obj: Scaler, *, ctx: SaveContext | None = None) -> dict[str, Any]:
        # JSON-safe config from Scaler class
        config = obj.get_config()

        # If using custom scaler (and policy==Package), we need to update the config
        if obj.scaler_name in SCALER_REGISTRY:
            config["impl_kind"] = "registry"
            config["impl_name"] = obj.scaler_name
        else:
            if ctx is None:
                msg = "SaveContext must be provided to handler when using PACKAGED serialization."
                raise RuntimeError(msg)

            impl_spec = ctx.serializer.make_class_spec(
                cls=obj._scaler.__class__,
                policy=SerializationPolicy.PACKAGED,
                artifact_path=ctx.artifact_path,
            )

            config["impl_kind"] = "packaged"
            config["impl_class"] = asdict(impl_spec)

        return config

    def decode_config(
        self,
        cls: type[Scaler],
        config: dict[str, Any],
        ctx: LoadContext | None = None,
    ) -> Scaler:
        impl_kind = config["impl_kind"]

        # Case 1: registry-backed scaler
        if impl_kind == "registry":
            impl_name = config["impl_name"]
            impl_cls = SCALER_REGISTRY[impl_name]
            impl = impl_cls(**config.get("scaler_kwargs", {}))
            return cls(impl)

        # Case 2: packaged custom scaler
        if impl_kind == "packaged":
            if ctx is None:
                msg = "LoadContext must be provided to handler when using PACKAGED deserialization."
                raise RuntimeError(msg)

            impl_spec = ClassSpec(**config["impl_class"])
            impl_cls = class_registry.resolve_class(
                impl_spec,
                allow_packaged_code=True,
                packaged_code_loader=lambda source_ref: ctx.packaged_code_loader(
                    artifact_path=ctx.artifact_path,
                    source_ref=source_ref,
                    allow_packaged=ctx.allow_packaged_code,
                ),
            )
            impl = impl_cls(**config.get("scaler_kwargs", {}))
            return cls(impl)

        msg = f"Unknown impl_kind: {impl_kind}"
        raise ValueError(msg)

    def encode_state(self, obj: Scaler, state_dir: str) -> dict[str, Any] | None:
        if not obj.get_state().get("is_fit", False):
            return None

        return self._encode_state_pickle(obj=obj, state_dir=state_dir)

    def decode_state(
        self,
        obj: Scaler,
        state_dir: str,
        state_spec: dict[str, Any],
    ) -> None:
        return self._decode_state_pickle(
            obj=obj,
            state_dir=state_dir,
            state_spec=state_spec,
        )
