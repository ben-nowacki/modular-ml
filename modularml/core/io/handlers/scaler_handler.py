from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from modularml.core.io.handlers.handler import TypeHandler
from modularml.core.io.packaged_code_loaders.default_loader import default_packaged_code_loader
from modularml.core.io.symbol_registry import symbol_registry
from modularml.core.io.symbol_spec import SymbolSpec
from modularml.core.transforms.registry import SCALER_REGISTRY
from modularml.core.transforms.scaler import Scaler

if TYPE_CHECKING:
    from modularml.core.io.serializer import LoadContext, SaveContext


class ScalerHandler(TypeHandler[Scaler]):
    """
    TypeHandler for Scaler objects.

    Description:
        Encodes Scaler configuration via Configurable and stores learned
        parameters in JSON form.
    """

    object_version: ClassVar[str] = "1.0"

    config_rel_path = "config.json"
    state_rel_path = "state.pkl"

    # ================================================
    # Scaler encoding
    # ================================================
    def encode(
        self,
        obj: Scaler,
        save_dir: Path,
        *,
        ctx: SaveContext,
    ) -> dict[str, str | None]:
        """
        Encodes state and config to files.

        Args:
            obj (Scaler):
                Object instance to encode.
            save_dir (Path):
                Directory to write encodings to.
            ctx (SaveContext, optional):
                Additional serialization context.

        Returns:
            dict[str, str | None]: Mapping of "config" and "state" keys to saved files.

        """
        file_mapping = self.encode_config(obj=obj, save_dir=save_dir, ctx=ctx)
        file_mapping.update(self.encode_state(obj=obj, save_dir=save_dir, ctx=ctx))
        return file_mapping

    def encode_config(
        self,
        obj: Scaler,
        save_dir: Path,
        *,
        ctx: SaveContext,
    ) -> dict[str, str]:
        """
        Encodes config to a json file.

        Args:
            obj (Scaler):
                Object to encode config for.
            save_dir (Path):
                Directory to write encodings to.
            ctx (SaveContext, optional):
                Additional serialization context.

        Returns:
            dict[str, str]: Mapping of config to saved json file

        """
        if not hasattr(obj, "get_config"):
            raise NotImplementedError("Scaler must implement a `get_config` method.")

        # JSON-safe config from Scaler class
        config = obj.get_config()

        # If using custom scaler (and policy==Package), we need to update the config
        if obj.scaler_name in SCALER_REGISTRY:
            config["impl_kind"] = "registry"
            config["impl_name"] = obj.scaler_name
        else:
            impl_spec = ctx.package_symbol(obj._scaler.__class__)

            config["impl_kind"] = "packaged"
            config["impl_class"] = asdict(impl_spec)

        # Save config to file
        with (Path(save_dir) / self.config_rel_path).open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)

        return {"config": self.config_rel_path}

    def encode_state(
        self,
        obj: Scaler,
        save_dir: Path,
        *,
        ctx: SaveContext,
    ) -> dict[str, str]:
        """
        Encodes object state to a pickle file.

        Args:
            obj (Scaler):
                Object to encode state for.
            save_dir (Path):
                Directory to write encodings to.
            ctx (SaveContext, optional):
                Additional serialization context.

        Returns:
            dict[str, str]: Mapping of state to saved pkl file

        """
        return super().encode_state(obj=obj, save_dir=save_dir, ctx=ctx, state_rel_path=self.state_rel_path)

    # ================================================
    # Scaler decoding
    # ================================================
    def decode(
        self,
        cls: type[Scaler],
        load_dir: Path,
        *,
        ctx: LoadContext,
    ) -> Scaler:
        """
        Decodes an object from a saved artifact.

        Description:
            Instantiates an object (instantiates from config and sets state).

        Args:
            cls (type[Scaler]):
                Load config for class.
            load_dir (Path):
                Directory to decode from.
            ctx (LoadContext, optional):
                Additional de-serialization context.

        Returns:
            Scaler: The re-instantiated object.

        """
        config: dict[str, Any] = self.decode_config(load_dir=load_dir, ctx=ctx)
        state: dict[str, Any] = self.decode_state(load_dir=load_dir, ctx=ctx)

        # ================================================
        # Instantiate Scaler from config
        # ================================================
        scaler_obj = None
        impl_kind = config["impl_kind"]

        # Case 1: registry-backed scaler
        if impl_kind == "registry":
            impl_name = config["impl_name"]
            impl_cls = SCALER_REGISTRY[impl_name]
            impl = impl_cls(**config.get("scaler_kwargs", {}))
            scaler_obj = cls(impl)

        # Case 2: packaged custom scaler
        elif impl_kind == "packaged":
            if ctx is None:
                msg = "LoadContext must be provided to handler when using PACKAGED deserialization."
                raise RuntimeError(msg)

            impl_spec = SymbolSpec(**config["impl_class"])
            impl_cls = symbol_registry.resolve_symbol(
                impl_spec,
                allow_packaged_code=ctx.allow_packaged_code,
                packaged_code_loader=lambda source_ref: default_packaged_code_loader(
                    artifact_path=ctx.artifact_path,
                    source_ref=source_ref,
                    allow_packaged=ctx.allow_packaged_code,
                ),
            )
            impl = impl_cls(**config.get("scaler_kwargs", {}))
            scaler_obj = cls(impl)

        else:
            msg = f"Unknown impl_kind: {impl_kind}"
            raise ValueError(msg)

        # ================================================
        # Set Scaler state
        # ================================================
        if not hasattr(scaler_obj, "set_state"):
            msg = "Scaler must implement a `set_state` method."
            raise NotImplementedError(msg)

        scaler_obj.set_state(state=state)
        return scaler_obj

    def decode_config(
        self,
        load_dir: Path,
        *,
        ctx: LoadContext,
    ) -> dict[str, Any] | None:
        """
        Decodes config from a json file.

        Args:
            load_dir (Path):
                Directory to decode from.
            ctx (LoadContext, optional):
                Additional de-serialization context.

        Returns:
            dict[str, Any] | None: The decoded config data.

        """
        return super().decode_config(load_dir=load_dir, ctx=ctx, config_rel_path=self.config_rel_path)

    def decode_state(
        self,
        load_dir: Path,
        *,
        ctx: LoadContext,
    ) -> dict[str, Any]:
        """
        Decodes state from a pkl file.

        Args:
            load_dir (Path):
                Directory to decode from.
            ctx (LoadContext, optional):
                Additional de-serialization context.

        Returns:
            dict[str, Any]: The decoded state data.

        """
        return super().decode_state(load_dir=load_dir, ctx=ctx, state_rel_path=self.state_rel_path)
