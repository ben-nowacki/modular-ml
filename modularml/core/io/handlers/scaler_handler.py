from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, ClassVar

from modularml.core.io.class_registry import class_registry
from modularml.core.io.class_spec import ClassSpec
from modularml.core.io.handlers.handler import LoadContext, SaveContext, TypeHandler
from modularml.core.transforms.scaler import Scaler
from modularml.core.transforms.scaler_registry import SCALER_REGISTRY


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
        ctx: SaveContext | None = None,
    ) -> dict[str, str | None]:
        """
        Encodes state and config to files.

        Args:
            obj (Scaler):
                Scaler instance to encode.
            save_dir (Path):
                Parent dir to save config and state files.
            ctx (SaveContext, optional):
                Additional serialization context.
                Strictly required for SerializationPolicy.PACKAGED.

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
        ctx: SaveContext | None = None,
    ) -> dict[str, str]:
        """
        Encodes config to a json file.

        Args:
            obj (Scaler):
                Scaler to encode config for.
            save_dir (Path):
                Parent dir to save 'config.json' file.
            ctx (SaveContext, optional):
                Additional serialization context.
                Strictly required for SerializationPolicy.PACKAGED.

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
            if ctx is None:
                msg = "SaveContext must be provided to handler when using PACKAGED serialization."
                raise RuntimeError(msg)

            impl_spec = ctx.package_class(obj._scaler.__class__)

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
        ctx: SaveContext | None = None,
    ) -> dict[str, str]:
        """
        Encodes Scaler state to a pickle file.

        Args:
            obj (Scaler):
                Scaler to encode state for.
            save_dir (Path):
                Parent dir to save 'state.pkl' file.
            ctx (SaveContext, optional):
                Additional serialization context.
                Strictly required for SerializationPolicy.PACKAGED.

        Returns:
            dict[str, str]: Mapping of state to saved pkl file

        """
        return self._encode_state_pickle(
            obj=obj,
            save_dir=save_dir,
            state_rel_path=self.state_rel_path,
            ctx=ctx,
        )

    # ================================================
    # Scaler decoding
    # ================================================
    def decode(
        self,
        cls: type[Scaler],
        parent_dir: Path,
        *,
        ctx: LoadContext | None = None,
    ) -> Scaler:
        """
        Decodes a Scaler from a saved artifact.

        Description:
            Instantiates a Scaler (instantiates from config and sets state).

        Args:
            cls (type[Scaler]):
                Load config for Scaler class.
            parent_dir (Path):
                Directory contains a saved 'config.json' file
            ctx (LoadContext, optional):
                Additional de-serialization context.
                Strictly required for SerializationPolicy.PACKAGED.

        Returns:
            Scaler: The re-instantiated Scaler.

        """
        config: dict[str, Any] = self.decode_config(config_dir=parent_dir, ctx=ctx)
        state: dict[str, Any] = self.decode_state(state_dir=parent_dir, ctx=ctx)

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
        config_dir: Path,
        *,
        ctx: LoadContext | None = None,
    ) -> dict[str, Any]:
        """
        Decodes config from a json file.

        Args:
            config_dir (Path):
                Directory contains a saved 'config.json' file
            ctx (LoadContext, optional):
                Additional de-serialization context.
                Strictly required for SerializationPolicy.PACKAGED.

        Returns:
            dict[str, Any]: The decoded config data.

        """
        return self._decode_config_json(
            config_dir=config_dir,
            config_rel_path=self.config_rel_path,
            ctx=ctx,
        )

    def decode_state(
        self,
        state_dir: str,
        *,
        ctx: LoadContext | None = None,
    ) -> dict[str, Any]:
        """
        Decodes state from a pkl file.

        Args:
            state_dir (Path):
                Directory containing a saved 'state.pkl' file
            ctx (LoadContext, optional):
                Additional de-serialization context.
                Strictly required for SerializationPolicy.PACKAGED.

        Returns:
            dict[str, Any]: The decoded state data.

        """
        return self._decode_state_pickle(
            state_dir=state_dir,
            state_rel_path=self.state_rel_path,
            ctx=ctx,
        )
