from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from modularml.core.io.handlers.handler import TypeHandler
from modularml.core.io.packaged_code_loaders.default_loader import default_packaged_code_loader
from modularml.core.io.serialization_policy import SerializationPolicy
from modularml.core.io.symbol_registry import symbol_registry
from modularml.core.io.symbol_spec import SymbolSpec
from modularml.core.training.loss import Loss

if TYPE_CHECKING:
    from modularml.core.io.serializer import LoadContext, SaveContext


class LossHandler(TypeHandler[Loss]):
    """
    TypeHandler for Loss objects.

    Description:
        Encodes Loss configuration and internal state.
    """

    object_version: ClassVar[str] = "1.0"

    config_rel_path = "config.json"
    state_rel_path = "state.pkl"

    # ================================================
    # Encoding
    # ================================================
    def encode(
        self,
        obj: Loss,
        save_dir: Path,
        *,
        ctx: SaveContext,
    ) -> dict[str, str | None]:
        """
        Encodes state and config to files.

        Args:
            obj (Loss):
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
        obj: Loss,
        save_dir: Path,
        *,
        ctx: SaveContext,
    ) -> dict[str, str]:
        """
        Encodes config to a json file.

        Args:
            obj (Loss):
                Object to encode config for.
            save_dir (Path):
                Directory to write encodings to.
            ctx (SaveContext, optional):
                Additional serialization context.

        Returns:
            dict[str, str]: Mapping of config to saved json file

        """
        if not hasattr(obj, "get_config"):
            raise NotImplementedError("Loss must implement a `get_config` method.")

        # JSON-safe config from Loss class
        config = obj.get_config()

        # "loss" may be a string, a class, or a callable
        cfg_loss = config.get("loss")
        # If a callable, we need to package it
        if isinstance(cfg_loss, Callable):
            loss_spec = ctx.make_symbol_spec(
                symbol=cfg_loss,
                policy=SerializationPolicy.PACKAGED,
            )
            # Store spec for later reload instructions
            config["loss"] = asdict(loss_spec)

        # We'll also need to packaged factory, if defined
        factory = config.get("factory")
        if factory is not None:
            factory_spec = ctx.make_symbol_spec(
                symbol=factory,
                policy=SerializationPolicy.PACKAGED,
            )
            # Store spec for later reload instructions
            config["factory"] = asdict(factory_spec)

        # Save config to file
        with (Path(save_dir) / self.config_rel_path).open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)

        return {"config": self.config_rel_path}

    def encode_state(
        self,
        obj: Loss,
        save_dir: Path,
        *,
        ctx: SaveContext,
    ) -> dict[str, str]:
        """
        Encodes object state to a pickle file.

        Args:
            obj (Loss):
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
    # Decoding
    # ================================================
    def decode(
        self,
        cls: type[Loss],
        load_dir: Path,
        *,
        ctx: LoadContext,
    ) -> Loss:
        """
        Decodes an object from a saved artifact.

        Description:
            Instantiates an object (instantiates from config and sets state).

        Args:
            cls (type[Loss]):
                Load config for class.
            load_dir (Path):
                Directory to decode from.
            ctx (LoadContext, optional):
                Additional de-serialization context.

        Returns:
            Loss: The re-instantiated object.

        """
        config: dict[str, Any] = self.decode_config(load_dir=load_dir, ctx=ctx)
        state: dict[str, Any] = self.decode_state(load_dir=load_dir, ctx=ctx)

        # Reload loss (may be encoded)
        cfg_loss = config.get("loss")
        if isinstance(cfg_loss, dict):
            loss_fn: Callable = symbol_registry.resolve_symbol(
                spec=SymbolSpec(**cfg_loss),
                allow_packaged_code=ctx.allow_packaged_code,
                packaged_code_loader=lambda source_ref: default_packaged_code_loader(
                    artifact_path=ctx.artifact_path,
                    source_ref=source_ref,
                    allow_packaged=ctx.allow_packaged_code,
                ),
            )
            config["loss"] = loss_fn

        # Reload factory (if encoded)
        factory_spec = config.get("factory")
        if factory_spec is not None:
            factory: Callable = symbol_registry.resolve_symbol(
                spec=SymbolSpec(**factory_spec),
                allow_packaged_code=ctx.allow_packaged_code,
                packaged_code_loader=lambda source_ref: default_packaged_code_loader(
                    artifact_path=ctx.artifact_path,
                    source_ref=source_ref,
                    allow_packaged=ctx.allow_packaged_code,
                ),
            )
            config["factory"] = factory

        # Create Loss instance
        loss: Loss = cls.from_config(config=config)

        # Set Loss state
        if hasattr(loss, "set_state"):
            loss.set_state(state=state)

        return loss

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
        rel_path = self.config_rel_path
        if ctx.file_mapping is not None:
            rel_path = ctx.file_mapping.get("config")
            if rel_path is None:
                return {}
        return super().decode_config(load_dir=load_dir, ctx=ctx, config_rel_path=rel_path)

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
        rel_path = self.config_rel_path
        if ctx.file_mapping is not None:
            rel_path = ctx.file_mapping.get("state")
            if rel_path is None:
                return {}
        return super().decode_state(load_dir=load_dir, ctx=ctx, state_rel_path=rel_path)
