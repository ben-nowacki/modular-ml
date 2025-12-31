from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from modularml.core.io.handlers.handler import TypeHandler
from modularml.core.splitting.base_splitter import BaseSplitter

if TYPE_CHECKING:
    from pathlib import Path

    from modularml.core.io.serializer import LoadContext, SaveContext


class SplitterHandler(TypeHandler[BaseSplitter]):
    """
    TypeHandler for BaseSplitter objects.

    Description:
        Encodes splitter configuration and state (if defined).
        Supports both registry-backed and packaged custom splitters.
    """

    object_version: ClassVar[str] = "1.0"

    config_rel_path = "config.json"
    state_rel_path = "state.pkl"

    # ================================================
    # Splitter Encoding
    # ================================================
    def encode(
        self,
        obj: BaseSplitter,
        save_dir: Path,
        *,
        ctx: SaveContext,
    ) -> dict[str, str | None]:
        """
        Encodes state and config to files.

        Args:
            obj (BaseSplitter):
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
        obj: BaseSplitter,
        save_dir: Path,
        *,
        ctx: SaveContext,
    ) -> dict[str, str]:
        """
        Encodes config to a json file.

        Args:
            obj (BaseSplitter):
                Object to encode config for.
            save_dir (Path):
                Directory to write encodings to.
            ctx (SaveContext, optional):
                Additional serialization context.

        Returns:
            dict[str, str]: Mapping of config to saved json file

        """
        return super().encode_config(obj=obj, save_dir=save_dir, ctx=ctx, config_rel_path=self.config_rel_path)

    def encode_state(
        self,
        obj: BaseSplitter,
        save_dir: Path,
        *,
        ctx: SaveContext,
    ) -> dict[str, str]:
        """
        Encodes object state to a pickle file.

        Args:
            obj (BaseSplitter):
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
    # Splitter Decoding
    # ================================================
    def decode(
        self,
        cls: type[BaseSplitter],
        load_dir: Path,
        *,
        ctx: LoadContext,
    ) -> BaseSplitter:
        """
        Decodes an object from a saved artifact.

        Description:
            Instantiates an object (instantiates from config and sets state).

        Args:
            cls (type[BaseSplitter]):
                Load config for class.
            load_dir (Path):
                Directory to decode from.
            ctx (LoadContext, optional):
                Additional de-serialization context.

        Returns:
            BaseSplitter: The re-instantiated object.

        """
        config: dict[str, Any] = self.decode_config(load_dir=load_dir, ctx=ctx)

        # Instantiate BaseSplitter from config
        splitter_obj = cls.from_config(config)

        # Set state (if defined - most don't have a state)
        if hasattr(splitter_obj, "set_state"):
            state: dict[str, Any] = self.decode_state(load_dir=load_dir, ctx=ctx)
            splitter_obj.set_state(state=state)

        return splitter_obj

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
