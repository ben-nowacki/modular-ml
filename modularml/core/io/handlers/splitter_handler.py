from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from modularml.core.io.handlers.handler import LoadContext, SaveContext, TypeHandler
from modularml.core.splitting.base_splitter import BaseSplitter

if TYPE_CHECKING:
    from pathlib import Path


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
        ctx: SaveContext | None = None,
    ) -> dict[str, str | None]:
        """
        Encodes state and config to files.

        Args:
            obj (BaseSplitter):
                BaseSplitter instance to encode.
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
        obj: BaseSplitter,
        save_dir: Path,
        *,
        ctx: SaveContext | None = None,
    ) -> dict[str, str]:
        """
        Encodes config to a json file.

        Args:
            obj (BaseSplitter):
                BaseSplitter to encode config for.
            save_dir (Path):
                Parent dir to save 'config.json' file.
            ctx (SaveContext, optional):
                Additional serialization context.
                Strictly required for SerializationPolicy.PACKAGED.

        Returns:
            dict[str, str]: Mapping of config to saved json file

        """
        return self._encode_config_json(
            obj=obj,
            save_dir=save_dir,
            config_rel_path=self.config_rel_path,
            ctx=ctx,
        )

    def encode_state(
        self,
        obj: BaseSplitter,
        save_dir: Path,
        *,
        ctx: SaveContext | None = None,
    ) -> dict[str, str]:
        """
        Encodes BaseSplitter state to a pickle file.

        Args:
            obj (BaseSplitter):
                BaseSplitter to encode state for.
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
    # Splitter Decoding
    # ================================================
    def decode(
        self,
        cls: type[BaseSplitter],
        parent_dir: Path,
        *,
        ctx: LoadContext | None = None,
    ) -> BaseSplitter:
        """
        Decodes a BaseSplitter from a saved artifact.

        Description:
            Instantiates a BaseSplitter (instantiates from config and sets state).

        Args:
            cls (type[BaseSplitter]):
                Load config for BaseSplitter class.
            parent_dir (Path):
                Directory contains a saved 'config.json' file
            ctx (LoadContext, optional):
                Additional de-serialization context.
                Strictly required for SerializationPolicy.PACKAGED.

        Returns:
            BaseSplitter: The re-instantiated splitter.

        """
        config: dict[str, Any] = self.decode_config(config_dir=parent_dir, ctx=ctx)

        # Instantiate BaseSplitter from config
        splitter_obj = cls.from_config(config)

        # Set state (if defined - most don't have a state)
        if hasattr(splitter_obj, "set_state"):
            state: dict[str, Any] = self.decode_state(state_dir=parent_dir, ctx=ctx)
            splitter_obj.set_state(state=state)

        return splitter_obj

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
