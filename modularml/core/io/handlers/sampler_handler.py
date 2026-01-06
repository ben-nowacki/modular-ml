from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from modularml.core.io.handlers.handler import TypeHandler
from modularml.core.io.serialization_policy import SerializationPolicy
from modularml.core.sampling.base_sampler import BaseSampler

if TYPE_CHECKING:
    from modularml.core.io.serializer import LoadContext, SaveContext
    from modularml.core.sampling.similiarity_condition import SimilarityCondition


class SamplerHandler(TypeHandler[BaseSampler]):
    """
    TypeHandler for BaseSampler objects.

    Description:
        Encodes sampler configuration and state (if defined).
        Supports both registry-backed and packaged custom samplers.
    """

    object_version: ClassVar[str] = "1.0"

    config_rel_path = "config.json"
    state_rel_path = "state.pkl"

    # ================================================
    # Sampler Encoding
    # ================================================
    def encode(
        self,
        obj: BaseSampler,
        save_dir: Path,
        *,
        ctx: SaveContext,
    ) -> dict[str, str | None]:
        """
        Encodes state and config to files.

        Args:
            obj (BaseSampler):
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
        obj: BaseSampler,
        save_dir: Path,
        *,
        ctx: SaveContext,
    ) -> dict[str, str]:
        """
        Encodes config to a json file.

        Args:
            obj (BaseSampler):
                Object to encode config for.
            save_dir (Path):
                Directory to write encodings to.
            ctx (SaveContext, optional):
                Additional serialization context.

        Returns:
            dict[str, str]: Mapping of config to saved json file

        """
        if not self.has_config(obj):
            return {"config": None}

        if not hasattr(obj, "get_config"):
            raise NotImplementedError("Object must implement a `get_config` method.")

        config = copy.deepcopy(obj.get_config())

        # If sampler holds SimilarityConditions, we need to serialize those
        if "condition_mapping" in config:
            cond_mapping: dict[str, dict[str, SimilarityCondition]] = config.pop("condition_mapping")

            # Replace SimilarityCondition objects with references to serialized artifacts
            cond_map_refs: dict[str, dict[str, Any]] = {}

            for role, role_map in cond_mapping.items():
                cond_map_refs[role] = {}
                for key, cond in role_map.items():
                    outpath = save_dir / f"conditions/{role}_{key}"
                    symbol_spec = ctx.make_symbol_spec(
                        symbol=cond,
                        policy=SerializationPolicy.BUILTIN,
                        builtin_key="SimilarityCondition",
                    )
                    mml_path = ctx.emit_mml(
                        obj=cond,
                        symbol_spec=symbol_spec,
                        out_path=outpath,
                        overwrite=False,
                    )

                    # Replace SimilarityCondition instance with path to mml artifact
                    cond_mapping[role][key] = str(mml_path.relative_to(save_dir))

            config["condition_mapping"] = cond_mapping

        # Save config to file
        path = self._write_json(config, Path(save_dir) / self.config_rel_path)
        return {"config": path.name}

    def encode_state(
        self,
        obj: BaseSampler,
        save_dir: Path,
        *,
        ctx: SaveContext,
    ) -> dict[str, str]:
        """
        Encodes object state to a pickle file.

        Args:
            obj (BaseSampler):
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
    # Sampler Decoding
    # ================================================
    def decode(
        self,
        cls: type[BaseSampler],
        load_dir: Path,
        *,
        ctx: LoadContext,
    ) -> BaseSampler:
        """
        Decodes an object from a saved artifact.

        Description:
            Instantiates an object (instantiates from config and sets state).

        Args:
            cls (type[BaseSampler]):
                Load config for class.
            load_dir (Path):
                Directory to decode from.
            ctx (LoadContext, optional):
                Additional de-serialization context.

        Returns:
            BaseSampler: The re-instantiated object.

        """
        config: dict[str, Any] = self.decode_config(load_dir=load_dir, ctx=ctx)

        # Instantiate BaseSampler from config
        sampler_obj = cls.from_config(config)

        # Set state (if defined - most don't have a state)
        if hasattr(sampler_obj, "set_state"):
            state: dict[str, Any] = self.decode_state(load_dir=load_dir, ctx=ctx)
            sampler_obj.set_state(state=state)

        return sampler_obj

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
        config = super().decode_config(load_dir=load_dir, ctx=ctx, config_rel_path=self.config_rel_path)

        # If sampler holds similarity conditions, we need to de-serialize those separately
        if "condition_mapping" in config:
            cond_mapping = {}
            for role, role_map in config["condition_mapping"].items():
                cond_mapping[role] = {}
                for key, ref in role_map.items():
                    cond: SimilarityCondition = ctx.load_from_mml(mml_file=Path(load_dir) / ref)
                    cond_mapping[role][key] = cond

            config["condition_mapping"] = cond_mapping

        return config

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
