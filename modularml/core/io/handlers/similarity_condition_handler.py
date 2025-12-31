from __future__ import annotations

import inspect
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from modularml.core.io.handlers.handler import TypeHandler
from modularml.core.io.packaged_code_loaders.default_loader import default_packaged_code_loader
from modularml.core.io.symbol_registry import symbol_registry
from modularml.core.io.symbol_spec import SymbolSpec
from modularml.core.sampling.similiarity_condition import SimilarityCondition

if TYPE_CHECKING:
    from modularml.core.io.serializer import LoadContext, SaveContext


class SimilarityConditionHandler(TypeHandler[SimilarityCondition]):
    """
    TypeHandler for SimilarityCondition.

    Handles optional packaging of metric callable.
    """

    object_version: ClassVar[str] = "1.0"
    config_rel_path = "config.json"
    state_rel_path = "state.json"

    # ================================================
    # Encoding
    # ================================================
    def encode(
        self,
        obj: SimilarityCondition,
        save_dir: Path,
        *,
        ctx: SaveContext,
    ) -> dict[str, str | None]:
        """
        Encodes state and config to files.

        Args:
            obj (SimilarityCondition):
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
        obj: SimilarityCondition,
        save_dir: Path,
        *,
        ctx: SaveContext,
    ) -> dict[str, str]:
        """
        Encodes config to a json file.

        Args:
            obj (SimilarityCondition):
                Object to encode config for.
            save_dir (Path):
                Directory to write encodings to.
            ctx (SaveContext, optional):
                Additional serialization context.

        Returns:
            dict[str, str]: Mapping of config to saved json file

        """
        if not hasattr(obj, "get_config"):
            raise NotImplementedError("SimilarityCondition must implement a `get_config` method.")

        config = obj.get_config()
        metric = config.pop("metric")

        # Metric handling
        if metric is None:
            config["metric_kind"] = "none"

        else:
            # Package callable definition
            impl_spec = ctx.package_symbol(metric)

            config["metric_kind"] = "packaged"
            config["metric_class"] = asdict(impl_spec)
            config["metric_kwargs"] = getattr(metric, "get_config", dict)()

        # Save config to file
        path = self._write_json(config, Path(save_dir) / self.config_rel_path)
        return {"config": path.name}

    def encode_state(
        self,
        obj: SimilarityCondition,
        save_dir: Path,
        *,
        ctx: SaveContext,
    ) -> dict[str, str]:
        """
        Encodes object state to a pickle file.

        Args:
            obj (T):
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
    # Object decoding
    # ================================================
    def decode(
        self,
        cls: type[SimilarityCondition],
        load_dir: Path,
        *,
        ctx: LoadContext,
    ) -> SimilarityCondition:
        """
        Decodes an object from a saved artifact.

        Description:
            Instantiates an object (instantiates from config and sets state).

        Args:
            cls (type[SimilarityCondition]):
                Load config for class.
            load_dir (Path):
                Directory to decode from.
            ctx (LoadContext, optional):
                Additional de-serialization context.

        Returns:
            SimilarityCondition: The re-instantiated object.

        """
        config = self.decode_config(load_dir=load_dir, ctx=ctx)

        # Some conditions use custom metrics
        # We may need to load packaged code
        metric_kind = config.pop("metric_kind")
        metric = None
        if metric_kind == "packaged":
            impl_spec = SymbolSpec(**config.pop("metric_class"))
            metric = symbol_registry.resolve_symbol(
                spec=impl_spec,
                allow_packaged_code=ctx.allow_packaged_code,
                packaged_code_loader=lambda source_ref: default_packaged_code_loader(
                    artifact_path=ctx.artifact_path,
                    source_ref=source_ref,
                    allow_packaged=ctx.allow_packaged_code,
                ),
            )
            # If metric is a class (not a function), init with kwargs
            metric_kwargs = config.pop("metric_kwargs", {})
            if inspect.isclass(metric):
                metric = metric(**metric_kwargs)

        elif metric_kind != "none":
            msg = f"Unknown metric_kind: {metric_kind}"
            raise ValueError(msg)

        # Restore condition with metric (if defined)
        obj = cls(metric=metric, **config)

        if hasattr(obj, "set_state"):
            state = self.decode_state(load_dir=load_dir, ctx=ctx)
            obj.set_state(**state)
        return obj

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
        # Check that config.json exists
        file_config = Path(load_dir) / self.config_rel_path
        if not file_config.exists():
            msg = f"Could not find config file in directory: '{file_config}'."
            raise FileNotFoundError(msg)

        # Read config
        config = self._read_json(file_config)
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
